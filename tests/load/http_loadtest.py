"""Lightweight async HTTP load test for the Compare Competencies API.

Why not k6
----------
`.github/workflows/load-test.yml` targets `tests/load/k6_script.js` with the k6
binary, but that script does not exist in the repo and k6 is not installed on
the dev machine, so the workflow is not runnable offline (schedule is disabled
there for the same reason). This script is the runnable, dependency-light
replacement: it needs only `httpx` (already a project dependency) and stdlib.

What it does
------------
Hits three read-only endpoints concurrently and reports p50/p95/p99, RPS and
error rate:

* ``GET /health``                      - baseline, no DB
* ``GET /api/vacancies?limit=20``      - DB-backed, requires a token
* ``GET /api/taxonomy/categories``     - in-memory taxonomy, requires a token

Credentials come from the ``LOADTEST_EMAIL`` / ``LOADTEST_PASSWORD`` env vars
(email defaults to ``DEFAULT_USER``). If ``LOADTEST_PASSWORD`` is unset, a local
``users.json`` (local/optional, not committed) is read instead; if that file is
absent too, the script exits with an error. The password is never printed.
The only write the script performs is that single login POST (which inserts a
session row); every measured request is a GET, so a shared DB is safe.

Usage
-----
    python tests/load/http_loadtest.py --base-url http://127.0.0.1:8099 \
        --duration 30 --concurrency 10

Idempotent: re-running only creates one more session row per run.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
USERS_FILE = REPO_ROOT / "users.json"

DEFAULT_ENDPOINTS: list[tuple[str, str, bool]] = [
    ("GET", "/health", False),
    ("GET", "/api/vacancies?limit=20", True),
    ("GET", "/api/taxonomy/categories", True),
]

DEFAULT_USER = "teacher@compare-competencies.local"


@dataclass
class Outcome:
    """Per-worker accumulator (no shared lock needed - merged after gather)."""

    latencies_ms: list[float] = field(default_factory=list)
    ok_latencies_ms: list[float] = field(default_factory=list)
    status_counts: dict[int, int] = field(default_factory=dict)
    exceptions: int = 0

    def merge(self, other: Outcome) -> None:
        self.latencies_ms.extend(other.latencies_ms)
        self.ok_latencies_ms.extend(other.ok_latencies_ms)
        self.exceptions += other.exceptions
        for code, count in other.status_counts.items():
            self.status_counts[code] = self.status_counts.get(code, 0) + count


@dataclass
class EndpointResult:
    endpoint: str
    method: str
    concurrency: int
    duration_s: float
    total: int
    ok: int
    rate_limited: int
    errors: int
    exceptions: int
    rps: float
    p50: float
    p95: float
    p99: float
    error_rate: float
    status_counts: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "concurrency": self.concurrency,
            "duration_s": round(self.duration_s, 3),
            "total_requests": self.total,
            "rps": round(self.rps, 2),
            "ok_2xx": self.ok,
            "rate_limited_429": self.rate_limited,
            "errors_non_2xx": self.errors,
            "exceptions": self.exceptions,
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "error_rate": round(self.error_rate, 4),
            "status_counts": self.status_counts,
        }


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in 0..100). NaN on empty input."""
    if not values:
        return float("nan")
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(data) - 1)
    frac = rank - lower
    return data[lower] + (data[upper] - data[lower]) * frac


async def _login(client: httpx.AsyncClient, base_url: str, email: str) -> str:
    """Log in once and return the bearer token. Never prints the password."""
    password = os.environ.get("LOADTEST_PASSWORD")
    if password is None:
        if not USERS_FILE.exists():
            raise SystemExit(
                "set LOADTEST_PASSWORD or provide a local users.json "
                f"(looked for users.json at {USERS_FILE})"
            )
        users = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        if email not in users:
            raise SystemExit(f"user {email!r} not present in users.json")
        password = users[email]["password"]
    payload = {"email": email, "password": password}
    for attempt in range(3):
        resp = await client.post(f"{base_url}/api/auth/login", json=payload)
        if resp.status_code == 429:
            wait = 20 * (attempt + 1)
            print(f"login rate-limited (429), retry in {wait}s ...", file=sys.stderr)
            await asyncio.sleep(wait)
            continue
        if resp.status_code != 200:
            raise SystemExit(f"login failed: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        if "token" not in data:
            raise SystemExit(f"login returned no token (2FA?): {list(data)}")
        return data["token"]
    raise SystemExit("login failed after retries (rate limit)")


async def _worker(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    deadline: float,
    out: Outcome,
) -> None:
    while time.perf_counter() < deadline:
        start = time.perf_counter()
        try:
            resp = await client.request(method, url, headers=headers)
        except Exception:  # noqa: BLE001 - transport errors are part of the metric
            out.exceptions += 1
            out.latencies_ms.append((time.perf_counter() - start) * 1000)
            continue
        latency_ms = (time.perf_counter() - start) * 1000
        out.latencies_ms.append(latency_ms)
        if 200 <= resp.status_code < 300:
            out.ok_latencies_ms.append(latency_ms)
        out.status_counts[resp.status_code] = out.status_counts.get(resp.status_code, 0) + 1


async def run_endpoint(
    client: httpx.AsyncClient,
    base_url: str,
    method: str,
    path: str,
    headers: dict[str, str],
    duration: float,
    concurrency: int,
    warmup: int,
) -> EndpointResult:
    url = base_url + path
    for _ in range(warmup):
        with contextlib.suppress(Exception):
            await client.request(method, url, headers=headers)

    outcomes = [Outcome() for _ in range(concurrency)]
    deadline = time.perf_counter() + duration
    wall_start = time.perf_counter()
    await asyncio.gather(
        *[_worker(client, method, url, headers, deadline, out) for out in outcomes]
    )
    wall = time.perf_counter() - wall_start

    merged = Outcome()
    for out in outcomes:
        merged.merge(out)

    total = len(merged.latencies_ms)
    # Percentiles are computed over 2xx responses only: when the rate limiter
    # answers 429 in ~1 ms it would otherwise hide the real DB-backed latency.
    latency_sample = merged.ok_latencies_ms or merged.latencies_ms
    errors = total - len(merged.ok_latencies_ms)
    return EndpointResult(
        endpoint=path,
        method=method,
        concurrency=concurrency,
        duration_s=wall,
        total=total,
        ok=len(merged.ok_latencies_ms),
        rate_limited=merged.status_counts.get(429, 0),
        errors=errors,
        exceptions=merged.exceptions,
        rps=total / wall if wall > 0 else 0.0,
        p50=percentile(latency_sample, 50),
        p95=percentile(latency_sample, 95),
        p99=percentile(latency_sample, 99),
        error_rate=errors / total if total else 0.0,
        status_counts={str(k): v for k, v in sorted(merged.status_counts.items())},
    )


def _print_table(results: list[EndpointResult], base_url: str, limit_note: bool) -> None:
    print()
    print(f"base_url={base_url}  (percentiles over 2xx only)")
    print("-" * 104)
    header = (
        f"{'endpoint':<34}{'conc':>5}{'RPS':>8}{'p50':>9}{'p95':>9}"
        f"{'p99':>9}{'err%':>8}{'429':>7}{'total':>8}"
    )
    print(header)
    print("-" * 104)
    for r in results:
        print(
            f"{r.endpoint:<34}{r.concurrency:>5}{r.rps:>8.1f}"
            f"{r.p50:>9.1f}{r.p95:>9.1f}{r.p99:>9.1f}"
            f"{r.error_rate * 100:>7.1f}%{r.rate_limited:>7}{r.total:>8}"
        )
    print("-" * 104)
    if limit_note:
        print("NOTE: 429 responses come from slowapi per-IP limits "
              "(60/min on these endpoints). Run with RATELIMIT_ENABLED=false "
              "to measure raw server capacity.")
    print()


async def amain(args: argparse.Namespace) -> int:
    headers: dict[str, str] = {}
    limits = httpx.Limits(
        max_connections=args.concurrency * 4,
        max_keepalive_connections=args.concurrency * 2,
    )
    async with httpx.AsyncClient(
        timeout=args.timeout, limits=limits, trust_env=False
    ) as client:
        if any(auth for _, _, auth in args.endpoints):
            token = await _login(client, args.base_url, args.user)
            headers = {"Authorization": f"Bearer {token}"}

        # readiness probe
        try:
            await client.get(f"{args.base_url}/health")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"cannot reach {args.base_url}/health: {exc}") from exc

        results: list[EndpointResult] = []
        for method, path, needs_auth in args.endpoints:
            hdrs = headers if needs_auth else {}
            result = await run_endpoint(
                client, args.base_url, method, path, hdrs,
                args.duration, args.concurrency, args.warmup,
            )
            results.append(result)

    _print_table(results, args.base_url, limit_note=True)
    summary = {
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "duration_s": args.duration,
        "results": [r.as_dict() for r in results],
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON written to {out_path}")
    else:
        print(json.dumps(summary, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async HTTP load test (httpx)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8099")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds per endpoint")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup requests per endpoint")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--user",
        default=os.environ.get("LOADTEST_EMAIL", DEFAULT_USER),
        help="login email (env LOADTEST_EMAIL, default DEFAULT_USER)",
    )
    parser.add_argument("--json-out", default=None, help="write JSON summary to this path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.endpoints = DEFAULT_ENDPOINTS
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
