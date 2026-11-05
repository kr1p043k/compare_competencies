"""Smoke-проверка после деплоя: минимум, который должен работать всегда.

Проверяет живость, готовность, логин и один авторизованный запрос.
Не меняет данные: логин только создаёт сессию, чтение — только GET.

Использование:
    python scripts/smoke.py --base http://127.0.0.1:8000 --email admin@example.local
Пароль берётся из --password или из SMOKE_PASSWORD (не логируется).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

TIMEOUT = 30


def call(base: str, path: str, token: str | None = None, body: dict | None = None):
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:200]
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-deploy smoke check")
    ap.add_argument("--base", default=os.getenv("SMOKE_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--email", default=os.getenv("SMOKE_EMAIL", ""))
    ap.add_argument("--password", default=os.getenv("SMOKE_PASSWORD", ""))
    args = ap.parse_args()

    failures = 0

    def report(ok: bool, name: str, detail: str = "") -> None:
        nonlocal failures
        print(f"[{'OK  ' if ok else 'FAIL'}] {name:<28} {detail}")
        if not ok:
            failures += 1

    status, body = call(args.base, "/health")
    report(status == 200, "/health", f"{status}")

    status, ready = call(args.base, "/ready")
    ok_ready = status in (200, 503) and isinstance(ready, dict) and "status" in ready
    report(ok_ready, "/ready", f"{status}")
    if status == 503:
        print("       (сервис поднят, но не готов: проверьте зависимости)")

    status, body = call(args.base, "/api/auth/login",
                        body={"email": args.email, "password": args.password}) if args.email else (0, None)
    if not args.email:
        print("[SKIP] login                       не задан --email")
        token = None
    elif status == 200 and isinstance(body, dict) and body.get("token"):
        report(True, "login", f"role={body.get('role')}")
        token = body["token"]
    else:
        report(False, "login", f"{status}")
        token = None

    if token:
        status, body = call(args.base, "/api/profiles", token=token)
        report(status == 200, "/api/profiles (auth)", f"{status}")
        status, body = call(args.base, "/api/taxonomy/categories", token=token)
        report(status == 200, "/api/taxonomy/categories", f"{status}")

    status, body = call(args.base, "/metrics")
    report(status in (200, 401, 403), "/metrics reachable", f"{status}")

    print(f"\nИтого проблем: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
