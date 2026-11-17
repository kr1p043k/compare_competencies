"""Drift / NDCG monitoring job for the shipped LTR ranker (BACKLOG 6.6.2).

One entrypoint that evaluates BOTH health signals and emits a machine-readable
verdict (JSON to stdout and, with ``--output``, to a file) for a scheduler or an
n8n webhook:

  a) NDCG@10 drop  - current NDCG (live scoring of the frozen serving split, or
     read from ``--ndcg-current``) against a stored baseline. Baseline lookup
     order: ``--ndcg-baseline`` file, then the artifact sidecar
     (``scripts.check_model_manifests.collect_sidecar``), then the frozen report.
  b) Feature drift - PSI / KS / mean-shift z per feature against the stored
     baseline of ``scripts.check_feature_drift.py`` (same statistics, reused).

Exit codes (n8n / cron contract): ``0`` ok, ``1`` alert, ``2`` missing inputs
(a verdict with status=alert is still printed, so monitors can render it).

Verdict contract::

    {
      "status": "ok" | "alert",
      "reasons": ["ndcg 0.6500 < baseline 0.7455 - 0.05", "feat: psi 0.312>0.25"],
      "metrics": {
        "ndcg": {"current": 0.65, "baseline": 0.7455, "drop": 0.0955,
                  "tolerance": 0.05},
        "feature_drift": {"max_ks": 0.11, "max_psi": 0.31, "max_mean_shift_z": 0.4,
                           "features": [{"name": "...", "ks": ..., "psi": ...,
                                          "mean_shift_z": ...}]}
      },
      "thresholds": {"ndcg_drop": 0.05, "ks": 0.25, "psi": 0.25,
                      "mean_shift_z": 1.0},
      "artifact": "ltr_ranker_ensemble_human.joblib",
      "checked_at": "2026-11-18T00:00:00+00:00"
    }

``--post-url`` additionally POSTs the verdict as JSON to an n8n webhook (a
dedicated workflow, NOT ``trend_alert.json`` - that one ships the market Excel
report; see docs note in the module README).

Usage:
    python scripts/monitor_model_health.py
    python scripts/monitor_model_health.py --output data/reports/model_health_verdict.json
    python scripts/monitor_model_health.py --ndcg-current data/reports/frozen_eval.json
    python scripts/monitor_model_health.py --post-url https://n8n/webhook/model-health
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.check_feature_drift import DEFAULT_BASELINE, DEFAULT_KS, DEFAULT_PSI, DEFAULT_Z, ks_stat, psi
from scripts.check_model_manifests import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORT,
    collect_sidecar,
    frozen_ndcg,
    pick_ndcg,
    read_json,
)

# ---- explicit thresholds (single source of truth for the job) ----------------
DEFAULT_NDCG_DROP_TOLERANCE = 0.05  # alert if baseline_ndcg - current > this
NDCG_EPSILON = 1e-9  # float noise guard: exact baseline equality must stay ok
DEFAULT_POST_TIMEOUT = 10.0

STATUS_OK = "ok"
STATUS_ALERT = "alert"


def _build_frames(gold: str | None = None, artifact: str | None = None) -> dict[str, Any]:
    """Deferred heavy import: pulls the real serving frame (engine + sklearn)."""
    from scripts.ml_eval_common import build_frames

    return build_frames(
        gold_path=Path(gold) if gold else None,
        artifact=Path(artifact) if artifact else None,
    )


def live_ndcg(frames: dict[str, Any]) -> float | None:
    """NDCG@10 of the shipped artifact on the frozen split (same code as eval_frozen)."""
    from eval_frozen import per_role_metrics

    metrics = per_role_metrics(frames["y_test"], frames["scores_test"], frames["roles_test"])
    value = metrics.get("ndcg@10")
    return float(value) if isinstance(value, (int, float)) else None


def resolve_baseline_ndcg(
    artifact_name: str,
    baseline_path: Path,
    models_dir: Path,
    report: Any,
) -> tuple[float | None, str]:
    """Baseline NDCG: explicit baseline file, then artifact sidecar, then frozen report."""
    stored = read_json(baseline_path)
    if isinstance(stored, dict):
        value = pick_ndcg(stored)
        if value is None and isinstance(stored.get("results"), dict):
            value = frozen_ndcg(stored, artifact_name)
        if value is not None:
            return value, str(baseline_path)
    _sidecar, _names, sidecar_ndcg = collect_sidecar(models_dir / artifact_name)
    if sidecar_ndcg is not None:
        return sidecar_ndcg, f"sidecar of {artifact_name}"
    report_ndcg = frozen_ndcg(report, artifact_name)
    if report_ndcg is not None:
        return report_ndcg, "frozen report"
    return None, "none"


def current_ndcg_from_report(report_path: Path, artifact_name: str) -> float | None:
    """Current NDCG from a JSON file: either {results: {name: {...}}} or a flat mapping."""
    data = read_json(report_path)
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("results"), dict):
        return frozen_ndcg(data, artifact_name)
    return pick_ndcg(data)


def check_ndcg_drop(
    baseline: float | None,
    current: float | None,
    tolerance: float = DEFAULT_NDCG_DROP_TOLERANCE,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "current": current,
        "baseline": baseline,
        "tolerance": tolerance,
    }
    if baseline is None:
        reasons.append("ndcg baseline missing - run scripts/check_model_manifests.py --write")
        return reasons, metrics
    if current is None:
        reasons.append("ndcg current value unavailable")
        return reasons, metrics
    drop = baseline - current
    metrics["drop"] = round(drop, 6)
    if drop > tolerance + NDCG_EPSILON:
        reasons.append(
            f"ndcg {current:.4f} < baseline {baseline:.4f} - tolerance {tolerance} (drop {drop:.4f})"
        )
    return reasons, metrics


def check_feature_drift(
    x_test: Any,
    baseline: dict[str, Any],
    ks_thr: float,
    psi_thr: float,
    z_thr: float,
) -> tuple[list[str], dict[str, Any]]:
    """PSI/KS/mean-shift against the check_feature_drift baseline (same stats)."""
    reasons: list[str] = []
    ref_features = baseline.get("features", {})
    per_feature: list[dict[str, Any]] = []
    max_ks = max_psi = max_z = 0.0
    for name in x_test.columns:
        ref_entry = ref_features.get(name)
        if not ref_entry or "values" not in ref_entry:
            reasons.append(f"{name}: missing baseline")
            continue
        reference = np.asarray(ref_entry["values"], dtype=float)
        current = np.asarray(x_test[name].to_numpy(), dtype=float)
        ks = ks_stat(reference, current)
        psi_val = psi(reference, current)
        ref_std = float(ref_entry["std"]) or 1.0
        shift = abs(float(current.mean()) - float(ref_entry["mean"])) / ref_std
        max_ks, max_psi, max_z = max(max_ks, ks), max(max_psi, psi_val), max(max_z, shift)
        per_feature.append(
            {
                "name": name,
                "ks": round(ks, 4),
                "psi": round(psi_val, 4),
                "mean_shift_z": round(shift, 4),
            }
        )
        trips = []
        if ks > ks_thr:
            trips.append(f"ks {ks:.3f}>{ks_thr}")
        if psi_val > psi_thr:
            trips.append(f"psi {psi_val:.3f}>{psi_thr}")
        if shift > z_thr:
            trips.append(f"mean_shift_z {shift:.3f}>{z_thr}")
        if trips:
            reasons.append(f"{name}: " + ", ".join(trips))
    return reasons, {
        "max_ks": round(max_ks, 4),
        "max_psi": round(max_psi, 4),
        "max_mean_shift_z": round(max_z, 4),
        "features": per_feature,
    }


def evaluate(
    x_test: Any,
    feature_baseline: dict[str, Any],
    baseline_ndcg: float | None,
    current_ndcg_value: float | None,
    ks_thr: float = DEFAULT_KS,
    psi_thr: float = DEFAULT_PSI,
    z_thr: float = DEFAULT_Z,
    ndcg_tol: float = DEFAULT_NDCG_DROP_TOLERANCE,
) -> dict[str, Any]:
    """Full verdict on in-memory inputs (the unit-tested core)."""
    ndcg_reasons, ndcg_metrics = check_ndcg_drop(baseline_ndcg, current_ndcg_value, ndcg_tol)
    drift_reasons, drift_metrics = check_feature_drift(x_test, feature_baseline, ks_thr, psi_thr, z_thr)
    reasons = ndcg_reasons + drift_reasons
    return {
        "status": STATUS_ALERT if reasons else STATUS_OK,
        "reasons": reasons,
        "metrics": {"ndcg": ndcg_metrics, "feature_drift": drift_metrics},
        "thresholds": {
            "ndcg_drop": ndcg_tol,
            "ks": ks_thr,
            "psi": psi_thr,
            "mean_shift_z": z_thr,
        },
        "checked_at": datetime.now(UTC).isoformat(),
    }


def post_verdict(payload: dict[str, Any], url: str, timeout: float = DEFAULT_POST_TIMEOUT) -> str:
    """POST the verdict JSON to an n8n webhook; raises URLError/OSError on failure."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # urlopen already raises HTTPError on non-2xx; this only guards exotic codes.
        status = int(getattr(response, "status", 200))
        if not 200 <= status < 300:
            raise OSError(f"webhook {url} responded {status}")
        return f"http {status}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Drift/NDCG monitoring job (BACKLOG 6.6.2)")
    parser.add_argument("--gold", default=None)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--ndcg-baseline", default=str(DEFAULT_REPORT.parent / "ndcg_baseline.json"))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--ndcg-current",
        default=None,
        help="JSON with the current NDCG (skips the live eval pass; e.g. a fresh frozen_eval.json)",
    )
    parser.add_argument("--ks-threshold", type=float, default=DEFAULT_KS)
    parser.add_argument("--psi-threshold", type=float, default=DEFAULT_PSI)
    parser.add_argument("--z-threshold", type=float, default=DEFAULT_Z)
    parser.add_argument("--ndcg-drop-tolerance", type=float, default=DEFAULT_NDCG_DROP_TOLERANCE)
    parser.add_argument("--output", default=None, help="also write the verdict JSON here")
    parser.add_argument("--post-url", default=None, help="POST the verdict JSON to this webhook")
    args = parser.parse_args()

    with contextlib.suppress(Exception):
        cast("Any", sys.stdout).reconfigure(encoding="utf-8", errors="replace")

    baseline_path = Path(args.baseline)
    ndcg_baseline_path = Path(args.ndcg_baseline)

    frames = _build_frames(args.gold, args.artifact)
    artifact_name = str(frames["artifact"])

    missing: list[str] = []
    if not baseline_path.exists():
        missing.append(f"feature baseline missing at {baseline_path} - run check_feature_drift.py --write-baseline")
    if args.ndcg_current and not Path(args.ndcg_current).exists():
        missing.append(f"--ndcg-current file not found: {args.ndcg_current}")
    if missing:
        verdict: dict[str, Any] = {
            "status": STATUS_ALERT,
            "reasons": missing,
            "metrics": {},
            "thresholds": {},
            "artifact": artifact_name,
            "checked_at": datetime.now(UTC).isoformat(),
        }
        _emit(verdict, args.output)
        return 2

    feature_baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    baseline_value, baseline_source = resolve_baseline_ndcg(
        artifact_name, ndcg_baseline_path, Path(args.models_dir), read_json(Path(args.report))
    )
    if args.ndcg_current:
        current_value = current_ndcg_from_report(Path(args.ndcg_current), artifact_name)
    else:
        current_value = live_ndcg(frames)

    verdict = evaluate(
        frames["x_test"],
        feature_baseline,
        baseline_value,
        current_value,
        ks_thr=args.ks_threshold,
        psi_thr=args.psi_threshold,
        z_thr=args.z_threshold,
        ndcg_tol=args.ndcg_drop_tolerance,
    )
    verdict["artifact"] = artifact_name
    metrics = verdict["metrics"]
    metrics["ndcg"]["baseline_source"] = baseline_source

    if args.post_url:
        try:
            print(f"webhook: {post_verdict(verdict, args.post_url)}", file=sys.stderr)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            verdict["reasons"].append(f"alert_delivery_failed: {exc}")
            verdict["status"] = STATUS_ALERT

    _emit(verdict, args.output)
    return 1 if verdict["status"] == STATUS_ALERT else 0


def _emit(verdict: dict[str, Any], output: str | None) -> None:
    text = json.dumps(verdict, ensure_ascii=False, indent=2)
    print(text)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
