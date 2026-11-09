"""Calibration monitoring for the shipped LTR ranker (log-loss / Brier).

The ranker emits query-relative scores: fine for ordering, meaningless as an
absolute probability. This script measures how well a monotonic map
(raw score -> P(grade>=1)) would calibrate on the frozen split, and writes a
small report.

IMPORTANT - the calibrator is NOT served. ``src/`` has no reference to any
``ranker_calibrator*.joblib``: the engine ranks raw scores (order-preserving)
and never exposes a probability. This report is monitoring only; wiring a
calibrator into serving is a separate decision.

Protocol (mirrors scripts/experiments/calibrate_ranker.py): per-role z-score the
raw scores, fit Platt (logistic) on the TRAIN frame, score the untouched TEST
frame. The no-skill baseline is a constant predicting the TRAIN positive rate.
Ranking order is unchanged by any monotonic map, so NDCG is unaffected.

Usage:
    python scripts/check_calibration.py
    python scripts/check_calibration.py --out data/reports/calibration_report.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.ml_eval_common import ROOT, build_frames, znorm_by_query  # noqa: E402


def reliability(y_true: np.ndarray, proba: np.ndarray, bins: int = 5) -> list[dict]:
    """Quantile-binned calibration curve: mean prediction vs observed rate."""
    edges = np.quantile(proba, np.linspace(0, 1, bins + 1))
    out: list[dict] = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (proba >= lo) & (proba <= hi if b == bins - 1 else proba < hi)
        if not mask.any():
            continue
        out.append(
            {
                "mean_pred": round(float(proba[mask].mean()), 3),
                "frac_pos": round(float(y_true[mask].mean()), 3),
                "n": int(mask.sum()),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=None)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--out", default=str(ROOT / "data" / "reports" / "calibration_report.json"))
    args = parser.parse_args()

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
    frames = build_frames(gold_path=args.gold, artifact=args.artifact)
    z_train = znorm_by_query(frames["scores_train"], frames["roles_train"])
    z_test = znorm_by_query(frames["scores_test"], frames["roles_test"])
    pos_train = (frames["y_train"] >= 1).astype(int)
    pos_test = (frames["y_test"] >= 1).astype(int)

    platt = LogisticRegression().fit(z_train.reshape(-1, 1), pos_train)
    proba = np.clip(platt.predict(z_test.reshape(-1, 1)), 1e-6, 1 - 1e-6)
    base_rate = float(pos_train.mean())

    report = {
        "artifact": frames["artifact"],
        "gold": frames["gold"],
        "split": "per-role stratified 25% holdout, seed 42",
        "n_train": frames["n_train"],
        "n_test": frames["n_test"],
        "served": False,
        "served_note": (
            "The calibrator is NOT served: src/ has no ranker_calibrator reference; "
            "the engine ranks raw query-relative scores and exposes no probability."
        ),
        "candidates": {
            "platt": {
                "log_loss": round(float(log_loss(pos_test, proba)), 4),
                "brier": round(float(brier_score_loss(pos_test, proba)), 4),
                "reliability": reliability(pos_test, proba),
            },
            "base_rate": {
                "log_loss": round(float(log_loss(pos_test, np.full_like(proba, base_rate))), 4),
                "brier": round(float(brier_score_loss(pos_test, np.full_like(proba, base_rate))), 4),
            },
        },
        "train_positive_rate": round(base_rate, 4),
    }
    candidates = report["candidates"]
    report["winner"] = min(candidates, key=lambda name: candidates[name]["log_loss"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    platt_metrics = report["candidates"]["platt"]
    base_metrics = report["candidates"]["base_rate"]
    print(f"artifact={frames['artifact']} n_test={frames['n_test']} served=False")
    print(f"platt     log_loss={platt_metrics['log_loss']} brier={platt_metrics['brier']}")
    print(f"base_rate log_loss={base_metrics['log_loss']} brier={base_metrics['brier']}")
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
