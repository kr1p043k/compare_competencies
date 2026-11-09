"""Lightweight feature-drift monitor for the shipped LTR ranker.

Compares the current feature distribution (on the frozen serving split,
``tests/fixtures/gold_frozen.csv``, seed 42) against a stored baseline and fails
when any feature moves out of band. Three complementary signals per feature:

  * KS   - two-sample Kolmogorov-Smirnov statistic (max CDF gap),
  * PSI  - population stability index over reference-quantile bins,
  * shift- z-score of the mean shift against the reference std.

The frame is produced by the same helper the gate and the calibration report use
(``scripts/ml_eval_common``), so all three see one feature space.

``--write-baseline`` stores the reference (per-feature summary + raw values) in
``data/reports/feature_drift_baseline.json``. Regenerate it after an INTENTIONAL
model or data change (a retrain moves ``cosine_sim`` because embeddings change);
otherwise a real regression or an accidental feature-code change will trip the
check.

Usage:
    python scripts/check_feature_drift.py --write-baseline
    python scripts/check_feature_drift.py
    python scripts/check_feature_drift.py --ks-threshold 0.2 --psi-threshold 0.25
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

from scripts.ml_eval_common import ROOT, build_frames  # noqa: E402

DEFAULT_BASELINE = ROOT / "data" / "reports" / "feature_drift_baseline.json"
DEFAULT_KS = 0.25
DEFAULT_PSI = 0.25
DEFAULT_Z = 1.0
_BINS = 10


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std()), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "p05": round(float(np.quantile(values, 0.05)), 6),
        "p50": round(float(np.quantile(values, 0.50)), 6),
        "p95": round(float(np.quantile(values, 0.95)), 6),
        "values": [float(v) for v in values],
    }


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    """Population stability index over reference-quantile bins.

    Edges are deduplicated: heavily tied features (e.g. skill frequency with many
    zeros) collapse quantile bins, and an empty bin would blow the index up on
    pure rounding noise. When ties leave fewer than two bins, PSI is undefined
    and reported as 0 (KS + mean shift still cover the feature).
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, _BINS + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.clip(np.histogram(reference, bins=edges)[0] / len(reference), 1e-6, None)
    cur_share = np.clip(np.histogram(current, bins=edges)[0] / len(current), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_stat(reference: np.ndarray, current: np.ndarray) -> float:
    try:
        from scipy.stats import ks_2samp
    except Exception:
        ref_sorted = np.sort(reference)
        cur_sorted = np.sort(current)
        grid = np.union1d(ref_sorted, cur_sorted)
        cdf_ref = np.searchsorted(ref_sorted, grid, side="right") / len(ref_sorted)
        cdf_cur = np.searchsorted(cur_sorted, grid, side="right") / len(cur_sorted)
        return float(np.max(np.abs(cdf_ref - cdf_cur)))
    return float(ks_2samp(reference, current).statistic)


def write_baseline(frames: dict, out_path: Path) -> int:
    x_test = frames["x_test"]
    features = {name: summarize(x_test[name].to_numpy()) for name in x_test.columns}
    payload = {
        "artifact": frames["artifact"],
        "gold": frames["gold"],
        "split": "per-role stratified 25% holdout, seed 42 (serving frame)",
        "n": int(frames["n_test"]),
        "bins": _BINS,
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"baseline written: {out_path} ({len(features)} features, n={payload['n']})")
    return 0


def check(frames: dict, baseline_path: Path, ks_thr: float, psi_thr: float, z_thr: float) -> int:
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}; run with --write-baseline first")
        return 2
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    ref_features = baseline.get("features", {})
    x_test = frames["x_test"]

    drifts: list[str] = []
    for name in x_test.columns:
        ref_entry = ref_features.get(name)
        if not ref_entry or "values" not in ref_entry:
            print(f"DRIFT {name:24} no reference in baseline")
            drifts.append(f"{name}: missing baseline")
            continue
        reference = np.asarray(ref_entry["values"], dtype=float)
        current = x_test[name].to_numpy(dtype=float)
        ks = ks_stat(reference, current)
        psi_val = psi(reference, current)
        ref_std = float(ref_entry["std"]) or 1.0
        shift = abs(float(current.mean()) - float(ref_entry["mean"])) / ref_std
        bad = ks > ks_thr or psi_val > psi_thr or shift > z_thr
        flag = "DRIFT" if bad else "ok"
        print(
            f"{flag:5} {name:24} ks={ks:.3f} psi={psi_val:.3f} "
            f"mean_shift_z={shift:.3f} (ref mean={ref_entry['mean']:.4f} "
            f"std={ref_entry['std']:.4f} -> now mean={current.mean():.4f})"
        )
        if bad:
            reasons = []
            if ks > ks_thr:
                reasons.append(f"ks {ks:.3f}>{ks_thr}")
            if psi_val > psi_thr:
                reasons.append(f"psi {psi_val:.3f}>{psi_thr}")
            if shift > z_thr:
                reasons.append(f"mean_shift_z {shift:.3f}>{z_thr}")
            drifts.append(f"{name}: " + ", ".join(reasons))

    if drifts:
        print("\nFAIL: feature drift out of band:")
        for item in drifts:
            print(f"  - {item}")
        return 1
    print("\nOK: no feature drift out of band")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=None)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--ks-threshold", type=float, default=DEFAULT_KS)
    parser.add_argument("--psi-threshold", type=float, default=DEFAULT_PSI)
    parser.add_argument("--z-threshold", type=float, default=DEFAULT_Z)
    args = parser.parse_args()

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frames = build_frames(gold_path=args.gold, artifact=args.artifact)
    print(f"artifact={frames['artifact']} gold={frames['gold']} n_test={frames['n_test']}\n")

    if args.write_baseline:
        return write_baseline(frames, Path(args.baseline))
    return check(
        frames, Path(args.baseline), args.ks_threshold, args.psi_threshold, args.z_threshold
    )


if __name__ == "__main__":
    raise SystemExit(main())
