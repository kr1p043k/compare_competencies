"""Stability of composite indices under weight perturbation (no labels needed).

Checks that Match Score *ranking* is robust to ±0.10 weight changes even though
absolute values shift. Run: python scripts/check_metric_stability.py
"""
import itertools

import numpy as np
from scipy.stats import spearmanr


def scores(skill, domain, strong, weak, w):
    wms, wdm, wmk, wst, www = w
    market = wms * skill + wdm * domain
    readiness = min(100, max(0, wmk * market + wst * strong - www * weak))
    return (skill + market + readiness) / 3


def main() -> None:
    base = (0.60, 0.40, 0.45, 0.30, 0.25)
    grid = list(itertools.product([10, 30, 50, 70, 90], repeat=4))
    base_vals = np.array([scores(s, d, st, w, base) for s, d, st, w in grid])
    rhos, mads = [], []
    rng = np.random.default_rng(42)
    for _ in range(200):
        pert = tuple(sorted(max(0.05, min(0.95, b + rng.uniform(-0.1, 0.1))) for b in base))
        v = np.array([scores(s, d, st, w, pert) for s, d, st, w in grid])
        rhos.append(spearmanr(base_vals, v).statistic)
        mads.append(float(np.mean(np.abs(v - base_vals))))
    print(f"points={len(grid)} variants=200")
    print(f"spearman: mean={np.mean(rhos):.4f} min={np.min(rhos):.4f}")
    print(f"mean|delta| Match Score points: {np.mean(mads):.2f} (scale 0..100)")


if __name__ == "__main__":
    main()
