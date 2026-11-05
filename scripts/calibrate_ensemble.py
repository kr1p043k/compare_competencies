"""Honest ensemble calibrator: isotonic fit on ensemble-OOF z-scores
(500 rows, each scored by models that never saw its role). No train reuse.
Overwrites data/models/ranker_calibrator_ensemble.joblib"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
oof = json.load(open(ROOT / "labeling" / "loro_ensemble_oof.json", encoding="utf-8"))
z, yb = [], []
for rl, d in oof.items():
    z.extend(d["ens_z"])
    yb.extend([1 if g >= 1 else 0 for g in d["grade"]])
z, yb = np.asarray(z), np.asarray(yb)
iso = IsotonicRegression(out_of_bounds="clip").fit(z, yb)
p = np.asarray(iso.predict(z))
print("oof_isotonic_ens ll=", round(float(log_loss(yb, p)), 4),
      "brier=", round(float(brier_score_loss(yb, p)), 4))
qs = np.quantile(p, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
for b in range(5):
    m = (p >= qs[b]) & (p <= qs[b + 1])
    print(f"bin{b}: pred={p[m].mean():.3f} frac={yb[m].mean():.3f} n={m.sum()}")
joblib.dump({"kind": "isotonic_on_ensemble_oof_z",
             "model": iso,
             "note": "input = z-averaged 3-seed ensemble score (per-query z-norm at serving)"},
            ROOT / "data" / "models" / "ranker_calibrator_ensemble.joblib")
print("saved honest ensemble calibrator")
