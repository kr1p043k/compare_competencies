"""Score saved HPO artifacts on the main test split (no retraining)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from sklearn.metrics import ndcg_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402
from scripts.train_ltr_ranker import read_gold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = ["300/d6/e.05", "500/d6/e.05", "300/d4/e.05", "300/d8/e.03", "200/d5/e.08", "500/d4/e.03"]

rows = read_gold(ROOT / "labeling" / "gold.csv")
roles = sorted({r["role"] for r in rows})
train_idx, test_idx = [], []
for role in roles:
    idx = [i for i, r in enumerate(rows) if r["role"] == role]
    grades = [rows[i]["grade"] for i in idx]
    tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(idx, grades))
    train_idx.extend(idx[i] for i in tr)
    test_idx.extend(idx[i] for i in te)

with patch("src.predictors.ltr_recommendation_engine.get_embedding_model", return_value=MagicMock()):
    eng = LTRRecommendationEngine()
    for i, name in enumerate(CFG):
        eng.load_model(ROOT / "temp" / f"hpo_{i}.joblib")
        train_core: dict[str, list[str]] = {r: [] for r in roles}
        for j in train_idx:
            if rows[j]["grade"] == 2:
                train_core[rows[j]["role"]].append(rows[j]["skill"])
        market_emb = np.mean(list(eng.skill_embeddings.values()), axis=0)
        qid_of = {r: k for k, r in enumerate(roles)}
        Xte, yte, _, _, test_order = eng._build_ranking_frame(test_idx, rows, train_core, market_emb, qid_of)
        pred = np.asarray(eng.model.predict(Xte))
        troles = [rows[j]["role"] for j in test_order]
        ns = []
        for rl in sorted(set(troles)):
            m = np.array([t == rl for t in troles])
            yt, pt = yte[m], pred[m]
            try:
                v = ndcg_score(yt.reshape(1, -1), pt.reshape(1, -1), k=min(10, len(yt)))
            except Exception:
                continue
            if not np.isnan(v):
                ns.append(float(v))
        print(name, "test_ndcg=", round(float(np.mean(ns)), 4))
