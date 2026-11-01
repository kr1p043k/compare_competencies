"""Light HPO for fit_ranker on the main split; winner confirmed on LORO-macro.

Grid is intentionally small (6 configs): test reuse is the main overfit
risk, so adoption requires a LORO-macro win, not just a test win.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from src import Ok  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402
from scripts.train_ltr_ranker import load_vacancies, read_gold  # noqa: E402
from unittest.mock import patch  # noqa: E402
from src.parsing.api.embedding_loader import get_embedding_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID = [
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.08},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03},
]


def loro_macro(eng, rows, roles, vacancies, **kw) -> float:
    from sklearn.metrics import ndcg_score
    ns = []
    for held in roles:
        sub = [g for g in rows if g["role"] != held]
        res = eng.fit_ranker(sub, vacancies, seed=42,
                             model_out=ROOT / "temp" / "hpo_loro.joblib", **kw)
        assert isinstance(res, Ok), res
        market_emb = np.mean(list(eng.skill_embeddings.values()), axis=0)
        rows_h = [g for g in rows if g["role"] == held]
        Xh, yh, _, _, _ = eng._build_ranking_frame(
            list(range(len(rows_h))), rows_h, {}, market_emb, {held: 0})
        pred = np.asarray(eng.model.predict(Xh))
        yt = np.asarray(yh)
        try:
            v = ndcg_score(yt.reshape(1, -1), pred.reshape(1, -1), k=min(10, len(yt)))
        except Exception:
            v = 0.0
        ns.append(float(v) if not np.isnan(v) else 0.0)
    return round(float(np.mean(ns)), 4)


def main() -> None:
    rows = read_gold(ROOT / "labeling" / "gold.csv")
    roles = sorted({g["role"] for g in rows})
    vacancies = asyncio.run(load_vacancies(3000))
    with patch("src.predictors.ltr_recommendation_engine.get_embedding_model",
               return_value=get_embedding_model()):
        eng = LTRRecommendationEngine()
        scored = []
        for i, kw in enumerate(GRID):
            res = eng.fit_ranker(rows, vacancies, seed=42,
                                 model_out=ROOT / "temp" / f"hpo_{i}.joblib", **kw)
            assert isinstance(res, Ok), res
            m = res.unwrap() if hasattr(res, "unwrap") else res.value
            scored.append((m["ndcg@10"], kw))
            print(f"cfg{i} {kw} -> test_ndcg={m['ndcg@10']}")
        scored.sort(reverse=True)
        print("RANK:", [(s, k) for s, k in scored])
        base_test = [s for s, k in scored if k == GRID[0]][0]
        for s, kw in scored[:2]:
            if kw == GRID[0]:
                continue
            macro = loro_macro(eng, rows, roles, vacancies, **kw)
            print(f"confirm {kw}: test={s} loro_macro={macro} (baseline loro=0.4393)")
            break
        else:
            print(f"baseline wins on test ({base_test}); no change")


if __name__ == "__main__":
    main()
