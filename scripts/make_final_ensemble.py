"""Final serving ensemble: canonical gold + role feats, FIXED split (42)."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from sklearn.metrics import ndcg_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402
from src import Ok  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402
from src.parsing.api.embedding_loader import get_embedding_model  # noqa: E402
from scripts.train_ltr_ranker import load_vacancies, read_gold  # noqa: E402
from unittest.mock import patch  # noqa: E402

ROOT = Path("C:/forCode/compare_online")
CANON = Path("C:/Users/максим/workvs/compare_competencies/labeling/gold.csv")

rows = read_gold(CANON)
roles = sorted({r["role"] for r in rows})
vacancies = asyncio.run(load_vacancies(3000))
with patch("src.predictors.ltr_recommendation_engine.get_embedding_model",
           return_value=get_embedding_model()):
    eng = LTRRecommendationEngine()
    for s in (42, 43, 44):
        res = eng.fit_ranker(rows, vacancies, seed=s, split_seed=42,
                             model_out=ROOT / "temp" / f"fin_ens_s{s}.joblib")
        m = res.value if hasattr(res, "value") else res.unwrap()
        print(f"seed {s}: ndcg={m['ndcg@10']} p={m['p@10']}", flush=True)
    # common frame (split 42) from last state
    train_idx, test_idx = [], []
    for role in roles:
        idx = [i for i, r in enumerate(rows) if r["role"] == role]
        grades = [rows[i]["grade"] for i in idx]
        tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(idx, grades))
        train_idx.extend(idx[i] for i in tr)
        test_idx.extend(idx[i] for i in te)
    train_core: dict[str, list[str]] = {r: [] for r in roles}
    for j in train_idx:
        if rows[j]["grade"] == 2:
            train_core[rows[j]["role"]].append(rows[j]["skill"])
    market_emb = np.mean(list(eng.skill_embeddings.values()), axis=0)
    qid_of = {r: k for k, r in enumerate(roles)}
    Xte, yte, _, _, test_order = eng._build_ranking_frame(test_idx, rows, train_core, market_emb, qid_of)
    troles = [rows[j]["role"] for j in test_order]
    P = [np.asarray(joblib.load(ROOT / "temp" / f"fin_ens_s{s}.joblib")["model"].predict(Xte))
         for s in (42, 43, 44)]
    zavg = np.mean([(p - p.mean()) / (p.std() or 1.0) for p in P], axis=0)

    def qndcg(yt, sc, rl):
        ns = []
        for r in sorted(set(rl)):
            m = np.array([t == r for t in rl])
            a, b = np.asarray(yt)[m], np.asarray(sc)[m]
            if len(a) < 2 or np.all(a == a[0]):
                continue
            try:
                v = ndcg_score(a.reshape(1, -1), b.reshape(1, -1), k=min(10, len(a)))
            except Exception:
                continue
            if not np.isnan(v):
                ns.append(float(v))
        return round(float(np.mean(ns)), 4)

    print("FINAL ENSEMBLE test_ndcg=", qndcg(yte, zavg, troles))
    bundle = {"kind": "xgb_ranker_ensemble_canon_rolefeat",
              "seeds": [42, 43, 44], "split_seed": 42,
              "models": [joblib.load(ROOT / "temp" / f"fin_ens_s{s}.joblib")["model"] for s in (42, 43, 44)],
              "feature_names": eng.feature_names,
              "skill_metadata": eng.skill_metadata,
              "skill_embeddings": eng.skill_embeddings,
              "vacancy_skills_corpus": eng.vacancy_skills_corpus,
              "category_avg_weight": eng.category_avg_weight,
              "total_vacancies": eng.total_vacancies,
              "role_skill_counts": eng.role_skill_counts,
              "role_vac_totals": eng.role_vac_totals,
              "serving": "per-query z-score each model, average, rank"}
    joblib.dump(bundle, ROOT / "data" / "models" / "ltr_ranker_ensemble_final.joblib")
    print("saved final ensemble")
