"""Production ensemble bundle: 3 seed-models (z-averaged at serve time).

Reuses b2_base_s{42,43,44}.joblib (current pipeline). Verifies bundle NDCG
on the main test split, then saves data/models/ltr_ranker_ensemble_human.joblib
Serving: z-score each model's raw scores per query, average, rank.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from sklearn.metrics import ndcg_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from scripts.train_ltr_ranker import read_gold  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)


def main() -> None:
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
        eng.load_model(ROOT / "temp" / "b2_base_s42.joblib")
        train_core: dict[str, list[str]] = {r: [] for r in roles}
        for j in train_idx:
            if rows[j]["grade"] == 2:
                train_core[rows[j]["role"]].append(rows[j]["skill"])
        market_emb = np.mean(list(eng.skill_embeddings.values()), axis=0)
        qid_of = {r: k for k, r in enumerate(roles)}
        Xte, yte, _, _, test_order = eng._build_ranking_frame(
            test_idx, rows, train_core, market_emb, qid_of)
        Xtr, ytr, _, _, _ = eng._build_ranking_frame(
            train_idx, rows, train_core, market_emb, qid_of)
        troles = [rows[j]["role"] for j in test_order]

        models = []
        for s in SEEDS:
            a = joblib.load(ROOT / "temp" / f"b2_base_s{s}.joblib")
            assert a.get("feature_names") == eng.feature_names, f"feature drift s{s}"
            models.append(a["model"])
        Pte = np.array([np.asarray(m.predict(Xte)) for m in models])
        Ptr = np.array([np.asarray(m.predict(Xtr)) for m in models])
        zte = np.mean([(p - p.mean()) / (p.std() or 1.0) for p in Pte], axis=0)
        ztr = np.mean([(p - p.mean()) / (p.std() or 1.0) for p in Ptr], axis=0)

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

        print("ensemble test_ndcg=", qndcg(yte, zte, troles))
        bundle = {"kind": "xgb_ranker_ensemble_human",
                  "seeds": list(SEEDS),
                  "models": models,
                  "feature_names": eng.feature_names,
                  "skill_metadata": eng.skill_metadata,
                  "skill_embeddings": eng.skill_embeddings,
                  "vacancy_skills_corpus": eng.vacancy_skills_corpus,
                  "category_avg_weight": eng.category_avg_weight,
                  "total_vacancies": eng.total_vacancies,
                  "serving": "per-query z-score each model, average, rank (order-preserving)"}
        out = ROOT / "data" / "models" / "ltr_ranker_ensemble_human.joblib"
        joblib.dump(bundle, out)
        print("saved", out)
        # train/test ensemble scores for calibration step
        np.save(ROOT / "temp" / "ens_scores_tr.npy", ztr)
        np.save(ROOT / "temp" / "ens_scores_te.npy", zte)
        np.save(ROOT / "temp" / "ens_y_tr.npy", np.asarray(ytr))
        np.save(ROOT / "temp" / "ens_y_te.npy", np.asarray(yte))


if __name__ == "__main__":
    main()
