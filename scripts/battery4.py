"""Clean ensemble test: FIXED split (split_seed=42), model seeds vary.
Compares 8-feature vs 10-feature (role_freq/lift) ensembles."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from sklearn.metrics import ndcg_score  # noqa: E402
from src import Ok  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402
from src.parsing.api.embedding_loader import get_embedding_model  # noqa: E402
from scripts.train_ltr_ranker import load_vacancies, read_gold  # noqa: E402
from unittest.mock import patch  # noqa: E402

ROOT = Path("C:/forCode/compare_online")
MSET = (42, 43, 44)
NOROLE = ("role_freq_normalized", "role_lift")


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


def main() -> None:
    rows = read_gold(ROOT / "labeling" / "gold.csv")
    roles = sorted({r["role"] for r in rows})
    vacancies = asyncio.run(load_vacancies(3000))
    with patch("src.predictors.ltr_recommendation_engine.get_embedding_model",
               return_value=get_embedding_model()):
        eng = LTRRecommendationEngine()
        for tag, drop in (("norole", NOROLE), ("rolefeat", ())):
            P = None
            singles = []
            for s in MSET:
                kw = {"drop_features": drop} if drop else {}
                res = eng.fit_ranker(rows, vacancies, seed=s, split_seed=42,
                                     model_out=ROOT / "temp" / f"b4_{tag}_s{s}.joblib", **kw)
                assert isinstance(res, Ok), (tag, s, res)
                m = res.value if hasattr(res, "value") else res.unwrap()
                singles.append(m["ndcg@10"])
            # rebuild common frame from last state (identical split => identical state)
            from sklearn.model_selection import StratifiedShuffleSplit
            train_idx, test_idx = [], []
            for role in roles:
                idx = [i for i, r in enumerate(rows) if r["role"] == role]
                grades = [rows[i]["grade"] for i in idx]
                tr, te = next(StratifiedShuffleSplit(
                    n_splits=1, test_size=0.25, random_state=42).split(idx, grades))
                train_idx.extend(idx[i] for i in tr)
                test_idx.extend(idx[i] for i in te)
            train_core: dict[str, list[str]] = {r: [] for r in roles}
            for j in train_idx:
                if rows[j]["grade"] == 2:
                    train_core[rows[j]["role"]].append(rows[j]["skill"])
            market_emb = np.mean(list(eng.skill_embeddings.values()), axis=0)
            qid_of = {r: k for k, r in enumerate(roles)}
            import joblib
            P = []
            for s in MSET:
                mdl = joblib.load(ROOT / "temp" / f"b4_{tag}_s{s}.joblib")["model"]
                Xte, yte, _, _, test_order = eng._build_ranking_frame(
                    test_idx, rows, train_core, market_emb, qid_of)
                cols = [c for c in eng.feature_names]  # artifact features
                P.append(np.asarray(mdl.predict(Xte[cols])))
            troles = [rows[j]["role"] for j in test_order]
            zavg = np.mean([(p - p.mean()) / (p.std() or 1.0) for p in P], axis=0)
            print(f"{tag}: singles={singles} mean={round(float(np.mean(singles)), 4)} "
                  f"ensemble={qndcg(yte, zavg, troles)}", flush=True)


if __name__ == "__main__":
    main()
