"""Round 3: seed ensemble (no retrain) + weak+human combined training."""
import asyncio
import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from sklearn.metrics import ndcg_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from scripts.train_ltr_ranker import load_vacancies, read_gold  # noqa: E402
from src import Ok  # noqa: E402
from src.parsing.api.embedding_loader import get_embedding_model  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HUMAN_ROLES = {"Data Scientist", "Backend-разработчик", "DevOps-инженер", "Системный аналитик"}


def read_weak():
    rows = []
    with open(ROOT / "labeling" / "gold_v1.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append({"role": "weak:" + r["role"].strip(),
                         "skill": r["skill"].strip().lower(), "grade": int(r["grade"])})
    return rows


def qndcg(y_true, scores, roles):
    ns = []
    for rl in sorted(set(roles)):
        m = np.array([t == rl for t in roles])
        yt, pt = np.asarray(y_true)[m], np.asarray(scores)[m]
        if len(yt) < 2 or np.all(yt == yt[0]):
            continue
        try:
            v = ndcg_score(yt.reshape(1, -1), pt.reshape(1, -1), k=min(10, len(yt)))
        except Exception:
            continue
        if not np.isnan(v):
            ns.append(float(v))
    return round(float(np.mean(ns)), 4) if ns else 0.0


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

    with patch("src.predictors.ltr_recommendation_engine.get_embedding_model",
               return_value=get_embedding_model()):
        eng = LTRRecommendationEngine()
        # --- (a) seed ensemble from saved artifacts ---
        train_core: dict[str, list[str]] = {r: [] for r in roles}
        # need state: load base seed42 artifact for embeddings/metadata
        eng.load_model(ROOT / "temp" / "b2_base_s42.joblib")
        for j in train_idx:
            if rows[j]["grade"] == 2:
                train_core[rows[j]["role"]].append(rows[j]["skill"])
        market_emb = np.mean(list(eng.skill_embeddings.values()), axis=0)
        qid_of = {r: k for k, r in enumerate(roles)}
        Xte, yte, _, _, test_order = eng._build_ranking_frame(
            test_idx, rows, train_core, market_emb, qid_of)
        troles = [rows[j]["role"] for j in test_order]
        preds = []
        for s in (42, 43, 44):
            eng.load_model(ROOT / "temp" / f"b2_base_s{s}.joblib")
            preds.append(np.asarray(eng.model.predict(Xte)))
        zavg = np.mean([(p - p.mean()) / (p.std() or 1.0) for p in preds], axis=0)
        print("seed-ensemble test_ndcg=", qndcg(yte, zavg, troles))
        print("single-seed mean was 0.693")

        # --- (b) weak+human combined ---
        vacancies = asyncio.run(load_vacancies(3000))
        weak = read_weak()
        combo = weak + rows
        print(f"combo rows: {len(combo)} (weak {len(weak)} + human {len(rows)})")
        for s in (42, 43):
            res = eng.fit_ranker(combo, vacancies, seed=s,
                                 model_out=ROOT / "temp" / f"b3_combo_s{s}.joblib")
            assert isinstance(res, Ok), res
            # human-only test eval with combo-trained state
            hcore: dict[str, list[str]] = {}
            for j in train_idx:
                if rows[j]["grade"] == 2:
                    hcore.setdefault(rows[j]["role"], []).append(rows[j]["skill"])
            memb = np.mean(list(eng.skill_embeddings.values()), axis=0)
            Xh, yh, _, _, horder = eng._build_ranking_frame(
                list(range(len(rows))), rows, hcore, memb,
                {r: k for k, r in enumerate(roles)})
            # restrict to human test idxs
            keep = [k for k, j in enumerate(horder) if j in set(test_idx)]
            pred = np.asarray(eng.model.predict(Xh))[keep]
            yh_s = np.asarray(yh)[keep]
            rh_s = [rows[horder[k]]["role"] for k in keep]
            print(f"combo_s{s} human-test_ndcg=", qndcg(yh_s, pred, rh_s))


if __name__ == "__main__":
    main()
