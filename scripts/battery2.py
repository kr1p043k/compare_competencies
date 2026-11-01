"""Improvement battery round 2: ablation, objectives, wide HPO.

Usage: python scripts/battery2.py [abl|hpo|all]
Each config = one fit_ranker call (~40s). Results -> labeling/battery2.json
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from src import Ok  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402
from scripts.train_ltr_ranker import load_vacancies, read_gold  # noqa: E402
from unittest.mock import patch  # noqa: E402
from src.parsing.api.embedding_loader import get_embedding_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05}

ABL = [
    ("base", {}),
    ("drop_inprofile", {"drop_features": ("in_student_profile",)}),
    ("drop_level", {"drop_features": ("level_encoded",)}),
    ("drop_both_dead", {"drop_features": ("in_student_profile", "level_encoded")}),
    ("drop_cosine", {"drop_features": ("cosine_sim",)}),
    ("drop_freq", {"drop_features": ("skill_freq_normalized",)}),
    ("obj_pairwise", {"objective": "rank:pairwise"}),
    ("obj_map", {"objective": "rank:map"}),
]
HPO = [
    ("eta03_500", {"n_estimators": 500, "learning_rate": 0.03}),
    ("eta08_200", {"n_estimators": 200, "learning_rate": 0.08}),
    ("sub07", {"subsample": 0.7, "colsample_bytree": 0.7}),
    ("mcw3", {"min_child_weight": 3}),
    ("lambda5", {"reg_lambda": 5.0}),
    ("d3_500", {"n_estimators": 500, "max_depth": 3}),
    ("d5_300", {"max_depth": 5}),
    ("eta03_d5", {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 500}),
]


SEEDS = (42, 43, 44)


def run_set(eng, rows, vacancies, cfgs):
    out = {}
    for name, kw in cfgs:
        params = dict(BASE)
        params.update(kw)
        ndcgs, ps = [], []
        for s in SEEDS:
            res = eng.fit_ranker(rows, vacancies, seed=s,
                                 model_out=ROOT / "temp" / f"b2_{name}_s{s}.joblib", **params)
            assert isinstance(res, Ok), (name, s, res)
            m = res.value if hasattr(res, "value") else res.unwrap()
            ndcgs.append(m["ndcg@10"])
            ps.append(m["p@10"])
        import numpy as _np
        out[name] = {"ndcg_mean": round(float(_np.mean(ndcgs)), 4),
                     "ndcg_std": round(float(_np.std(ndcgs)), 4),
                     "p_mean": round(float(_np.mean(ps)), 4),
                     "seeds": ndcgs,
                     "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}}
        print(f"{name} -> ndcg={out[name]['ndcg_mean']}±{out[name]['ndcg_std']} {ndcgs}", flush=True)
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    rows = read_gold(ROOT / "labeling" / "gold.csv")
    vacancies = asyncio.run(load_vacancies(3000))
    rep = {}
    with patch("src.predictors.ltr_recommendation_engine.get_embedding_model",
               return_value=get_embedding_model()):
        eng = LTRRecommendationEngine()
        if which in ("abl", "all"):
            rep["ablation_objectives"] = run_set(eng, rows, vacancies, ABL)
        if which in ("hpo", "all"):
            rep["hpo"] = run_set(eng, rows, vacancies, HPO)
    with open(ROOT / "labeling" / "battery2.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("saved battery2.json")


if __name__ == "__main__":
    main()
