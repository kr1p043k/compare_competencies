"""Ensemble LORO: per held-out role, 3-seed ensemble; blend retune; OOF dump."""
import asyncio
import json
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

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)


def qndcg(yt, sc):
    yt = np.asarray(yt)
    try:
        v = ndcg_score(yt.reshape(1, -1), np.asarray(sc, dtype=float).reshape(1, -1), k=min(10, len(yt)))
    except Exception:
        return 0.0
    return float(v) if not np.isnan(v) else 0.0


def main() -> None:
    rows = read_gold(ROOT / "labeling" / "gold.csv")
    roles = sorted({r["role"] for r in rows})
    vacancies = asyncio.run(load_vacancies(3000))
    with patch("src.predictors.ltr_recommendation_engine.get_embedding_model",
               return_value=get_embedding_model()):
        eng = LTRRecommendationEngine()
        rep, oof = {}, {}
        for held in roles:
            sub = [g for g in rows if g["role"] != held]
            P, F, Y = [], None, None
            for s in SEEDS:
                res = eng.fit_ranker(sub, vacancies, seed=s,
                                     model_out=ROOT / "temp" / f"elor_{held[:6]}_s{s}.joblib")
                assert isinstance(res, Ok), (held, s, res)
                market_emb = np.mean(list(eng.skill_embeddings.values()), axis=0)
                rows_h = [g for g in rows if g["role"] == held]
                Xh, yh, _, _, _ = eng._build_ranking_frame(
                    list(range(len(rows_h))), rows_h, {}, market_emb, {held: 0})
                P.append(np.asarray(eng.model.predict(Xh)))
                F = Xh["skill_freq_normalized"].to_numpy()
                Y = np.asarray(yh)
            zens = np.mean([(p - p.mean()) / (p.std() or 1.0) for p in P], axis=0)
            f = np.asarray(F, dtype=float)
            zf = (f - f.mean()) / (f.std() or 1.0)
            rep[held] = {"ens": round(qndcg(Y, zens), 4),
                         "freq": round(qndcg(Y, f), 4), "n": len(Y)}
            oof[held] = {"grade": [int(x) for x in Y],
                         "ens_z": [float(x) for x in zens],
                         "freq_z": [float(x) for x in zf]}
            print(held, rep[held], flush=True)
    macro_ens = round(float(np.mean([v["ens"] for v in rep.values()])), 4)
    macro_fq = round(float(np.mean([v["freq"] for v in rep.values()])), 4)
    best = (0.0, -1.0)
    for i in range(21):
        a = round(i * 0.05, 2)
        ns = []
        for held in roles:
            d = oof[held]
            b = a * np.asarray(d["ens_z"]) + (1 - a) * np.asarray(d["freq_z"])
            ns.append(qndcg(d["grade"], b))
        m = float(np.mean(ns))
        if m > best[1]:
            best = (a, m)
    print("MACRO ens:", macro_ens, "freq:", macro_fq, f"BLEND alpha*={best[0]} macro={round(best[1], 4)}")
    with open(ROOT / "labeling" / "loro_ensemble_report.json", "w", encoding="utf-8") as f:
        json.dump({"per_role": rep, "macro_ens": macro_ens, "macro_freq": macro_fq,
                   "blend": {"alpha": best[0], "macro": round(best[1], 4)}}, f, ensure_ascii=False, indent=2)
    with open(ROOT / "labeling" / "loro_ensemble_oof.json", "w", encoding="utf-8") as f:
        json.dump(oof, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
