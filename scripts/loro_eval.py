"""Leave-one-role-out: generalization to unseen roles (cold start).

For each role R: train fit_ranker on the other roles, then rank R's pairs
with a market-mean profile (exactly what production does for a new role).
Metric: NDCG@10 / P@10 on R's pairs as a single query + same-row baselines.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.metrics import ndcg_score  # noqa: E402

from src import Ok  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402
from train_ltr_ranker import load_vacancies, read_gold  # noqa: E402


def single_query_metrics(yt, scores):
    yt = np.asarray(yt)
    scores = np.asarray(scores, dtype=float)
    out = {}
    try:
        v = ndcg_score(yt.reshape(1, -1), scores.reshape(1, -1), k=min(10, len(yt)))
        out["ndcg@10"] = round(float(v), 4) if not np.isnan(v) else 0.0
    except Exception:
        out["ndcg@10"] = 0.0
    topk = np.argsort(-scores, kind="stable")[: min(10, len(yt))]
    out["p@10"] = round(sum(1 for j in topk if yt[j] >= 1) / min(10, len(yt)), 4)
    return out


def main() -> None:
    import asyncio
    gold_path = ROOT / "tests" / "fixtures" / "gold_frozen.csv"
    if not gold_path.exists():
        gold_path = ROOT / "labeling" / "gold.csv"
    gold = read_gold(gold_path)
    vacancies = asyncio.run(load_vacancies(3000))
    roles = sorted({g["role"] for g in gold})
    print("roles:", roles, "gold:", len(gold), "vac:", len(vacancies))

    engine = LTRRecommendationEngine()
    report = {}
    dump = {}
    for held in roles:
        sub = [g for g in gold if g["role"] != held]
        res = engine.fit_ranker(sub, vacancies, seed=42,
                                model_out=Path("temp") / f"loro_{held}.joblib")
        assert isinstance(res, Ok), res
        market_emb = np.mean(list(engine.skill_embeddings.values()), axis=0)
        rows_h = [g for g in gold if g["role"] == held]
        Xh, yh, _, _, _ = engine._build_ranking_frame(
            list(range(len(rows_h))), rows_h, {}, market_emb, {held: 0})
        pred = np.asarray(engine.model.predict(Xh))
        m = {"model": single_query_metrics(yh, pred)}
        for name, col in (("frequency", "skill_freq_normalized"), ("cosine", "cosine_sim")):
            m[name] = single_query_metrics(yh, Xh[col].to_numpy())
        m["n"] = len(rows_h)
        report[held] = m
        dump[held] = {"grade": [int(x) for x in yh],
                      "ranker": [float(x) for x in pred],
                      "freq": [float(x) for x in Xh["skill_freq_normalized"].to_numpy()]}
        print(held, m)
    macro = {k: round(float(np.mean([v[k]["ndcg@10"] for v in report.values()])), 4)
             for k in ("model", "frequency", "cosine")}
    print("MACRO-NDCG:", macro)
    # --- blend tuning: alpha*z(ranker) + (1-alpha)*z(freq), grid on macro ---
    best = (0.0, -1.0)
    for a in [round(x * 0.05, 2) for x in range(21)]:
        ns = []
        for held in roles:
            d = dump[held]
            r = np.asarray(d["ranker"])
            f = np.asarray(d["freq"])
            zr = (r - r.mean()) / (r.std() or 1.0)
            zf = (f - f.mean()) / (f.std() or 1.0)
            b = a * zr + (1 - a) * zf
            yt = np.asarray(d["grade"])
            try:
                v = ndcg_score(yt.reshape(1, -1), b.reshape(1, -1), k=min(10, len(yt)))
            except Exception:
                v = 0.0
            ns.append(float(v) if not np.isnan(v) else 0.0)
        macro_b = float(np.mean(ns))
        if macro_b > best[1]:
            best = (a, macro_b)
    print(f"BLEND: alpha*={best[0]} macro={round(best[1], 4)}")
    with open(ROOT / "labeling" / "loro_report.json", "w", encoding="utf-8") as f:
        json.dump({"per_role": report, "macro_ndcg@10": macro,
                   "blend": {"alpha_ranker": best[0], "macro_ndcg@10": round(best[1], 4)}},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
