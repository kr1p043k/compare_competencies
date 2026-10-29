"""Frozen-split evaluation of every shipped LTR artifact.

The shipped artifacts were trained on different gold sets (gold.csv vs
gold_v1.csv) with different role counts, so their recorded metrics are not
comparable. This script pins ONE split and scores everything on it:

  split   : labeling/gold.csv, per-role stratified 25% holdout, seed 42
            (the same split fit_ranker used for ltr_ranker_canon)
  metrics : NDCG@10, P@10, MAP@10, averaged over roles
  rows    : each artifact + frequency baseline + cosine baseline
  extra   : blend sweep (cosine vs LTR) to pick BLEND_LTR_WEIGHT

Usage:
    python scripts/eval_frozen.py
    python scripts/eval_frozen.py --gold labeling/gold_v1.csv --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from sklearn.metrics import ndcg_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402
from train_ltr_ranker import read_gold  # noqa: E402

MODELS = ROOT / "data" / "models"
REPORT = ROOT / "data" / "reports" / "frozen_eval.json"

ARTIFACTS = [
    "ltr_ranker_canon.joblib",
    "ltr_ranker_xgb_rank.joblib",
    "ltr_ranker_xgb_rank_human.joblib",
    "ltr_ranker_ensemble_human.joblib",
    "ltr_ranker_ensemble_final.joblib",
]


def frozen_split(rows: list[dict], seed: int, test_size: float = 0.25):
    roles = sorted({r["role"] for r in rows})
    train_idx, test_idx = [], []
    for role in roles:
        idx = [i for i, r in enumerate(rows) if r["role"] == role]
        grades = [rows[i]["grade"] for i in idx]
        if len(idx) < 4 or len(set(grades)) < 2:
            train_idx.extend(idx)
            continue
        tr, te = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(
                idx, grades
            )
        )
        train_idx.extend(idx[i] for i in tr)
        test_idx.extend(idx[i] for i in te)
    return roles, train_idx, test_idx


def predict(engine, X):
    if engine.model is not None:
        return np.asarray(engine.model.predict(X), dtype=float)
    models = getattr(engine, "models", None)
    if models:
        preds = [np.asarray(m.predict(X), dtype=float) for m in models]
        return np.mean(preds, axis=0)
    raise RuntimeError("artifact has neither 'model' nor 'models'")


def per_role_metrics(y_true, scores, roles_of_row, k: int = 10) -> dict:
    ndcgs, precs, aps = [], [], []
    for role in sorted(set(roles_of_row)):
        mask = np.array([r == role for r in roles_of_row])
        yt, sc = y_true[mask], scores[mask]
        if len(yt) < 2 or np.all(yt == yt[0]):
            continue
        try:
            v = ndcg_score(yt.reshape(1, -1), sc.reshape(1, -1), k=min(k, len(yt)))
            if not np.isnan(v):
                ndcgs.append(float(v))
        except Exception:
            pass

        order = np.argsort(-sc, kind="stable")[: min(k, len(yt))]
        rel = [1 if yt[j] >= 1 else 0 for j in order]
        precs.append(sum(rel) / len(rel) if rel else 0.0)

        hits, ap = 0, 0.0
        for rank, j in enumerate(order, start=1):
            if yt[j] >= 1:
                hits += 1
                ap += hits / rank
        total_rel = int((yt >= 1).sum())
        aps.append(ap / total_rel if total_rel else 0.0)

    return {
        "ndcg@10": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        "p@10": round(float(np.mean(precs)), 4) if precs else 0.0,
        "map@10": round(float(np.mean(aps)), 4) if aps else 0.0,
        "roles_scored": len(ndcgs),
    }


def zscore(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    return (x - float(np.mean(x))) / (std if std > 1e-9 else 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="tests/fixtures/gold_frozen.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    gold_path = ROOT / args.gold
    if not hasattr(LTRRecommendationEngine, "_build_ranking_frame"):
        print(
            "eval_frozen: this branch's LTR engine has no _build_ranking_frame; "
            "the frozen split cannot be scored here."
        )
        return 2

    rows = read_gold(gold_path)
    roles, train_idx, test_idx = frozen_split(rows, args.seed, args.test_size)
    print(f"gold={args.gold} rows={len(rows)} roles={len(roles)}")
    print(f"frozen split: train={len(train_idx)} test={len(test_idx)} seed={args.seed}\n")

    train_core: dict[str, list[str]] = {r: [] for r in roles}
    for j in train_idx:
        if rows[j]["grade"] == 2:
            train_core[rows[j]["role"]].append(rows[j]["skill"])
    qid_of = {r: k for k, r in enumerate(roles)}

    results: dict[str, dict] = {}
    baseline_frame = None

    for name in ARTIFACTS:
        path = MODELS / name
        if not path.exists():
            print(f"skip {name}: not found")
            continue
        try:
            engine = LTRRecommendationEngine(model_path=path)
            loaded = engine.load_model(path)
            if loaded.is_err():
                print(f"skip {name}: {loaded.unwrap_err()}")
                continue
            if engine.model is None and not getattr(engine, "models", None):
                engine.models = loaded.ok().__dict__.get("models") if hasattr(loaded.ok(), "__dict__") else None

            market_emb = np.mean(list(engine.skill_embeddings.values()), axis=0)
            X, y, _, _, order = engine._build_ranking_frame(
                test_idx, rows, train_core, market_emb, qid_of
            )
            X = X[engine.feature_names]
            scores = predict(engine, X)
            roles_of_row = [rows[j]["role"] for j in order]
            results[name] = per_role_metrics(y, scores, roles_of_row)
            results[name]["features"] = len(engine.feature_names)
            if baseline_frame is None:
                baseline_frame = (X, y, roles_of_row)
            print(f"{name:40} {results[name]}")
        except Exception as exc:
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")

    if baseline_frame is None:
        print("no artifact could be scored")
        return 1

    X, y, roles_of_row = baseline_frame
    for label, col in (("baseline_frequency", "skill_freq_normalized"), ("baseline_cosine", "cosine_sim")):
        if col not in X.columns:
            continue
        scores = np.asarray(X[col].to_numpy(), dtype=float)
        results[label] = per_role_metrics(y, scores, roles_of_row)
        print(f"{label:40} {results[label]}")

    # Blend sweep on the cosine baseline vs the best artifact.
    if "cosine_sim" in X.columns:
        cos = np.asarray(X["cosine_sim"].to_numpy(), dtype=float)
        best_name, best_w, best_ndcg = None, None, -1.0
        for name in ARTIFACTS:
            if name not in results:
                continue
            engine = LTRRecommendationEngine(model_path=MODELS / name)
            if engine.load_model(MODELS / name).is_err():
                continue
            market_emb = np.mean(list(engine.skill_embeddings.values()), axis=0)
            Xa, _, _, _, order = engine._build_ranking_frame(
                test_idx, rows, train_core, market_emb, qid_of
            )
            Xa = Xa[engine.feature_names]
            ltr = predict(engine, Xa)
            roles_a = [rows[j]["role"] for j in order]
            for w in [round(0.1 * i, 1) for i in range(11)]:
                blended = (1 - w) * zscore(cos) + w * zscore(ltr)
                m = per_role_metrics(y, blended, roles_a)
                if m["ndcg@10"] > best_ndcg:
                    best_name, best_w, best_ndcg = name, w, m["ndcg@10"]
        results["best_blend"] = {
            "artifact": best_name,
            "weight_ltr": best_w,
            "ndcg@10": best_ndcg,
        }
        print(f"\nbest blend: {best_name} with weight_ltr={best_w} (ndcg@10={best_ndcg})")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "gold": args.gold,
                "seed": args.seed,
                "test_size": args.test_size,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "roles": len(roles),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
