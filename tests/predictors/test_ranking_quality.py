"""Ranking-quality regression guard for the shipped LTR artifact.

The metric is NDCG@10 on a frozen split of labeling/gold.csv (per-role
stratified 25% holdout, seed 42). The same split is used by
scripts/eval_frozen.py, so the numbers here are directly comparable.

The artifact itself lives under data/, which is gitignored, so the test skips
when it is absent (fresh clone, CI without the model). Run it locally after
retraining or after touching the ranker:

    pytest tests/predictors/test_ranking_quality.py -q
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import sys

import numpy as np
import pytest

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from sklearn.metrics import ndcg_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# The gate must run in CI, so the frozen split lives in the repo as a fixture
# (role / skill / grade only, no personal data). labeling/gold.csv stays the
# source of truth for local evaluation.
FIXTURE = ROOT / "tests" / "fixtures" / "gold_frozen.csv"
GOLD = FIXTURE if FIXTURE.exists() else ROOT / "labeling" / "gold.csv"

# Measured with the default artifact (ltr_ranker_ensemble_human) on the frozen
# split: NDCG@10 = 0.7455, P@10 = 0.750. Leave headroom for environment noise.
MIN_NDCG_AT_10 = 0.70
MIN_P_AT_10 = 0.65

pytestmark = pytest.mark.skipif(
    not GOLD.exists(), reason="labeling/gold.csv is not present"
)


def read_gold(path: Path) -> list[dict]:
    import csv

    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        reader.fieldnames = [c.replace("\ufeff", "").strip() for c in (reader.fieldnames or [])]
        for r in reader:
            try:
                grade = int(r["grade"])
            except (KeyError, ValueError, TypeError):
                continue
            if grade in (0, 1, 2):
                rows.append({"role": r["role"].strip(), "skill": r["skill"].strip(), "grade": grade})
    return rows


def frozen_split(rows: list[dict], seed: int = 42, test_size: float = 0.25):
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


def score_artifact(engine: LTRRecommendationEngine, rows, train_idx, test_idx, roles) -> tuple[float, float]:
    train_core: dict[str, list[str]] = {r: [] for r in roles}
    for j in train_idx:
        if rows[j]["grade"] == 2:
            train_core[rows[j]["role"]].append(rows[j]["skill"])

    qid_of = {r: k for k, r in enumerate(roles)}
    market_emb = np.mean(list(engine.skill_embeddings.values()), axis=0)
    X, y, _, _, order = engine._build_ranking_frame(test_idx, rows, train_core, market_emb, qid_of)
    X = X[engine.feature_names]
    scores = engine._predict_raw(X)
    roles_of_row = [rows[j]["role"] for j in order]

    ndcgs, precs = [], []
    for role in sorted(set(roles_of_row)):
        mask = np.array([r == role for r in roles_of_row])
        yt, sc = y[mask], np.asarray(scores, dtype=float)[mask]
        if len(yt) < 2 or np.all(yt == yt[0]):
            continue
        try:
            v = ndcg_score(yt.reshape(1, -1), sc.reshape(1, -1), k=min(10, len(yt)))
            if not np.isnan(v):
                ndcgs.append(float(v))
        except Exception:
            continue
        top = np.argsort(-sc, kind="stable")[: min(10, len(yt))]
        precs.append(sum(1 for j in top if yt[j] >= 1) / len(top))

    return (
        float(np.mean(ndcgs)) if ndcgs else 0.0,
        float(np.mean(precs)) if precs else 0.0,
    )


def _artifact_path() -> Path | None:
    """Resolve the default artifact against the repo root, not the cwd.

    The test suite chdirs into a temp directory (tests/conftest.py), while
    config.MODELS_DIR is a relative path, so the engine's own default would
    point at the temp dir.
    """
    models = ROOT / "data" / "models"
    for name in LTRRecommendationEngine.MODEL_CANDIDATES:
        candidate = models / name
        if candidate.exists():
            return candidate
    return None


def test_default_artifact_ranking_quality():
    artifact = _artifact_path()
    if artifact is None:
        pytest.skip("no LTR artifact under data/models")

    engine = LTRRecommendationEngine(model_path=artifact)
    loaded = engine.load_model()
    assert loaded.is_ok(), f"artifact failed to load: {loaded}"
    assert engine.model is not None or engine.models, "artifact has no model"

    rows = read_gold(GOLD)
    roles, train_idx, test_idx = frozen_split(rows)
    assert test_idx, "frozen split produced no test rows"

    ndcg, precision = score_artifact(engine, rows, train_idx, test_idx, roles)

    assert ndcg >= MIN_NDCG_AT_10, (
        f"NDCG@10 dropped to {ndcg:.4f} (floor {MIN_NDCG_AT_10}); "
        f"artifact={engine.model_path.name}"
    )
    assert precision >= MIN_P_AT_10, (
        f"P@10 dropped to {precision:.4f} (floor {MIN_P_AT_10}); "
        f"artifact={engine.model_path.name}"
    )
