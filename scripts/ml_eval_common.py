"""Shared helpers for offline ML monitoring scripts (calibration, drift).

Both monitoring scripts must score the exact same feature space on the exact
same split as ``scripts/eval_frozen.py`` (per-role stratified 25% holdout, seed
42, on ``tests/fixtures/gold_frozen.csv``). That logic lives here so it is
written once and cannot silently diverge between the gate and the monitors.

Importing this module is side-effect free apart from the standard shap/cv2
stubs the LTR engine needs under test.

Usage:
    from scripts.ml_eval_common import build_frames, FEATURE_NAMES
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from scripts.train_ltr_ranker import read_gold  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402

DEFAULT_GOLD = ROOT / "tests" / "fixtures" / "gold_frozen.csv"

# The 8-feature runtime space (LTRRecommendationEngine._RUNTIME_FEATURES).
FEATURE_NAMES: tuple[str, ...] = (
    "level_encoded",
    "category_encoded",
    "cosine_sim",
    "in_student_profile",
    "skill_freq_normalized",
    "co_occurrence",
    "category_avg_weight",
    "student_skills_count",
)


def frozen_split(rows: list[dict], seed: int = 42, test_size: float = 0.25):
    """Per-role stratified holdout, identical to eval_frozen.py / fit_ranker."""
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


def load_engine(artifact: Path | None = None) -> LTRRecommendationEngine:
    """Load the shipped artifact (or the metric-selected one) into an engine.

    The live sentence-transformers model is NOT needed for frozen-split scoring:
    the artifact already carries the training ``skill_embeddings`` and
    ``_build_ranking_frame`` reads only those. Patching ``get_embedding_model``
    keeps the monitors fast and avoids loading a ~1 GB model they never use
    (same approach as scripts/experiments/make_ensemble.py).
    """
    model_path = Path(artifact) if artifact else LTRRecommendationEngine._resolve_model_path()
    with patch(
        "src.predictors.ltr_recommendation_engine.get_embedding_model",
        return_value=MagicMock(),
    ):
        engine = LTRRecommendationEngine(model_path=model_path)
    result = engine.load_model(model_path)
    if result.is_err():
        raise RuntimeError(f"could not load {model_path}: {result.unwrap_err()}")
    return engine


def build_frames(
    gold_path: Path | None = None,
    artifact: Path | None = None,
    seed: int = 42,
    test_size: float = 0.25,
) -> dict:
    """Build train/test ranking frames + raw scores for the shipped artifact.

    Returns a dict with rows, roles, indices, the feature matrices, labels, the
    per-row role labels in frame order, and the raw model scores.
    """
    gold_path = Path(gold_path) if gold_path else DEFAULT_GOLD
    rows = read_gold(gold_path)
    roles, train_idx, test_idx = frozen_split(rows, seed, test_size)
    engine = load_engine(artifact)
    if tuple(engine.feature_names) != FEATURE_NAMES:
        raise RuntimeError(
            f"artifact feature space {list(engine.feature_names)} != runtime {list(FEATURE_NAMES)}"
        )

    train_core: dict[str, list[str]] = {r: [] for r in roles}
    for j in train_idx:
        if rows[j]["grade"] == 2:
            train_core[rows[j]["role"]].append(rows[j]["skill"])
    market_emb = np.mean(list(engine.skill_embeddings.values()), axis=0)
    qid_of = {r: k for k, r in enumerate(roles)}

    x_train, y_train, _, _, order_train = engine._build_ranking_frame(
        train_idx, rows, train_core, market_emb, qid_of
    )
    x_test, y_test, _, _, order_test = engine._build_ranking_frame(
        test_idx, rows, train_core, market_emb, qid_of
    )
    x_train = x_train[list(engine.feature_names)]
    x_test = x_test[list(engine.feature_names)]

    try:
        gold_label = str(gold_path.relative_to(ROOT))
    except ValueError:
        gold_label = str(gold_path)

    return {
        "artifact": Path(engine.model_path).name,
        "rows": rows,
        "roles": roles,
        "gold": gold_label,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "x_train": x_train,
        "y_train": np.asarray(y_train),
        "x_test": x_test,
        "y_test": np.asarray(y_test),
        "roles_train": [rows[i]["role"] for i in order_train],
        "roles_test": [rows[i]["role"] for i in order_test],
        "scores_train": engine._predict_raw(x_train),
        "scores_test": engine._predict_raw(x_test),
    }


def znorm_by_query(scores: np.ndarray, row_roles: list[str]) -> np.ndarray:
    """Per-query (per-role) z-score; LTR scores are query-relative."""
    out = np.empty_like(np.asarray(scores, dtype=float))
    roles = np.asarray(row_roles)
    for role in sorted(set(row_roles)):
        mask = roles == role
        chunk = np.asarray(scores, dtype=float)[mask]
        std = float(np.std(chunk)) or 1.0
        out[mask] = (chunk - float(np.mean(chunk))) / std
    return out
