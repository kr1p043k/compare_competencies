"""CI-safe tests for the LTR artifact contract.

These run without data/models and without labeling/gold.csv, so they execute
in CI on a fresh clone. They pin the behaviour that the shipped artifacts rely
on: candidate resolution order, ensemble scoring, and the evaluator/LTR blend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402


class _FixedModel:
    """Minimal stand-in exposing the single method the engine calls."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def test_model_candidates_prefer_the_evaluated_best():
    candidates = LTRRecommendationEngine.MODEL_CANDIDATES
    assert candidates[0] == "ltr_ranker_ensemble_human.joblib"
    # Trained on the old 272-skill corpus and incompatible with the current
    # feature space; keeping it in the list would break loading.
    assert "ltr_ranker_xgb_rank.joblib" not in candidates


def test_resolve_model_path_prefers_canonical_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    canonical = tmp_path / LTRRecommendationEngine.MODEL_FILENAME
    canonical.write_bytes(b"stub")

    resolved = LTRRecommendationEngine._resolve_model_path()
    assert resolved == canonical


def test_resolve_model_path_falls_back_in_declared_order(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    # Only the third candidate exists.
    third = LTRRecommendationEngine.MODEL_CANDIDATES[2]
    (tmp_path / third).write_bytes(b"stub")

    resolved = LTRRecommendationEngine._resolve_model_path()
    assert resolved.name == third


def test_resolve_model_path_returns_canonical_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    resolved = LTRRecommendationEngine._resolve_model_path()
    assert resolved.name == LTRRecommendationEngine.MODEL_FILENAME


def test_predict_raw_uses_single_model():
    engine = LTRRecommendationEngine(model_path="unused.joblib")
    engine.model = _FixedModel(0.25)
    engine.models = []

    scores = engine._predict_raw(pd.DataFrame({"a": [1, 2, 3]}))
    assert np.allclose(scores, 0.25)


def test_predict_raw_averages_ensemble():
    engine = LTRRecommendationEngine(model_path="unused.joblib")
    engine.model = None
    engine.models = [_FixedModel(0.1), _FixedModel(0.5)]

    scores = engine._predict_raw(pd.DataFrame({"a": [1, 2]}))
    assert np.allclose(scores, 0.3)


def test_predict_raw_without_any_model_raises():
    engine = LTRRecommendationEngine(model_path="unused.joblib")
    engine.model = None
    engine.models = []

    with pytest.raises(Exception):
        engine._predict_raw(pd.DataFrame({"a": [1]}))


def test_blend_weights_are_a_convex_combination():
    total = config.BLEND_EVALUATOR_WEIGHT + config.BLEND_LTR_WEIGHT
    assert total == pytest.approx(1.0), "blend weights must sum to 1"
    # Frozen-split eval gave the ranker a clear lead over both baselines, so it
    # must carry at least half of the blend.
    assert config.BLEND_LTR_WEIGHT >= 0.5


def test_similarity_ladder_is_monotonic():
    ladder = [
        config.SIM_ACADEMIC_MARKET,
        config.SIM_ACADEMIC_GAP,
        config.SIM_MARKET_MATCH_MEDIUM,
        config.SIM_MARKET_MATCH_HIGH,
        config.SIM_SKILL_MATCH,
        config.SIM_DEDUP,
    ]
    assert ladder == sorted(ladder), f"similarity thresholds must increase: {ladder}"
