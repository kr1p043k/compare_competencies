"""CI-safe tests for the LTR artifact contract.

These run without data/models and without labeling/gold.csv, so they execute
in CI on a fresh clone. They pin the behaviour that the shipped artifacts rely
on: candidate resolution order, ensemble scoring, and the evaluator/LTR blend.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402

_RUNTIME_FEATURES = [
    "level_encoded",
    "category_encoded",
    "cosine_sim",
    "in_student_profile",
    "skill_freq_normalized",
    "co_occurrence",
    "category_avg_weight",
    "student_skills_count",
]


class _FixedModel:
    """Minimal stand-in exposing the single method the engine calls."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _write_artifact(
    models_dir,
    name: str,
    metric: float | None,
    *,
    features: list[str] | None = None,
    metric_key: str = "human_ndcg_at_10",
    metrics_json: bool = False,
):
    """Materialise a fake but loadable LTR artifact (+ its recorded metric).

    ``metrics_json=True`` writes the sidecar ``<name>.joblib.metrics.json`` with
    ``ndcg@10`` instead of a manifest, so the metrics.json reader is exercised.
    """
    joblib.dump({"feature_names": features if features is not None else _RUNTIME_FEATURES}, models_dir / name)
    if metric is None:
        return
    if metrics_json:
        (models_dir / f"{name}.metrics.json").write_text(
            json.dumps({"gold": "labeling/gold.csv", "ndcg@10": metric}),
            encoding="utf-8",
        )
    else:
        (models_dir / name.replace(".joblib", ".manifest.json")).write_text(
            json.dumps({"artifact": name, "metrics": {metric_key: metric}}),
            encoding="utf-8",
        )


def test_resolve_model_path_picks_highest_recorded_ndcg(tmp_path, monkeypatch):
    """Selection is metric-driven: the artifact with the highest NDCG@10 wins."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    _write_artifact(tmp_path, "ltr_ranker_low.joblib", 0.61)
    _write_artifact(tmp_path, "ltr_ranker_high.joblib", 0.74)

    resolved = LTRRecommendationEngine._resolve_model_path()
    assert resolved.name == "ltr_ranker_high.joblib"


def test_resolve_model_path_tie_breaks_by_filename(tmp_path, monkeypatch):
    """Equal metrics resolve deterministically by ascending filename."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    _write_artifact(tmp_path, "ltr_ranker_b.joblib", 0.70)
    _write_artifact(tmp_path, "ltr_ranker_a.joblib", 0.70)

    resolved = LTRRecommendationEngine._resolve_model_path()
    assert resolved.name == "ltr_ranker_a.joblib"


def test_resolve_model_path_reads_metrics_json_without_manifest(tmp_path, monkeypatch):
    """A sidecar metrics.json (ndcg@10) is a valid metric source."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    _write_artifact(tmp_path, "ltr_ranker_plain.joblib", 0.72, metrics_json=True)
    _write_artifact(tmp_path, "ltr_ranker_worse.joblib", 0.50)

    resolved = LTRRecommendationEngine._resolve_model_path()
    assert resolved.name == "ltr_ranker_plain.joblib"


def test_resolve_model_path_rejects_incompatible_feature_space(tmp_path, monkeypatch):
    """A legacy 10-feature artifact is never selected, even with a high metric."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    _write_artifact(
        tmp_path,
        "ltr_ranker_legacy.joblib",
        0.99,
        features=_RUNTIME_FEATURES + ["role_freq_normalized", "role_lift"],
    )
    _write_artifact(tmp_path, "ltr_ranker_ok.joblib", 0.55)

    resolved = LTRRecommendationEngine._resolve_model_path()
    assert resolved.name == "ltr_ranker_ok.joblib"


def test_resolve_model_path_skips_known_incompatible_models(tmp_path, monkeypatch):
    """xgb_rank is feature-compatible by count but unusable (stale corpus)."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert "ltr_ranker_xgb_rank.joblib" in LTRRecommendationEngine.INCOMPATIBLE_MODELS
    _write_artifact(tmp_path, "ltr_ranker_xgb_rank.joblib", 0.99)
    _write_artifact(tmp_path, "ltr_ranker_xgb_rank_human.joblib", 0.72)

    resolved = LTRRecommendationEngine._resolve_model_path()
    assert resolved.name == "ltr_ranker_xgb_rank_human.joblib"


def test_resolve_model_path_falls_back_to_surrogate_without_human(tmp_path, monkeypatch):
    """No compatible human artifact - valid fallback to the surrogate, no raise."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    (tmp_path / LTRRecommendationEngine.MODEL_FILENAME).write_bytes(b"stub")

    resolved = LTRRecommendationEngine._resolve_model_path()
    assert resolved.name == LTRRecommendationEngine.MODEL_FILENAME


def test_resolve_model_path_returns_surrogate_when_nothing_exists(tmp_path, monkeypatch):
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
