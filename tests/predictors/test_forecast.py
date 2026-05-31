"""Tests for SkillForecastEngine."""

from __future__ import annotations

from src import Ok
from src.predictors.skill_forecast import SkillForecastEngine, ForecastResult


def test_forecast_engine_creation():
    engine = SkillForecastEngine()
    assert not engine.is_fitted
    assert engine.name == "SkillForecastEngine"


def test_forecast_engine_fit_with_frequencies():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3}
    result = engine.fit(freqs)
    assert isinstance(result, Ok)
    assert engine.is_fitted


def test_forecast_engine_forecast():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3}
    result = engine.fit(freqs)
    fitted = result._value
    fr = fitted.forecast("python")
    assert fr is not None
    assert fr.skill == "python"
    assert fr.current_frequency == 0.8
    assert fr.next_year_frequency > 0
    assert 0 <= fr.confidence <= 1.0


def test_forecast_engine_forecast_unknown_skill():
    engine = SkillForecastEngine()
    engine.fit({"python": 0.8})
    fr = engine.forecast("unknown_skill_xyz")
    assert fr is None


def test_forecast_engine_forecast_all():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3, "typescript": 0.5}
    engine.fit(freqs)
    results = engine.forecast_all()
    assert len(results) == 4
    assert all(isinstance(r, ForecastResult) for r in results)


def test_forecast_engine_top_growing():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3, "typescript": 0.5}
    engine.fit(freqs)
    top = engine.top_growing(2)
    assert len(top) <= 2
    assert all(isinstance(r, ForecastResult) for r in top)


def test_forecast_engine_top_declining():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3, "typescript": 0.5}
    engine.fit(freqs)
    decl = engine.top_declining(2)
    assert len(decl) <= 2


def test_forecast_engine_method_assignment():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8}
    history = {"python": {"2024-01": 0.7, "2024-06": 0.75, "2025-01": 0.8}}
    engine.fit(freqs, history=history)
    fr = engine.forecast("python")
    assert fr is not None
    assert fr.method in ("ets", "linear", "genetic")


def test_forecast_engine_empty_fit():
    engine = SkillForecastEngine()
    engine.fit({})
    assert engine.is_fitted
    assert engine.forecast_all() == []
