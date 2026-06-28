"""Extended tests for SkillForecastEngine (trend-based)."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src import Ok, Err
from src.predictors.skill_forecast import SkillForecastEngine, ForecastResult


class TestSkillForecastTrend:
    def test_engine_name(self):
        engine = SkillForecastEngine()
        assert engine.name == "TrendForecast"

    def test_fit_empty(self):
        engine = SkillForecastEngine()
        result = engine.fit({})
        assert result.is_ok()
        assert engine.is_fitted
        assert engine.forecast_all().ok() == []

    def test_fit_with_frequencies(self):
        engine = SkillForecastEngine()
        freqs = {"python": 80, "java": 60, "rust": 30}
        result = engine.fit(freqs)
        assert result.is_ok()
        assert engine.is_fitted

    def test_predict_known_skill(self):
        engine = SkillForecastEngine()
        engine.fit({"python": 80, "java": 60})
        result = engine.predict("python")
        assert result.is_ok()
        fr = result.ok()
        assert fr.skill == "python"
        assert fr.current_frequency == 80
        assert fr.next_year_frequency > 0
        assert 0 <= fr.confidence <= 1.0
        assert fr.engine_used in ("trend", "trend_flat")

    def test_predict_unknown_skill(self):
        engine = SkillForecastEngine()
        engine.fit({"python": 80})
        result = engine.predict("unknown_skill")
        assert result.is_err()

    def test_forecast_all(self):
        engine = SkillForecastEngine()
        freqs = {"python": 80, "java": 60, "rust": 30}
        engine.fit(freqs)
        results = engine.forecast_all().ok()
        assert len(results) == 3
        assert all(isinstance(r, ForecastResult) for r in results)

    def test_top_growing_filter_by_min_freq(self):
        engine = SkillForecastEngine()
        freqs = {"python": 80, "java": 60, "rust": 5}  # rust below MIN_FREQ=10
        engine.fit(freqs)
        top = engine.top_growing(5).ok()
        for r in top:
            assert r.current_frequency >= engine.MIN_FREQ
        skills = [r.skill for r in top]
        assert "rust" not in skills

    def test_top_growing_sorted(self):
        engine = SkillForecastEngine()
        freqs = {"python": 100, "java": 90, "rust": 80, "go": 70}
        engine.fit(freqs)
        top = engine.top_growing(4).ok()
        assert len(top) == 4
        for i in range(len(top) - 1):
            assert top[i].predicted_growth >= top[i + 1].predicted_growth

    def test_forecast_alias(self):
        engine = SkillForecastEngine()
        engine.fit({"python": 80})
        result = engine.forecast("python")
        assert result.is_ok()
        assert result.ok().skill == "python"

    def test_engine_used_field(self):
        engine = SkillForecastEngine()
        engine.fit({"python": 80})
        fr = engine.predict("python").ok()
        assert fr.engine_used in ("trend", "trend_flat")
