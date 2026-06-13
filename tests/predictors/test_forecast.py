"""Tests for SkillForecastEngine and ProphetForecastEngine."""

from __future__ import annotations

from datetime import date

from src import Err, Ok
from src.predictors.prophet_forecast import ProphetForecastEngine, Snapshot
from src.predictors.skill_forecast import SkillForecastEngine, ForecastResult


def test_forecast_engine_creation():
    engine = SkillForecastEngine()
    assert not engine.is_fitted
    assert engine.name == "SkillForecastGA"


def test_forecast_engine_fit_with_frequencies():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3}
    result = engine.fit(freqs)
    assert isinstance(result, Ok)
    assert engine.is_fitted


def test_forecast_engine_forecast():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3}
    engine.fit(freqs)
    fr = engine.forecast("python").ok()
    assert fr is not None
    assert fr.skill == "python"
    assert fr.current_frequency == 0.8
    assert fr.next_year_frequency > 0
    assert 0 <= fr.confidence <= 1.0


def test_forecast_engine_forecast_unknown_skill():
    engine = SkillForecastEngine()
    engine.fit({"python": 0.8})
    fr = engine.forecast("unknown_skill_xyz").ok()
    assert fr is None


def test_forecast_engine_forecast_all():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3, "typescript": 0.5}
    engine.fit(freqs)
    results = engine.forecast_all().ok()
    assert len(results) == 4
    assert all(isinstance(r, ForecastResult) for r in results)


def test_forecast_engine_top_growing():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3, "typescript": 0.5}
    engine.fit(freqs)
    top = engine.top_growing(2).ok()
    assert len(top) <= 2
    assert all(isinstance(r, ForecastResult) for r in top)


def test_forecast_engine_top_growing_sorted_desc():
    engine = SkillForecastEngine()
    freqs = {"python": 0.8, "java": 0.6, "rust": 0.3, "typescript": 0.5}
    engine.fit(freqs)
    top = engine.top_growing(4).ok()
    assert len(top) == 4
    for i in range(len(top) - 1):
        assert top[i].predicted_growth >= top[i + 1].predicted_growth



def test_forecast_engine_empty_fit():
    engine = SkillForecastEngine()
    engine.fit({})
    assert engine.is_fitted
    assert engine.forecast_all().ok() == []


# ─── ProphetForecastEngine tests ────────────────────────────────────────────────


def _make_snapshots(dates: list[date], skill_freqs: dict[str, list[float]]) -> list[Snapshot]:
    """Build snapshots from parallel date/freq arrays."""
    return [Snapshot(d, {s: freqs[i] for s, freqs_list in skill_freqs.items() for i, _ in enumerate(dates) if i < len(freqs_list)}) for d in dates]


def test_prophet_engine_creation():
    engine = ProphetForecastEngine()
    assert not engine.is_fitted
    assert engine.name == "ProphetForecast"


def test_prophet_engine_fit_returns_ok():
    snapshots = [
        Snapshot(date(2025, 1, 1), {"python": 0.5, "sql": 0.4}),
        Snapshot(date(2025, 2, 1), {"python": 0.6, "sql": 0.5}),
        Snapshot(date(2025, 3, 1), {"python": 0.7, "sql": 0.6}),
        Snapshot(date(2025, 4, 1), {"python": 0.8, "sql": 0.7}),
    ]
    engine = ProphetForecastEngine()
    result = engine.fit(snapshots)
    assert isinstance(result, Ok)
    assert engine.is_fitted
    assert "python" in engine._models


def test_prophet_engine_predict():
    snapshots = [
        Snapshot(date(2025, 1, 1), {"python": 0.5}),
        Snapshot(date(2025, 2, 1), {"python": 0.6}),
        Snapshot(date(2025, 3, 1), {"python": 0.7}),
        Snapshot(date(2025, 4, 1), {"python": 0.8}),
    ]
    engine = ProphetForecastEngine()
    engine.fit(snapshots)
    result = engine.predict("python")
    assert isinstance(result, Ok)
    fr = result.unwrap()
    assert fr.skill == "python"
    assert fr.current_frequency > 0
    assert fr.next_year_frequency > 0
    assert 0 <= fr.confidence <= 1.0


def test_prophet_engine_predict_unknown_skill():
    snapshots = [
        Snapshot(date(2025, 1, 1), {"python": 0.5}),
        Snapshot(date(2025, 2, 1), {"python": 0.6}),
    ]
    engine = ProphetForecastEngine()
    engine.fit(snapshots)
    result = engine.predict("unknown_skill")
    assert isinstance(result, Err)


def test_prophet_engine_fallback_to_genetic():
    """Skills with < 2 history points fall back to genetic engine."""
    snapshots = [Snapshot(date(2025, 1, 1), {"rust": 0.3})]
    engine = ProphetForecastEngine()
    engine.fit(snapshots, fallback_freqs={"rust": 0.3})
    assert "rust" not in engine._models
    result = engine.predict("rust")
    assert isinstance(result, Ok)
    fr = result.unwrap()
    assert fr.skill == "rust"


def test_prophet_engine_empty_snapshots():
    engine = ProphetForecastEngine()
    result = engine.fit([])
    assert isinstance(result, Err)


def test_prophet_engine_forecast_all():
    snapshots = [
        Snapshot(date(2025, 1, 1), {"python": 0.5, "sql": 0.4}),
        Snapshot(date(2025, 2, 1), {"python": 0.6, "sql": 0.5}),
        Snapshot(date(2025, 3, 1), {"python": 0.7, "sql": 0.6}),
    ]
    engine = ProphetForecastEngine()
    engine.fit(snapshots)
    result = engine.forecast_all()
    assert isinstance(result, Ok)
    forecasts = result.unwrap()
    assert len(forecasts) == 2
    assert all(isinstance(r, ForecastResult) for r in forecasts)


def test_prophet_engine_top_growing():
    snapshots = [
        Snapshot(date(2025, 1, 1), {"python": 0.5, "sql": 0.4, "java": 0.3}),
        Snapshot(date(2025, 2, 1), {"python": 0.6, "sql": 0.5, "java": 0.3}),
        Snapshot(date(2025, 3, 1), {"python": 0.7, "sql": 0.6, "java": 0.3}),
    ]
    engine = ProphetForecastEngine()
    engine.fit(snapshots)
    result = engine.top_growing(2)
    assert isinstance(result, Ok)
    top = result.unwrap()
    assert len(top) == 2
    for i in range(len(top) - 1):
        assert top[i].predicted_growth >= top[i + 1].predicted_growth


def test_prophet_engine_result_types():
    """All public methods return Result type."""
    snapshots = [
        Snapshot(date(2025, 1, 1), {"python": 0.5}),
        Snapshot(date(2025, 2, 1), {"python": 0.6}),
    ]
    engine = ProphetForecastEngine()
    engine.fit(snapshots)
    assert isinstance(engine.forecast_all(), (Ok, Err))
    assert isinstance(engine.top_growing(), (Ok, Err))
    assert isinstance(engine.predict("python"), (Ok, Err))
