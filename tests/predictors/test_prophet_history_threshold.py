"""Барьеры прогноза: порог истории >=6 точек и мягкий кап роста.

Без Prophet и БД: проверяем чистые функции и ветку insufficient_data.
"""

from __future__ import annotations

import pytest

from src.predictors.prophet_forecast import (
    MIN_HISTORY_POINTS,
    ProphetForecastEngine,
    has_enough_history,
)


@pytest.mark.parametrize("n,expected", [(0, False), (3, False), (5, False), (6, True), (12, True)])
def test_has_enough_history_threshold(n, expected):
    assert has_enough_history(n) is expected
    assert MIN_HISTORY_POINTS == 6


def test_predict_returns_insufficient_data_below_threshold():
    engine = ProphetForecastEngine()
    # Модель «есть», но точек мало: до модели дойти не должны.
    engine._models["sql"] = object()  # type: ignore[assignment]
    engine._skill_npoints["sql"] = 3
    engine._last_actual_freq["sql"] = 12.0

    result = engine.predict("sql")
    assert result.is_ok(), "ожидался Ok (мягкий статус, не ошибка)"
    fr = result.unwrap()
    assert fr.engine_used == "insufficient_data"
    assert fr.predicted_growth == 0.0
    assert fr.confidence == 0.0
    assert fr.data_points == 3
    assert fr.forecast_months == 0
    assert fr.next_year_frequency == pytest.approx(12.0)


def _fake_result(skill: str, data_points: int):
    from src.predictors.prophet_forecast import ForecastResult

    return ForecastResult(
        skill=skill,
        current_frequency=10.0,
        predicted_growth=0.1,
        confidence=0.5,
        next_year_frequency=11.0,
        engine_used="trend",
        data_points=data_points,
        mape=None,
        forecast_months=6,
    )


def test_forecast_all_uses_same_threshold_as_predict():
    """forecast_all и predict должны давать insufficient_data одинаково (<6 точек)."""

    engine = ProphetForecastEngine()
    engine._fallback_engine = _FakeFallback([
        _fake_result("few", data_points=3),
        _fake_result("enough", data_points=6),
    ])
    engine._skill_npoints.update({"few": 3, "enough": 6})

    out = engine.forecast_all().unwrap()
    by_skill = {r.skill: r for r in out}
    assert by_skill["few"].engine_used == "insufficient_data"
    assert by_skill["enough"].engine_used != "insufficient_data"


class _FakeFallback:
    def __init__(self, results):
        self._results = results

    def forecast_all(self, _months):
        from src import Ok

        return Ok(self._results)
