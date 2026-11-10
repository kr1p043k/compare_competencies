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
