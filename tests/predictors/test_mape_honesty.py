"""MAPE honesty: uncomputable hold-out returns None, not fake 0.0."""

from datetime import date, timedelta

from src.predictors.prophet_forecast import _holdout_mape, _rounded_mape


def _pts(n, start=100.0):
    base = date(2026, 1, 1)
    return [(base + timedelta(days=30 * i), start + i) for i in range(n)]


class TestMapeHonesty:
    def test_too_few_points_is_none(self):
        assert _holdout_mape([]) is None
        assert _holdout_mape(_pts(1)) is None

    def test_degenerate_series_is_none(self):
        base = date(2026, 1, 1)
        assert _holdout_mape([(base, 5.0), (base, 5.0)]) is None

    def test_tiny_actual_is_none(self):
        base = date(2026, 1, 1)
        pts = [(base, 50.0), (base + timedelta(days=30), 60.0),
               (base + timedelta(days=60), 0.5)]
        assert _holdout_mape(pts) is None

    def test_healthy_series_is_float(self):
        v = _holdout_mape(_pts(6))
        assert isinstance(v, float) and v >= 0.0

    def test_rounded_helper(self):
        assert _rounded_mape({"a": 0.123456}, "a") == 0.1235
        assert _rounded_mape({"a": None}, "a") is None
        assert _rounded_mape({}, "zz") is None
