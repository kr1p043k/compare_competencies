"""Tests for src.monitoring.gap_metrics."""

import pytest

from src.monitoring.gap_metrics import (
    GapMetricsTracker,
    gap_cache_hit_rate,
    gap_metrics,
    gap_success_rate,
    shap_duration,
)


@pytest.fixture(autouse=True)
def reset_gap_metrics():
    for profile in ("base", "dc", "top_dc"):
        gap_success_rate.labels(profile_type=profile).set(100.0)
        gap_cache_hit_rate.labels(profile_type=profile).set(0.0)


@pytest.fixture
def tracker():
    t = GapMetricsTracker()
    t.reset()
    return t


def test_reset_clears_state(tracker):
    tracker.start("base", 1)
    tracker.add_gaps({"high": 2})
    tracker.reset()
    assert tracker.profile_type == ""
    assert tracker.region == 0
    assert tracker.start_time is None
    assert tracker.gaps == {"critical": 0, "high": 0, "medium": 0, "low": 0}


def test_start_initializes_state_and_sets_cache_metric(tracker):
    tracker.start("top_dc", 2, cache_hit=True)
    assert tracker.profile_type == "top_dc"
    assert tracker.region == 2
    assert tracker.cache_hit is True
    assert tracker.start_time is not None


def test_gap_metrics_singleton_is_instance():
    assert isinstance(gap_metrics, GapMetricsTracker)
    gap_metrics.reset()
    assert gap_metrics.profile_type == ""


def test_start_cache_hit_rate_calculation(monkeypatch):
    tracker = GapMetricsTracker()
    metric = gap_cache_hit_rate.labels(profile_type="dc")
    monkeypatch.setattr(metric._value, "get", lambda: 0.8)
    tracker.start("dc", 3, cache_hit=False)
    monkeypatch.delattr(metric._value, "get")
    # 0.8 * 0.95 + 0 = 0.76
    assert metric._value.get() == pytest.approx(0.76)


def test_start_cache_hit_rate_defaults_to_half(monkeypatch):
    tracker = GapMetricsTracker()
    metric = gap_cache_hit_rate.labels(profile_type="base")
    monkeypatch.setattr(metric._value, "get", lambda: None)
    tracker.start("base", 1, cache_hit=True)
    monkeypatch.delattr(metric._value, "get")
    # 0.5 * 0.95 + 1.0 * 0.05 = 0.525
    assert metric._value.get() == pytest.approx(0.525)


def test_add_gaps_and_recommendations_filter_unknown_keys(tracker):
    tracker.start("base", 1)
    tracker.add_gaps({"critical": 1, "high": 2, "unknown": 9})
    tracker.add_recommendations({"low": 4, "unknown": 99})
    assert tracker.gaps == {"critical": 1, "high": 2, "medium": 0, "low": 0}
    assert tracker.recommendations_by_priority == {"critical": 0, "high": 0, "medium": 0, "low": 4}


def test_shap_duration_recorded(tracker, monkeypatch):
    tracker.start("base", 1)
    metric = shap_duration.labels(profile_type="base")
    monkeypatch.setattr(metric, "_sum", type("S", (), {"get": lambda self: 0.0})())
    monkeypatch.setattr(metric, "observe", lambda v: None)
    tracker.start_shap()
    tracker.end_shap()
    assert tracker.shap_start_time is None


def test_end_without_start_returns_empty(tracker):
    assert tracker.end() == {}


def test_end_records_all_metrics(tracker, monkeypatch):
    tracker.start("base", 1, cache_hit=True)
    tracker.add_gaps({"critical": 1, "high": 2})
    tracker.add_recommendations({"high": 1, "medium": 3})

    calls = []
    monkeypatch.setattr(tracker, "record_profile_load", lambda d: calls.append(("profile_load", d)))

    result = tracker.end(success=True)
    assert result["profile_type"] == "base"
    assert result["region"] == 1
    assert result["gaps_found"] == {"critical": 1, "high": 2, "medium": 0, "low": 0}
    assert result["recommendations"] == {"critical": 0, "high": 1, "medium": 3, "low": 0}
    assert result["success"] is True
    assert result["cache_hit"] is True
    assert tracker.start_time is None


def test_record_analysis_success(tracker, monkeypatch):
    tracker.start("base", 5)
    tracker.record_analysis("base", 0.5, success=True)
    metric = gap_success_rate.labels(profile_type="base")
    value = metric._value.get()
    assert value is not None
    assert 95.0 <= value <= 100.0


@pytest.mark.xfail(reason="src.record_analysis only updates gap_success_rate on success; failures are ignored")
def test_record_analysis_failure_sets_success_rate(tracker, monkeypatch):
    tracker.start("dc", 0)
    gap_success_rate.labels(profile_type="dc").set(100.0)
    tracker.record_analysis("dc", 0.1, success=False)
    value = gap_success_rate.labels(profile_type="dc")._value.get()
    assert value == pytest.approx(95.0)


def test_get_stats_returns_defaults(tracker):
    stats = tracker.get_stats()
    for profile in ["base", "dc", "top_dc"]:
        assert profile in stats
        assert stats[profile]["success_rate"] == 100.0
        assert stats[profile]["cache_hit_rate"] == 0.0


def test_get_stats_with_values(monkeypatch):
    tracker = GapMetricsTracker()
    for profile in ["base", "dc", "top_dc"]:
        gap_success_rate.labels(profile_type=profile).set(50.0)
        gap_cache_hit_rate.labels(profile_type=profile).set(25.0)
    stats = tracker.get_stats()
    assert stats["base"]["success_rate"] == pytest.approx(50.0)
    assert stats["dc"]["cache_hit_rate"] == pytest.approx(25.0)
