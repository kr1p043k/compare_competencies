"""Tests for SnapshotTrendAnalyzer from analyzers/trend_analyzer.py."""
from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.errors import TrendError
from src.analyzers.trend_analyzer import SnapshotTrendAnalyzer


class TestTrendAnalyzer:
    def test_init_empty(self):
        ta = SnapshotTrendAnalyzer()
        assert ta.snapshots == []

    def test_init_with_records(self):
        records = [{"skill_freq": {"python": 10}}]
        ta = SnapshotTrendAnalyzer(records)
        assert ta.snapshots == records

    def test_set_snapshots_success(self):
        ta = SnapshotTrendAnalyzer()
        records = [{"skill_freq": {"python": 10}}, {"skill_freq": {"python": 20}}]
        result = ta.set_snapshots(records)
        assert result.is_ok()
        assert len(ta.snapshots) == 2

    def test_set_snapshots_empty_returns_err(self):
        ta = SnapshotTrendAnalyzer()
        result = ta.set_snapshots([])
        assert result.is_err()
        assert isinstance(result.err(), TrendError)
        assert result.err().reason == "empty"

    def test_get_rising_success(self):
        # Share-based math: prev shares python=10/30, sql=20/30;
        # cur shares python=30/45, sql=15/45.
        # python +100.0% (rising); sql -50% (excluded from rising).
        records = [
            {"skill_freq": {"python": 10, "sql": 20}},
            {"skill_freq": {"python": 30, "sql": 15}},
        ]
        ta = SnapshotTrendAnalyzer(records)
        result = ta.get_rising(top_n=10)
        assert result.is_ok()
        rising = result.ok()
        assert len(rising) == 1
        python_item = next(r for r in rising if r["skill"] == "python")
        assert python_item["change_pct"] == 100.0
        assert not any(r["skill"] == "sql" for r in rising)

    def test_get_rising_insufficient_snapshots(self):
        ta = SnapshotTrendAnalyzer([{"skill_freq": {"python": 10}}])
        result = ta.get_rising()
        assert result.is_err()
        assert result.err().reason == "insufficient_data"

    def test_get_rising_invalid_top_n(self):
        records = [{"skill_freq": {"a": 1}}, {"skill_freq": {"a": 2}}]
        ta = SnapshotTrendAnalyzer(records)
        result = ta.get_rising(top_n=0)
        assert result.is_err()
        assert result.err().reason == "invalid_args"

    def test_get_rising_prev_freq_below_threshold(self):
        records = [
            {"skill_freq": {"python": 5}},
            {"skill_freq": {"python": 50}},
        ]
        ta = SnapshotTrendAnalyzer(records)
        result = ta.get_rising(top_n=10)
        assert result.is_ok()
        rising = result.ok()
        python_item = next((r for r in rising if r["skill"] == "python"), None)
        assert python_item is None

    def test_get_rising_top_n_limits(self):
        # Unequal growth: equal 10->100 x3 gives 0% shares.
        # Here a +76.5% and b +5.9% rise; c falls out of rising.
        records = [
            {"skill_freq": {"a": 10, "b": 10, "c": 10}},
            {"skill_freq": {"a": 100, "b": 60, "c": 10}},
        ]
        ta = SnapshotTrendAnalyzer(records)
        result = ta.get_rising(top_n=2)
        assert result.is_ok()
        assert len(result.ok()) == 2

    def test_get_declining_success(self):
        # Shares: python 50/80 -> 10/30 = -46.7% (declining);
        # sql 30/80 -> 20/30 = +77.8% (rising, excluded here).
        records = [
            {"skill_freq": {"python": 50, "sql": 30}},
            {"skill_freq": {"python": 10, "sql": 20}},
        ]
        ta = SnapshotTrendAnalyzer(records)
        result = ta.get_declining(top_n=10)
        assert result.is_ok()
        declining = result.ok()
        assert len(declining) == 1
        python_item = next(r for r in declining if r["skill"] == "python")
        assert python_item["change_pct"] == -46.7
        assert not any(r["skill"] == "sql" for r in declining)

    def test_get_declining_insufficient_snapshots(self):
        ta = SnapshotTrendAnalyzer()
        result = ta.get_declining()
        assert result.is_err()
        assert result.err().reason == "insufficient_data"

    def test_get_declining_invalid_top_n(self):
        records = [{"skill_freq": {"a": 1}}, {"skill_freq": {"a": 2}}]
        ta = SnapshotTrendAnalyzer(records)
        result = ta.get_declining(top_n=-1)
        assert result.is_err()
        assert result.err().reason == "invalid_args"

    def test_get_declining_skill_in_previous_not_in_latest(self):
        records = [
            {"skill_freq": {"python": 10, "legacy": 50}},
            {"skill_freq": {"python": 20}},
        ]
        ta = SnapshotTrendAnalyzer(records)
        result = ta.get_declining(top_n=10)
        assert result.is_ok()
        declining = result.ok()
        legacy_item = next((r for r in declining if r["skill"] == "legacy"), None)
        assert legacy_item is not None
        assert legacy_item["change_pct"] == -100.0

    def test_get_declining_top_n_limits(self):
        # Unequal decline: equal 10->1 x3 gives 0% shares.
        # Here only "a" falls (-72.0%); b/c rise (+180%) and are excluded.
        records = [
            {"skill_freq": {"a": 50, "b": 10, "c": 10}},
            {"skill_freq": {"a": 5, "b": 10, "c": 10}},
        ]
        ta = SnapshotTrendAnalyzer(records)
        result = ta.get_declining(top_n=1)
        assert result.is_ok()
        assert len(result.ok()) == 1

    def test_rising_and_declining_integration(self):
        records = [
            {"skill_freq": {"python": 10, "sql": 20, "docker": 10}},
            {"skill_freq": {"python": 30, "sql": 10, "docker": 15}},
        ]
        ta = SnapshotTrendAnalyzer(records)
        rising = ta.get_rising(10).ok()
        declining = ta.get_declining(10).ok()
        rising_skills = {r["skill"] for r in rising}
        declining_skills = {r["skill"] for r in declining}
        assert "python" in rising_skills
        assert "docker" in rising_skills
        assert "sql" in declining_skills