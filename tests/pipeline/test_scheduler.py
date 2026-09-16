"""Scheduler tests: settings persistence, due logic, busy guard, profiles override."""
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest  # noqa: E402

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

import src.pipeline.background_collector as sched  # noqa: E402
from src.pipeline.background_collector import (  # noqa: E402
    is_collect_due,
    is_gap_due,
    is_gap_overdue,
    is_scheduler_busy,
    load_scheduler_settings,
    save_scheduler_settings,
    scheduler_status,
)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.background_collector.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.background_collector.config.settings.BACKGROUND_COLLECTOR_ENABLED", False)
    monkeypatch.setattr("src.pipeline.background_collector.config.settings.SCHEDULER_COLLECT_INTERVAL_HOURS", 12)
    monkeypatch.setattr("src.pipeline.background_collector.config.settings.SCHEDULER_DAILY_GAP_ENABLED", False)
    sched._scheduler_cache = None
    sched._scheduler_busy = None
    yield tmp_path
    sched._scheduler_cache = None
    sched._scheduler_busy = None


class TestSettings:
    def test_seed_defaults_when_no_file(self, tmp_data):
        st = load_scheduler_settings()
        assert st["collector_enabled"] is False
        assert st["collect_interval_hours"] == 12
        assert st["daily_gap_enabled"] is False
        assert st["last_collect_ts"] is None
        assert st["last_gap_date"] is None

    def test_seed_respects_env_flag(self, tmp_data, monkeypatch):
        monkeypatch.setattr(
            "src.pipeline.background_collector.config.settings.BACKGROUND_COLLECTOR_ENABLED", True)
        assert load_scheduler_settings()["collector_enabled"] is True

    def test_roundtrip(self, tmp_data):
        out = save_scheduler_settings({"collector_enabled": True, "collect_interval_hours": 6})
        assert out["collector_enabled"] is True
        assert out["collect_interval_hours"] == 6
        assert (tmp_data / "settings" / "scheduler.json").exists()
        assert load_scheduler_settings()["collect_interval_hours"] == 6

    def test_unknown_keys_ignored_and_clamped(self, tmp_data):
        out = save_scheduler_settings({"nope": 1, "collect_interval_hours": 999})
        assert "nope" not in out
        assert out["collect_interval_hours"] == 72
        out = save_scheduler_settings({"collect_interval_hours": 0})
        assert out["collect_interval_hours"] == 1

    def test_corrupt_file_falls_back_to_defaults(self, tmp_data):
        f = tmp_data / "settings" / "scheduler.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{broken", encoding="utf-8")
        st = load_scheduler_settings()
        assert st["collect_interval_hours"] == 12

    def test_status_shape(self, tmp_data):
        st = scheduler_status()
        for key in ("collector_enabled", "collect_interval_hours", "daily_gap_enabled",
                    "last_collect_ts", "last_gap_date", "busy", "pipeline_running",
                    "next_collect_ts", "server_time"):
            assert key in st, key


class TestDueLogic:
    def test_collect_due_first_run(self):
        assert is_collect_due(1000.0, None, 12) is True

    def test_collect_due_interval(self):
        assert is_collect_due(1000.0, 1000.0 - 13 * 3600, 12) is True
        assert is_collect_due(1000.0, 1000.0 - 11 * 3600, 12) is False

    def test_gap_due(self):
        assert is_gap_due("2026-09-16", "2026-09-15") is True
        assert is_gap_due("2026-09-16", "2026-09-16") is False
        assert is_gap_due("2026-09-16", None) is True

    def test_gap_overdue(self):
        assert is_gap_overdue("2026-09-16", None) is True
        assert is_gap_overdue("2026-09-16", "2026-09-16") is False
        assert is_gap_overdue("2026-09-23", "2026-09-16") is True
        assert is_gap_overdue("2026-09-20", "2026-09-16") is False


class TestBusyGuard:
    async def test_collect_once_skips_when_busy(self, tmp_data):
        sched._scheduler_busy = "gap"
        try:
            assert await sched._run_collect_once() == ("skipped", 0)
        finally:
            sched._scheduler_busy = None

    async def test_scheduled_gap_skips_when_busy(self, tmp_data):
        sched._scheduler_busy = "collect"
        try:
            assert await sched._run_scheduled_gap() is False
        finally:
            sched._scheduler_busy = None

    def test_is_scheduler_busy(self, tmp_data):
        assert is_scheduler_busy() is False
        sched._scheduler_busy = "collect"
        try:
            assert is_scheduler_busy() is True
        finally:
            sched._scheduler_busy = None


class TestResolveProfiles:
    def test_override_wins(self):
        from src.pipeline.runner import _resolve_profiles
        args = SimpleNamespace(profiles_override={"custom": object()})
        with patch("src.pipeline.runner.build_profiles") as mocked:
            out = _resolve_profiles(args, {"base": ["x"]}, {})
        assert set(out) == {"custom"}
        mocked.assert_not_called()

    def test_fallback_to_files(self):
        from src.pipeline.runner import _resolve_profiles
        args = SimpleNamespace()
        with patch("src.pipeline.runner.build_profiles", return_value={"base": 1}) as mocked:
            assert _resolve_profiles(args, {"base": ["x"]}, {"m": 1}) == {"base": 1}
        mocked.assert_called_once()
