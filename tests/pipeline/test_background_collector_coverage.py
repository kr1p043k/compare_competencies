"""Покрытие планировщика: цикл, цепочки, одиночный сбор и снапшоты.

Реальные hh.ru и БД не трогаем: подменяем сбор (_try_collect), pipeline-задачи
и deps. Проверяем ветки гейтинга, circuit breaker и флагов.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.pipeline.background_collector as sched


@pytest.fixture()
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(sched.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sched.config.settings, "BACKGROUND_COLLECTOR_ENABLED", False)
    monkeypatch.setattr(sched.config.settings, "SCHEDULER_COLLECT_INTERVAL_HOURS", 12)
    monkeypatch.setattr(sched.config.settings, "SCHEDULER_DAILY_GAP_ENABLED", False)
    sched._scheduler_cache = None
    sched._scheduler_busy = None
    yield tmp_path
    sched._scheduler_cache = None
    sched._scheduler_busy = None


class _Breaker:
    def __init__(self, allow=True):
        self._allow = allow
        self.success = 0
        self.failure = 0

    def allow(self):
        return self._allow

    def record_success(self):
        self.success += 1

    def record_failure(self):
        self.failure += 1


# ---------- _seed_defaults ----------


def test_seed_defaults_fallbacks_on_broken_settings(tmp_data, monkeypatch):
    class _Boom:
        def __getattr__(self, _name):
            # Любое чтение настройки падает - seed обязан вернуть фолбэки.
            raise RuntimeError("no")

    monkeypatch.setattr(sched.config, "settings", _Boom())
    st = sched._seed_defaults()
    assert st["collector_enabled"] is False
    assert st["collect_interval_hours"] == sched.SCHEDULER_DEFAULT_INTERVAL_HOURS
    assert st["daily_gap_enabled"] is False


# ---------- _pipeline_running ----------


def test_pipeline_running_true(tmp_data, monkeypatch):
    import src.api_pkg.routers.pipeline as pl

    monkeypatch.setattr(pl, "pipeline_tasks", {"t": SimpleNamespace(status="running")})
    assert sched._pipeline_running() is True


def test_pipeline_running_false_when_idle(tmp_data, monkeypatch):
    import src.api_pkg.routers.pipeline as pl

    monkeypatch.setattr(pl, "pipeline_tasks", {"t": SimpleNamespace(status="completed")})
    assert sched._pipeline_running() is False


# ---------- _wait_until_ready ----------


async def test_wait_until_ready_immediate(tmp_data, monkeypatch):
    from src.api_pkg import deps as d

    monkeypatch.setattr(d, "is_ready", True, raising=False)
    assert await sched._wait_until_ready(timeout_s=1) is True


async def test_wait_until_ready_timeout(tmp_data, monkeypatch):
    from src.api_pkg import deps as d

    monkeypatch.setattr(d, "is_ready", False, raising=False)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(sched.asyncio, "sleep", _no_sleep)
    assert await sched._wait_until_ready(timeout_s=15) is False


# ---------- start_background_collector ----------


async def test_start_background_collector_creates_task(tmp_data, monkeypatch):
    created = {}

    def _create_task(coro):
        created["called"] = True
        coro.close()
        return SimpleNamespace()

    monkeypatch.setattr(sched.asyncio, "create_task", _create_task)
    await sched.start_background_collector()
    assert created["called"] is True


# ---------- _maybe_chained_gap / _run_chained_gap ----------


async def test_maybe_chained_gap_disabled(tmp_data, monkeypatch):
    called = {"gap": False}

    async def _gap(_b):
        called["gap"] = True
        return True

    monkeypatch.setattr(sched, "_run_chained_gap", _gap)
    await sched._maybe_chained_gap({"daily_gap_enabled": False, "last_gap_date": None}, _Breaker())
    assert called["gap"] is False


async def test_maybe_chained_gap_breaker_blocked(tmp_data, monkeypatch):
    called = {"gap": False}

    async def _gap(_b):
        called["gap"] = True
        return True

    monkeypatch.setattr(sched, "_run_chained_gap", _gap)
    await sched._maybe_chained_gap(
        {"daily_gap_enabled": True, "last_gap_date": None}, _Breaker(allow=False)
    )
    assert called["gap"] is False


async def test_maybe_chained_gap_not_due(tmp_data, monkeypatch):
    called = {"gap": False}

    async def _gap(_b):
        called["gap"] = True
        return True

    monkeypatch.setattr(sched, "_run_chained_gap", _gap)
    monkeypatch.setattr(sched, "_today_utc", lambda: "2026-01-01")
    await sched._maybe_chained_gap(
        {"daily_gap_enabled": True, "last_gap_date": "2026-01-01"}, _Breaker()
    )
    assert called["gap"] is False


async def test_maybe_chained_gap_pipeline_running(tmp_data, monkeypatch):
    called = {"gap": False}

    async def _gap(_b):
        called["gap"] = True
        return True

    monkeypatch.setattr(sched, "_run_chained_gap", _gap)
    monkeypatch.setattr(sched, "_today_utc", lambda: "2026-01-02")
    monkeypatch.setattr(sched, "_pipeline_running", lambda: True)
    await sched._maybe_chained_gap(
        {"daily_gap_enabled": True, "last_gap_date": "2026-01-01"}, _Breaker()
    )
    assert called["gap"] is False


async def test_maybe_chained_gap_runs(tmp_data, monkeypatch):
    called = {"gap": False}

    async def _gap(_b):
        called["gap"] = True
        return True

    monkeypatch.setattr(sched, "_run_chained_gap", _gap)
    monkeypatch.setattr(sched, "_today_utc", lambda: "2026-01-02")
    monkeypatch.setattr(sched, "_pipeline_running", lambda: False)
    await sched._maybe_chained_gap(
        {"daily_gap_enabled": True, "last_gap_date": "2026-01-01"}, _Breaker()
    )
    assert called["gap"] is True


async def test_run_chained_gap_success_and_failure(tmp_data, monkeypatch):
    res = {"ok": True}

    async def _scheduled():
        return res["ok"]

    monkeypatch.setattr(sched, "_run_scheduled_gap", _scheduled)
    b = _Breaker()
    assert await sched._run_chained_gap(b) is True
    assert b.success == 1
    res["ok"] = False
    assert await sched._run_chained_gap(b) is False
    assert b.failure == 1


# ---------- _run_collect_once ----------


async def test_collect_once_collected_updates_ts(tmp_data, monkeypatch):
    async def _try(**_kw):
        return "collected", 5

    monkeypatch.setattr(sched, "_try_collect", _try)
    status, count = await sched._run_collect_once()
    assert status == "collected"
    assert count == 5
    assert sched.load_scheduler_settings()["last_collect_ts"] is not None
    assert sched.is_scheduler_busy() is False


async def test_collect_once_empty_does_not_update_ts(tmp_data, monkeypatch):
    async def _try(**_kw):
        return "collected-empty", 0

    monkeypatch.setattr(sched, "_try_collect", _try)
    status, _ = await sched._run_collect_once()
    assert status == "collected-empty"
    assert sched.load_scheduler_settings()["last_collect_ts"] is None


async def test_collect_once_failure(tmp_data, monkeypatch):
    async def _try(**_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(sched, "_try_collect", _try)
    assert await sched._run_collect_once() == ("failed", 0)
    assert sched.is_scheduler_busy() is False


# ---------- _run_scheduled_gap ----------


async def test_scheduled_gap_no_evaluator(tmp_data, monkeypatch):
    from src.api_pkg import deps as d

    monkeypatch.setattr(d, "evaluator", None, raising=False)
    assert await sched._run_scheduled_gap() is False
    assert sched.is_scheduler_busy() is False


async def test_scheduled_gap_no_profiles(tmp_data, monkeypatch):
    from src.api_pkg import deps as d

    monkeypatch.setattr(d, "evaluator", object(), raising=False)
    monkeypatch.setattr(d, "student_profiles", {}, raising=False)
    assert await sched._run_scheduled_gap() is False


async def test_scheduled_gap_success(tmp_data, monkeypatch):
    import src.api_pkg.routers.pipeline as pl
    from src.api_pkg import deps as d

    monkeypatch.setattr(d, "evaluator", object(), raising=False)
    monkeypatch.setattr(d, "student_profiles", {"base": object()}, raising=False)

    async def _run_pipeline(*_a, **_kw):
        pl.pipeline_tasks["scheduled_gap_2026-01-01"] = SimpleNamespace(status="completed")

    monkeypatch.setattr(pl, "run_pipeline_task", _run_pipeline)
    monkeypatch.setattr(sched, "_today_utc", lambda: "2026-01-01")
    pl.pipeline_tasks.pop("scheduled_gap_2026-01-01", None)
    assert await sched._run_scheduled_gap() is True
    assert sched.load_scheduler_settings()["last_gap_date"] == "2026-01-01"


async def test_scheduled_gap_not_completed(tmp_data, monkeypatch):
    import src.api_pkg.routers.pipeline as pl
    from src.api_pkg import deps as d

    monkeypatch.setattr(d, "evaluator", object(), raising=False)
    monkeypatch.setattr(d, "student_profiles", {"base": object()}, raising=False)

    async def _run_pipeline(*_a, **_kw):
        pl.pipeline_tasks["scheduled_gap_2026-02-02"] = SimpleNamespace(status="failed")

    monkeypatch.setattr(pl, "run_pipeline_task", _run_pipeline)
    monkeypatch.setattr(sched, "_today_utc", lambda: "2026-02-02")
    assert await sched._run_scheduled_gap() is False


async def test_scheduled_gap_exception(tmp_data, monkeypatch):
    import src.api_pkg.routers.pipeline as pl
    from src.api_pkg import deps as d

    monkeypatch.setattr(d, "evaluator", object(), raising=False)
    monkeypatch.setattr(d, "student_profiles", {"base": object()}, raising=False)

    async def _run_pipeline(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(pl, "run_pipeline_task", _run_pipeline)
    assert await sched._run_scheduled_gap() is False
    assert sched.is_scheduler_busy() is False


# ---------- _refresh_history_snapshots ----------


async def test_refresh_snapshots_handles_errors(tmp_data, monkeypatch):
    import src.cli.backfill_market_snapshots as bf

    def _boom(*_a, **_kw):
        raise RuntimeError("no backfill")

    monkeypatch.setattr(bf, "main", _boom, raising=False)
    monkeypatch.setattr(sched.config, "HISTORY_DIR", tmp_data / "history", raising=False)
    # Не должно бросать: ошибки логируются и глотаются.
    await sched._refresh_history_snapshots()


# ---------- _scheduler_loop: один тик ----------


async def test_scheduler_loop_single_tick(tmp_data, monkeypatch):
    """Прогоняем ровно один проход цикла и прерываем его на sleep."""
    monkeypatch.setattr(sched, "_wait_until_ready", lambda *a, **k: _ok())
    monkeypatch.setattr(sched.config.settings, "BACKGROUND_COLLECTOR_ENABLED", True)
    calls = {"collect": 0, "sleeps": 0}

    async def _collect(*_a, **_kw):
        calls["collect"] += 1
        return "collected", 1

    async def _gap(*_a, **_kw):
        return True

    async def _sleep(_s):
        calls["sleeps"] += 1
        raise KeyboardInterrupt  # выходим из бесконечного цикла

    monkeypatch.setattr(sched, "_run_collect_once", _collect)
    monkeypatch.setattr(sched, "_maybe_chained_gap", _gap)
    monkeypatch.setattr(sched.asyncio, "sleep", _sleep)

    with pytest.raises(KeyboardInterrupt):
        await sched._scheduler_loop()
    assert calls["collect"] == 1
    assert calls["sleeps"] == 1


async def _ok():
    return True


# ---------- _try_collect: ранние ветки ----------


class _FakePool:
    def __init__(self, last_run=None, count=0):
        self._last_run = last_run
        self._count = count

    async def fetchval(self, sql, *a):
        if "MAX(started_at)" in sql:
            return self._last_run
        if "COUNT(*)" in sql:
            return self._count
        return 0


async def test_try_collect_db_unavailable(tmp_data, monkeypatch):
    def _boom():
        raise RuntimeError("no db")

    monkeypatch.setattr(sched, "get_pool", _boom)
    assert await sched._try_collect() == ("failed", 0)


async def test_try_collect_skips_when_recent(tmp_data, monkeypatch):
    from datetime import UTC, datetime

    recent = datetime.now(UTC)
    monkeypatch.setattr(sched, "get_pool", lambda: _FakePool(last_run=recent))
    monkeypatch.setattr(sched.config.settings, "SCHEDULER_COLLECT_INTERVAL_HOURS", 12)
    assert await sched._try_collect() == ("skipped", 0)


async def test_try_collect_force_bypasses_recent_gate(tmp_data, monkeypatch):
    """force=True при недавнем запуске не должен возвращать skipped сразу."""
    from datetime import UTC, datetime

    recent = datetime.now(UTC)
    monkeypatch.setattr(sched, "get_pool", lambda: _FakePool(last_run=recent))

    calls = {"n": 0}

    class _API:
        def search_vacancies(self, **_kw):
            calls["n"] += 1
            from src import Ok

            return Ok([])

    import src.parsing.api.hh_api as hh

    monkeypatch.setattr(hh, "HeadHunterAPI", _API)
    # Урезаем число регионов, чтобы не гонять 200 задач.
    monkeypatch.setattr(sched, "_refresh_history_snapshots", lambda: _ok())

    status, _ = await sched._try_collect(force=True, force_period_days=30)
    assert status == "collected-empty"
