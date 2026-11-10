"""Graceful shutdown: фоновые флашеры останавливаются и сбрасывают буфер.

Проверяем без БД: подменяем _flush_to_db счётчиком, стартуем/останавливаем
задачу и убеждаемся, что она отменена, а финальный flush выполнен.
"""

from __future__ import annotations

import asyncio

import pytest

import src.api_pkg.request_logger as rl
import src.notifications.system as nsys


@pytest.fixture(autouse=True)
def _reset_tasks():
    rl._flusher_task = None
    nsys._flusher_task = None
    yield
    rl._flusher_task = None
    nsys._flusher_task = None


async def test_log_flusher_start_is_idempotent(monkeypatch):
    monkeypatch.setattr(rl, "_flush_to_db", _noop)
    rl.start_log_flusher()
    first = rl._flusher_task
    rl.start_log_flusher()
    assert first is rl._flusher_task, "повторный start не должен создавать вторую задачу"
    assert first is not None and not first.done()
    rl._flusher_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await rl._flusher_task


async def test_stop_log_flusher_cancels_and_flushes(monkeypatch):
    calls = {"n": 0}

    async def _count():
        calls["n"] += 1

    monkeypatch.setattr(rl, "_flush_to_db", _count)
    rl.start_log_flusher()
    task = rl._flusher_task
    assert task is not None

    await rl.stop_log_flusher()

    assert rl._flusher_task is None
    assert task.cancelled() or task.done()
    assert calls["n"] == 1, "финальный flush должен выполниться один раз"


async def test_stop_log_flusher_noop_when_never_started(monkeypatch):
    monkeypatch.setattr(rl, "_flush_to_db", _noop)
    await rl.stop_log_flusher()  # не должно падать
    assert rl._flusher_task is None


async def test_system_error_flusher_start_stop(monkeypatch):
    calls = {"n": 0}

    async def _count():
        calls["n"] += 1

    monkeypatch.setattr(nsys, "_flush_to_db", _count)
    nsys.start_system_error_flusher()
    first = nsys._flusher_task
    nsys.start_system_error_flusher()
    assert first is nsys._flusher_task

    await nsys.stop_system_error_flusher()
    assert nsys._flusher_task is None
    assert calls["n"] == 1


async def _noop() -> None:
    return None
