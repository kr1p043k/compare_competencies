"""Регресс: run_pipeline_task не падает с NameError и не теряет владельца."""

import asyncio

import pytest

import src.pipeline.runner as runner_mod


class _Ok:
    @staticmethod
    def is_err() -> bool:
        return False

    @staticmethod
    def err():
        return None


@pytest.fixture(autouse=True)
def clean_state():
    import src.api_pkg.routers.pipeline as pl

    pl._cancel_events.clear()
    pl.pipeline_tasks.clear()
    yield
    pl._cancel_events.clear()
    pl.pipeline_tasks.clear()


def _patch(monkeypatch, pl):
    async def _noop_save():
        return None

    monkeypatch.setattr(pl, "_asave_tasks", _noop_save)
    monkeypatch.setattr(runner_mod, "rebuild", lambda *a, **k: _Ok())


def test_keeps_owner_and_reaches_runner(monkeypatch):
    """Владелец, созданный обработчиком, не должен теряться."""
    import src.api_pkg.routers.pipeline as pl

    _patch(monkeypatch, pl)
    called = {"rebuild": False}

    def _rebuild(*_a, **_kw):
        called["rebuild"] = True
        return _Ok()

    monkeypatch.setattr(runner_mod, "rebuild", _rebuild)
    pl.pipeline_tasks["t1"] = pl._make_task_status(
        "t1", "running", "Запуск пересборки...", 1.0, step=0, owner_uid="user-42"
    )

    asyncio.run(pl.run_pipeline_task(pl.PipelineAction.REBUILD, "t1"))

    task = pl.pipeline_tasks["t1"]
    assert task.owner_uid == "user-42"
    assert called["rebuild"] is True
    assert task.status == "completed"


def test_no_name_error_without_request(monkeypatch):
    """Раньше здесь был NameError: request в фоновой задаче не существует."""
    import src.api_pkg.routers.pipeline as pl

    _patch(monkeypatch, pl)
    pl.pipeline_tasks["t2"] = pl._make_task_status(
        "t2", "running", "Запуск...", 1.0, step=0, owner_uid=None
    )

    asyncio.run(pl.run_pipeline_task(pl.PipelineAction.REBUILD, "t2"))

    assert pl.pipeline_tasks["t2"].owner_uid is None


def test_creates_task_when_caller_did_not(monkeypatch):
    """Если статуса ещё нет, задача создаётся сама."""
    import src.api_pkg.routers.pipeline as pl

    _patch(monkeypatch, pl)

    asyncio.run(
        pl.run_pipeline_task(pl.PipelineAction.REBUILD, "t3", owner_uid="user-7")
    )

    assert pl.pipeline_tasks["t3"].owner_uid == "user-7"
