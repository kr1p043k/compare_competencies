"""Владение задачами пайплайна: чужой прогон нельзя отменить и прочитать.

Задачи - глобальный ресурс процесса. До фикса любой teacher мог отменить
пересборку, запущенную другим пользователем, зная task_id, а также прочитать
output/logs (работодатели, регионы, запросы).
"""
import time

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.pipeline as pl
from src.api_pkg import create_app
from src.models.api_responses import PipelineTaskStatus

TASK = "full-cycle-2026-11-03-owner-test"


@pytest.fixture(autouse=True)
def _clean_tasks():
    """Не трогаем настоящие задачи процесса."""
    saved = dict(pl.pipeline_tasks)
    pl.pipeline_tasks.clear()
    yield
    pl.pipeline_tasks.clear()
    pl.pipeline_tasks.update(saved)


def _task(owner: str | None, status: str = "running") -> PipelineTaskStatus:
    return PipelineTaskStatus(
        task_id=TASK,
        status=status,
        message="Запуск пересборки...",
        started_at=time.time(),
        completed_at=None,
        output="secret output: HH responses",
        step=1,
        logs=["log line with employer name"],
        owner_uid=owner,
    )


def _client(monkeypatch, uid: str, role: str = "teacher"):
    app = create_app()

    async def _user(_request):
        return {"uid": uid, "u": f"{uid}@example.com", "r": role, "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _user)
    # Не писать в реальный data/cache/pipeline_tasks.json
    monkeypatch.setattr(pl, "_asave_tasks", lambda: _noop())
    return TestClient(app, raise_server_exceptions=False)


async def _noop():
    return None


class TestCancelOwnership:
    def test_owner_can_cancel(self, monkeypatch):
        pl.pipeline_tasks[TASK] = _task("owner-1")
        client = _client(monkeypatch, "owner-1")
        r = client.post(f"/api/pipeline/cancel/{TASK}")
        assert r.status_code == 200, r.text
        assert pl.pipeline_tasks[TASK].status == "cancelled"

    def test_other_teacher_cannot_cancel(self, monkeypatch):
        pl.pipeline_tasks[TASK] = _task("owner-1")
        client = _client(monkeypatch, "intruder-2")
        r = client.post(f"/api/pipeline/cancel/{TASK}")
        assert r.status_code == 403, r.text
        assert pl.pipeline_tasks[TASK].status == "running", "статус не должен меняться"

    def test_admin_can_cancel_any(self, monkeypatch):
        pl.pipeline_tasks[TASK] = _task("owner-1")
        client = _client(monkeypatch, "admin-9", role="admin")
        r = client.post(f"/api/pipeline/cancel/{TASK}")
        assert r.status_code == 200, r.text

    def test_ownerless_task_is_admin_only(self, monkeypatch):
        """Записи из JSON до появления владельца: teacher не управляет."""
        pl.pipeline_tasks[TASK] = _task(None)
        teacher = _client(monkeypatch, "teacher-1")
        assert teacher.post(f"/api/pipeline/cancel/{TASK}").status_code == 403
        assert pl.pipeline_tasks[TASK].status == "running"

        admin = _client(monkeypatch, "admin-1", role="admin")
        assert admin.post(f"/api/pipeline/cancel/{TASK}").status_code == 200

    def test_unknown_task_is_404(self, monkeypatch):
        client = _client(monkeypatch, "owner-1")
        assert client.post("/api/pipeline/cancel/nope").status_code == 404


class TestTaskStatusVisibility:
    def test_owner_sees_output_and_logs(self, monkeypatch):
        pl.pipeline_tasks[TASK] = _task("owner-1")
        client = _client(monkeypatch, "owner-1")
        body = client.get(f"/api/pipeline/task/{TASK}").json()
        assert body["output"] == "secret output: HH responses"
        assert body["logs"], "владелец должен видеть логи"

    def test_other_teacher_sees_progress_but_no_details(self, monkeypatch):
        pl.pipeline_tasks[TASK] = _task("owner-1")
        client = _client(monkeypatch, "intruder-2")
        r = client.get(f"/api/pipeline/task/{TASK}")
        assert r.status_code == 200
        body = r.json()
        # прогресс доступен всем
        assert body["status"] == "running"
        assert body["step"] == 1
        assert body["message"]
        # деталей чужого сбора нет
        assert body["output"] is None
        assert body["logs"] == []

    def test_admin_sees_details(self, monkeypatch):
        pl.pipeline_tasks[TASK] = _task("owner-1")
        client = _client(monkeypatch, "admin-1", role="admin")
        body = client.get(f"/api/pipeline/task/{TASK}").json()
        assert body["output"] is not None

    def test_task_list_redacts_details_for_others(self, monkeypatch):
        pl.pipeline_tasks[TASK] = _task("owner-1")
        client = _client(monkeypatch, "intruder-2")
        tasks = client.get("/api/pipeline/tasks").json()["tasks"]
        assert tasks, "список не должен быть пустым"
        row = next(t for t in tasks if t["task_id"] == TASK)
        assert row["output"] is None
        assert row["logs"] == []
