"""Coverage tests for src/api_pkg/routers/pipeline.py REST surface.

Focuses on idempotency, not-found branches, active/list endpoints, cancel,
and WebSocket auth. No subprocess / real DB / network.
"""

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import src.api_pkg.routers.pipeline as pl
import src.pipeline.runner as runner_mod
from src.api_pkg import create_app
from src.models.api_responses import PipelineTaskStatus


async def _fake_startup(_app):
    return None


async def _noop():
    return None


class _Ok:
    @staticmethod
    def is_err() -> bool:
        return False

    @staticmethod
    def err():
        return None


def _make_task(
    task_id: str,
    status: str = "running",
    owner: str | None = "owner-1",
) -> PipelineTaskStatus:
    return PipelineTaskStatus(
        task_id=task_id,
        status=status,
        message="msg",
        started_at=time.time(),
        completed_at=None,
        output="secret",
        step=1,
        logs=["log"],
        owner_uid=owner,
    )


def _client(monkeypatch, *, uid="owner-1", role="teacher"):
    monkeypatch.setattr("src.api_pkg.startup.run_startup", _fake_startup)

    async def _fake_user(_request):
        return {"u": f"{uid}@t.com", "r": role, "uid": uid}

    monkeypatch.setattr("src.api_pkg.routers.auth.get_current_user", _fake_user)
    monkeypatch.setattr(runner_mod, "rebuild", lambda *a, **k: _Ok())
    monkeypatch.setattr(runner_mod, "run_full_pipeline", lambda *a, **k: _Ok())
    monkeypatch.setattr(runner_mod, "run_train_model", lambda *a, **k: _Ok())
    monkeypatch.setattr(pl, "_asave_tasks", _noop)
    monkeypatch.setattr(pl, "_persist_idempotency_sync", lambda: None)
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    pl.pipeline_tasks.clear()
    pl._idempotency_map.clear()
    pl._cancel_events.clear()
    pl._running_futures.clear()
    yield
    pl.pipeline_tasks.clear()
    pl._idempotency_map.clear()
    pl._cancel_events.clear()
    pl._running_futures.clear()


# --------------------------------------------------------------------------- /pipeline/task/{id}


def test_get_pipeline_task_status_404(monkeypatch):
    client = _client(monkeypatch)
    r = client.get("/api/pipeline/task/missing")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


def test_get_pipeline_task_status_admin_sees_details(monkeypatch):
    pl.pipeline_tasks["t1"] = _make_task("t1", status="completed", owner="owner-1")
    client = _client(monkeypatch, uid="admin-1", role="admin")
    body = client.get("/api/pipeline/task/t1").json()
    assert body["output"] == "secret"
    assert body["logs"] == ["log"]


# --------------------------------------------------------------------------- /pipeline/tasks


def test_list_pipeline_tasks_respects_limit(monkeypatch):
    for i in range(5):
        pl.pipeline_tasks[f"t{i}"] = _make_task(f"t{i}", status="completed", owner="owner-1")
    client = _client(monkeypatch, uid="owner-1", role="teacher")
    body = client.get("/api/pipeline/tasks?limit=2").json()
    assert body["total"] == 5
    assert len(body["tasks"]) == 2


def test_list_pipeline_tasks_admin_unredacted(monkeypatch):
    pl.pipeline_tasks["t1"] = _make_task("t1", status="completed", owner="owner-1")
    client = _client(monkeypatch, uid="admin-1", role="admin")
    body = client.get("/api/pipeline/tasks").json()
    assert body["tasks"][0]["output"] == "secret"


# --------------------------------------------------------------------------- /pipeline/active


def test_get_active_pipeline_task_returns_running(monkeypatch):
    pl.pipeline_tasks["t1"] = _make_task("t1", status="running", owner="owner-1")
    pl.pipeline_tasks["t2"] = _make_task("t2", status="completed", owner="owner-1")
    client = _client(monkeypatch)
    body = client.get("/api/pipeline/active").json()
    assert body["task_id"] == "t1"
    assert body["status"] == "running"


def test_get_active_pipeline_task_empty(monkeypatch):
    pl.pipeline_tasks["t1"] = _make_task("t1", status="completed", owner="owner-1")
    client = _client(monkeypatch)
    assert client.get("/api/pipeline/active").json() is None


# --------------------------------------------------------------------------- /pipeline/cancel/{id}


def test_cancel_pipeline_task_not_found_404(monkeypatch):
    client = _client(monkeypatch)
    assert client.post("/api/pipeline/cancel/nope").status_code == 404


def test_cancel_pipeline_task_not_running_400(monkeypatch):
    pl.pipeline_tasks["t1"] = _make_task("t1", status="completed", owner="owner-1")
    client = _client(monkeypatch, uid="owner-1", role="teacher")
    assert client.post("/api/pipeline/cancel/t1").status_code == 400


def test_cancel_pipeline_task_owner_allowed(monkeypatch):
    pl.pipeline_tasks["t1"] = _make_task("t1", status="running", owner="owner-1")
    pl._cancel_events["t1"] = MagicMock()
    client = _client(monkeypatch, uid="owner-1", role="teacher")
    r = client.post("/api/pipeline/cancel/t1")
    assert r.status_code == 200
    assert pl.pipeline_tasks["t1"].status == "cancelled"


# --------------------------------------------------------------------------- idempotency / rebuild


def test_rebuild_idempotency_returns_existing_task(monkeypatch):
    pl._idempotency_map["idem-1"] = "t-first"
    pl.pipeline_tasks["t-first"] = _make_task("t-first", status="running", owner="owner-1")
    client = _client(monkeypatch)
    r = client.post("/api/pipeline/rebuild", headers={"Idempotency-Key": "idem-1"})
    assert r.status_code == 200
    body = r.json()
    assert "Duplicate" in body["message"]
    assert "t-first" in body["output"]


def test_rebuild_creates_new_task_with_idempotency(monkeypatch):
    client = _client(monkeypatch)
    r = client.post("/api/pipeline/rebuild", headers={"Idempotency-Key": "idem-new"})
    assert r.status_code == 200
    body = r.json()
    assert "rebuild" in body["command"].lower()
    assert "t-first" not in body["output"]
    created = list(pl.pipeline_tasks.values())[0].task_id
    assert pl._idempotency_map["idem-new"] == created


# --------------------------------------------------------------------------- /pipeline/{action} duplicate not-found


def test_full_cycle_duplicate_idempotency(monkeypatch):
    monkeypatch.setattr("src.pipeline.background_collector.is_scheduler_busy", lambda: False)
    pl._idempotency_map["idem-fc"] = "fc-1"
    pl.pipeline_tasks["fc-1"] = _make_task("fc-1", status="running", owner="owner-1")
    client = _client(monkeypatch)
    r = client.post("/api/pipeline/full-cycle", params={"regions": "0"}, headers={"Idempotency-Key": "idem-fc"})
    assert r.status_code == 200
    assert "Duplicate" in r.json()["message"]


def test_gap_analysis_rejected_when_scheduler_busy(monkeypatch):
    monkeypatch.setattr("src.pipeline.background_collector.is_scheduler_busy", lambda: True)
    client = _client(monkeypatch)
    r = client.post("/api/pipeline/gap-analysis", headers={"Idempotency-Key": "busy"})
    assert r.status_code == 409


# --------------------------------------------------------------------------- /pipeline/gap-progress/{id}


def test_gap_progress_no_file_returns_not_exists(monkeypatch):
    monkeypatch.setattr(pl, "GAP_PROGRESS_FILE", MagicMock(exists=lambda: False))
    client = _client(monkeypatch)
    body = client.get("/api/pipeline/gap-progress/any").json()
    assert body["exists"] is False


def test_gap_progress_with_file_returns_data(monkeypatch, tmp_path):
    path = tmp_path / "gap.json"
    path.write_text(json.dumps({"pct": 42, "message": "m", "stage": "s"}), encoding="utf-8")
    monkeypatch.setattr(pl, "GAP_PROGRESS_FILE", path)
    client = _client(monkeypatch)
    body = client.get("/api/pipeline/gap-progress/any").json()
    assert body["exists"] is True
    assert body["pct"] == 42


# --------------------------------------------------------------------------- /pipeline/ws


def test_pipeline_ws_rejects_without_token(monkeypatch):
    monkeypatch.setattr("src.api_pkg.startup.run_startup", _fake_startup)
    client = TestClient(create_app(), raise_server_exceptions=False)
    with pytest.raises(WebSocketDisconnect) as exc_ctx:
        with client.websocket_connect("/api/pipeline/ws"):
            pass
    assert exc_ctx.value.code == 4001


def test_pipeline_ws_rejects_invalid_token(monkeypatch):
    async def _reject(_token):
        return None

    monkeypatch.setattr("src.api_pkg.startup.run_startup", _fake_startup)
    monkeypatch.setattr("src.api_pkg.routers.auth.resolve_session", _reject)
    client = TestClient(create_app(), raise_server_exceptions=False)
    with pytest.raises(WebSocketDisconnect) as exc_ctx:
        with client.websocket_connect("/api/pipeline/ws", headers={"Authorization": "Bearer bad"}):
            pass
    assert exc_ctx.value.code == 4001


def test_pipeline_ws_accepts_valid_token_and_sends_progress(monkeypatch, tmp_path):
    async def _resolve(token):
        if token == "valid-token":
            return {"uid": "u-1", "u": "u-1@t.com", "r": "teacher", "t": 9999999999}
        return None

    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps({"pct": 12, "message": "m", "stage": "s"}), encoding="utf-8")
    monkeypatch.setattr("src.api_pkg.startup.run_startup", _fake_startup)
    monkeypatch.setattr("src.api_pkg.routers.auth.resolve_session", _resolve)
    monkeypatch.setattr(pl, "PIPELINE_PROGRESS_FILE", path)
    monkeypatch.setattr(pl, "GAP_PROGRESS_FILE", tmp_path / "gap.json")
    client = TestClient(create_app(), raise_server_exceptions=False)
    with client.websocket_connect(
        "/api/pipeline/ws", headers={"Authorization": "Bearer valid-token"}
    ) as ws:
        # Обработчик отдаёт прогресс сразу и только потом уходит в sleep(2).
        # Достаточно прочитать одно сообщение: выход из контекста закрывает
        # сокет, и серверный цикл завершается по WebSocketDisconnect.
        data = json.loads(ws.receive_text())
        assert data["pct"] == 12
        assert data["message"] == "m"
