"""Route-level coverage for src.api_pkg.routers.admin."""

from __future__ import annotations

import asyncio
import contextlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api_pkg.request_logger as request_logger_mod
import src.api_pkg.routers.admin as admin_mod
import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.pipeline as pipeline_mod
import src.api_pkg.skill_suggestions as suggestions_mod
import src.api_pkg.student_actions as student_actions_mod
import src.cli.backup_db as backup_db_mod
import src.cli.create_user as create_user_mod
import src.cli.embeddings as embeddings_mod
import src.cli.extend_skills as extend_skills_mod
import src.cli.import_students as import_students_mod
import src.cli.taxonomy_audit as taxonomy_mod
import src.config as config_mod
import src.database as database_mod
import src.db as db_mod
import src.monitoring.metrics as metrics_mod
import src.pipeline.background_collector as bg_collector_mod


class _FakeResult:
    def __init__(self, rows, *, scalar=None):
        self._rows = rows
        self._scalar = scalar

    def fetchall(self):
        return self._rows

    def all(self):
        return self._rows

    def scalar_one(self):
        return self._scalar

    def scalars(self):
        return self

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, *, users=None, directions=None, user_by_id=None):
        self.users = users or []
        self.directions = directions or []
        self.user_by_id = user_by_id or {}
        self.queries = []
        self.added = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        self.queries.append(sql)
        if "COUNT(*)" in sql:
            return _FakeResult([], scalar=len(self.users))
        if "FROM users" in sql and "LIMIT" in sql:
            return _FakeResult(self.users[params["offset"] : params["offset"] + params["limit"]])
        if "user_directions" in sql or "dir_code" in sql:
            return _FakeResult(self.directions)
        return _FakeResult([])

    async def get(self, model, key):
        return self.user_by_id.get(str(key))

    async def commit(self):
        pass

    def add(self, obj):
        self.added.append(obj)


async def _fake_current_user(_request):
    return {"u": "admin@example.com", "r": "admin"}


async def _fake_logs_merged(**_kwargs):
    return [{"path": "/x", "action": "krm.recommendation.add"}]


async def _fake_actions(**_kwargs):
    return [{"user": "student@example.com", "action": "save_profile"}]


async def _fake_actor(_request):
    return "admin@example.com", "admin"


async def _fake_coroutine(*_args, **_kwargs):
    return None


@pytest.fixture(autouse=True)
def _reset_limiter():
    with contextlib.suppress(Exception):
        admin_mod.limiter._storage.reset()
    yield


def _admin_client(monkeypatch, tmp_path, *, users=None, directions=None, user_by_id=None, user_return=None):
    monkeypatch.setattr(auth_mod, "get_current_user", user_return or _fake_current_user)
    monkeypatch.setattr(request_logger_mod, "audit_action", AsyncMock(return_value={}))
    monkeypatch.setattr(request_logger_mod, "actor_from_request", _fake_actor)
    monkeypatch.setattr(request_logger_mod, "get_logs_merged", _fake_logs_merged)
    monkeypatch.setattr(request_logger_mod, "get_logs_by_user", lambda: {})
    monkeypatch.setattr(student_actions_mod, "get_actions", _fake_actions)
    monkeypatch.setattr(db_mod, "get_pool", lambda: None)
    monkeypatch.setattr(metrics_mod, "get_metrics", lambda: (b"", "text/plain"))
    monkeypatch.setattr(config_mod, "LOG_FILE", tmp_path / "backend.log")
    monkeypatch.setattr(admin_mod, "IT_SKILLS_PATH", _ensure_it_skills(tmp_path))
    monkeypatch.setattr(taxonomy_mod, "TAXONOMY_PATH", _ensure_taxonomy(tmp_path))
    monkeypatch.setattr(
        admin_mod,
        "_load_whitelist",
        lambda: sorted(json.loads(admin_mod.IT_SKILLS_PATH.read_text(encoding="utf-8"))),
    )

    monkeypatch.setattr(database_mod, "async_session_factory", lambda: _FakeSession(
        users=users or [], directions=directions or [], user_by_id=user_by_id or {},
    ))

    app = FastAPI()
    app.state.limiter = admin_mod.limiter
    app.include_router(admin_mod.router)
    return TestClient(app, raise_server_exceptions=False)


def _ensure_it_skills(tmp_path):
    path = tmp_path / "reference" / "it_skills.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(["python", "sql"]), encoding="utf-8")
    return path


def _ensure_taxonomy(tmp_path):
    path = tmp_path / "reference" / "skill_taxonomy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"categories": {"backend": {"skills": []}}}), encoding="utf-8")
    return path


# ---------- whitelist ----------


def test_get_whitelist(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/whitelist")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert "python" in body["skills"]


def test_whitelist_add_new(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/whitelist/add", json={"skills": ["docker"]})
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == 1
    assert "docker" in json.loads(admin_mod.IT_SKILLS_PATH.read_text(encoding="utf-8"))


def test_whitelist_add_noop(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/whitelist/add", json={"skills": ["python"]})
    assert r.status_code == 200
    assert r.json()["added"] == 0


def test_whitelist_remove(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/whitelist/remove", json={"skills": ["python"]})
    assert r.status_code == 200
    body = r.json()
    assert body["removed"] == 1
    assert "python" not in json.loads(admin_mod.IT_SKILLS_PATH.read_text(encoding="utf-8"))


def test_whitelist_remove_noop(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/whitelist/remove", json={"skills": ["docker"]})
    assert r.status_code == 200
    assert r.json()["removed"] == 0


def test_whitelist_backup(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/whitelist/backup")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert Path(body["backup_path"]).exists()


# ---------- students ----------


def test_list_students_empty_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "no_data")
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/students")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_list_students_with_file(monkeypatch, tmp_path):
    data_dir = tmp_path / "app_data"
    students_dir = data_dir / "students"
    students_dir.mkdir(parents=True)
    (students_dir / "ivanov.json").write_text(
        json.dumps({"competencias_actuales": ["python", "docker"]}), encoding="utf-8"
    )
    monkeypatch.setattr(config_mod, "DATA_DIR", data_dir)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/students")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["students"][0]["skills_count"] == 2


# ---------- pipeline trigger ----------


def test_trigger_pipeline_success(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_mod, "run_pipeline_task", AsyncMock())
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/pipeline/trigger", json={"action": "full-cycle"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "started"
    assert body["task_id"].startswith("full-cycle_")


def test_trigger_pipeline_unknown_action(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_mod, "run_pipeline_task", AsyncMock())
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/pipeline/trigger", json={"action": "bad"})
    assert r.status_code == 400
    assert "Unknown action" in r.json()["detail"]


# ---------- export ----------


def test_export_excel_no_data(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "DATA_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config_mod, "DATA_PROCESSED_DIR", tmp_path / "processed")
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/export/excel")
    assert r.status_code == 404


@pytest.mark.xfail(reason="openpyxl may be missing in some envs")
def test_export_excel_success(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    detailed = processed / "hh_vacancies_detailed.json"
    detailed.write_text(json.dumps([{
        "id": "1", "name": "dev", "employer": {"name": "Corp"}, "area": {"name": "City"},
        "salary": None, "experience": {"id": "between1And3", "name": "1-3"},
        "key_skills": [{"name": "python"}], "alternate_url": "http://x",
    }]), encoding="utf-8")
    monkeypatch.setattr(config_mod, "DATA_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config_mod, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(config_mod, "REPORTS_DIR", tmp_path / "reports")
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/export/excel")
    assert r.status_code == 200
    assert r.headers["content-disposition"].startswith("attachment")


def test_export_full_report_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "DATA_RESULT_DIR", tmp_path / "empty_result")
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/export/full-report")
    assert r.status_code == 404


def test_export_full_report_success(monkeypatch, tmp_path):
    result_dir = tmp_path / "result"
    (result_dir / "sub").mkdir(parents=True)
    (result_dir / "sub" / "data.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config_mod, "DATA_RESULT_DIR", result_dir)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/export/full-report")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    zipfile.is_zipfile(BytesIO(r.content))


# ---------- directions ----------


def test_admin_directions(monkeypatch, tmp_path):
    import src.api_pkg.routers.teacher as teacher_mod

    monkeypatch.setattr(teacher_mod, "_list_krm_directions", lambda: ["09.03.02", "09.03.04"])
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/directions")
    assert r.status_code == 200
    assert r.json() == ["09.03.02", "09.03.04"]


# ---------- users directions ----------


def test_user_directions_success(monkeypatch, tmp_path):
    uid = "11111111-1111-1111-1111-111111111111"
    client = _admin_client(
        monkeypatch, tmp_path, directions=[("backend",), ("frontend",)]
    )
    r = client.get(f"/admin/users/{uid}/directions")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == uid
    assert body["directions"] == ["backend", "frontend"]


def test_user_directions_bad_uuid(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/users/not-a-uuid/directions")
    assert r.status_code == 400


def test_set_user_directions_success(monkeypatch, tmp_path):
    uid = "11111111-1111-1111-1111-111111111111"
    user = SimpleNamespace(id=uid, email="user@example.com", role="student")
    client = _admin_client(monkeypatch, tmp_path, user_by_id={uid: user})
    r = client.put(f"/admin/users/{uid}/directions", json={"directions": ["09.03.02"]})
    assert r.status_code == 200
    assert r.json()["directions"] == ["09.03.02"]


def test_set_user_directions_not_found(monkeypatch, tmp_path):
    uid = "11111111-1111-1111-1111-111111111111"
    client = _admin_client(monkeypatch, tmp_path, user_by_id={})
    r = client.put(f"/admin/users/{uid}/directions", json={"directions": ["09.03.02"]})
    assert r.status_code == 404


# ---------- monitoring ----------


def test_admin_monitoring(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics_mod, "get_metrics", lambda: (b"# HELP x y\nx 1\n", "text/plain"))
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/monitoring")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_admin_monitoring_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        metrics_mod,
        "get_metrics",
        lambda: (b"# HELP api_request_duration_seconds_bucket x\n", "text/plain"),
    )
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/monitoring/summary")
    assert r.status_code == 200
    body = r.json()
    assert "latency_p95_by_endpoint" in body
    assert body["errors_1h"] == {"4xx": 0, "5xx": 0}


# ---------- logs ----------


def test_admin_logs(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/logs")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1


def test_admin_logs_user_all(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/logs?user=all&limit=10")
    assert r.status_code == 200
    assert r.json()["total"] == 1


@pytest.mark.parametrize("qs", ["limit=0", "limit=1001", "limit=-1"])
def test_admin_logs_bad_params(monkeypatch, tmp_path, qs):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get(f"/admin/logs?{qs}")
    assert r.status_code == 422


def test_admin_student_actions(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/student-actions")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["actions"][0]["user"] == "student@example.com"


def test_admin_logs_file_empty(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/logs/file")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_admin_logs_file_level_filter(monkeypatch, tmp_path):
    log_path = tmp_path / "backend.log"
    log_path.write_text(
        "level=info INFO hello\nlevel=error ERROR boom\nlevel=info INFO world\n", encoding="utf-8"
    )
    monkeypatch.setattr(config_mod, "LOG_FILE", log_path)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/logs/file?level=error")
    assert r.status_code == 200
    lines = r.json()["lines"]
    assert len(lines) == 1
    assert "ERROR boom" in lines[0]


# ---------- db management ----------


def test_admin_seed_db(monkeypatch, tmp_path):
    monkeypatch.setattr(admin_mod, "_run_seed", lambda drop: None)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/db/seed", json={"drop": False})
    assert r.status_code == 200
    assert r.json()["status"] == "started"


def test_admin_create_user(monkeypatch, tmp_path):
    called = {}

    async def fake_main(email, password, role, name, directions):
        called["args"] = (email, password, role, name, directions)

    monkeypatch.setattr(create_user_mod, "main", fake_main)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/users/create", json={
        "email": "a@b.com", "password": "secret", "role": "teacher", "name": "A",
    })
    assert r.status_code == 200
    assert r.json()["email"] == "a@b.com"


def test_admin_generate_embeddings(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings_mod, "main", AsyncMock())
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/embeddings/generate", json={"force": False})
    assert r.status_code == 200
    assert r.json()["status"] == "started"


def test_admin_import_students(monkeypatch, tmp_path):
    monkeypatch.setattr(import_students_mod, "main", AsyncMock())
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/students/import", json=[{
        "full_name": "Ivan", "group_name": "101", "direction_code": "09.03.02", "skills": "python",
    }])
    assert r.status_code == 200
    assert r.json()["imported"] == 1


def test_admin_extend_skills(monkeypatch, tmp_path):
    monkeypatch.setattr(extend_skills_mod, "main", AsyncMock())
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/skills/extend")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_admin_export_db(monkeypatch, tmp_path):
    monkeypatch.setattr(admin_mod, "_run_export", lambda: None)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/export/db")
    assert r.status_code == 200
    assert r.json()["status"] == "started"


def test_admin_backup_db(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(backup_db_mod, "main", lambda: called.append(True))
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/db/backup")
    assert r.status_code == 200
    assert r.json()["message"] == "Backup created"
    assert called


# ---------- taxonomy ----------


def test_admin_skills_uncategorized(monkeypatch, tmp_path):
    monkeypatch.setattr(taxonomy_mod, "load_taxonomy", lambda: {"categories": {"backend": {"skills": ["python"]}}})
    monkeypatch.setattr(taxonomy_mod, "load_it_skills", lambda: {"python", "docker"})
    monkeypatch.setattr(taxonomy_mod, "taxonomy_skill_set", lambda tax: {"python"})
    monkeypatch.setattr(taxonomy_mod, "build_prototypes", lambda tax: {})
    monkeypatch.setattr(taxonomy_mod, "suggest_categories", lambda skills, protos: {"docker": ("backend", 0.5)})
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/skills/uncategorized")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["skill"] == "docker"


def test_admin_skills_uncategorized_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(taxonomy_mod, "load_taxonomy", lambda: {"categories": {"backend": {"skills": ["python"]}}})
    monkeypatch.setattr(taxonomy_mod, "load_it_skills", lambda: {"python"})
    monkeypatch.setattr(taxonomy_mod, "taxonomy_skill_set", lambda tax: {"python"})
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/skills/uncategorized")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_admin_skills_categorize_success(monkeypatch, tmp_path):
    monkeypatch.setattr(taxonomy_mod, "load_taxonomy", lambda: {"categories": {"backend": {"skills": []}}})
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/skills/categorize", json={"assignments": [{"skill": "docker", "category": "backend"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == 1
    assert body["invalid"] == []


def test_admin_skills_categorize_invalid_category(monkeypatch, tmp_path):
    monkeypatch.setattr(taxonomy_mod, "load_taxonomy", lambda: {"categories": {"backend": {"skills": []}}})
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/skills/categorize", json={"assignments": [{"skill": "docker", "category": "absent"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == 0
    assert body["invalid"][0]["category"] == "absent"


def test_admin_skills_suggestions(monkeypatch, tmp_path):
    suggestions = [{"id": "s1", "skill": "go", "status": "pending"}]
    monkeypatch.setattr(suggestions_mod, "load_all", lambda: suggestions)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/skills/suggestions?status=pending")
    assert r.status_code == 200
    assert r.json()["suggestions"] == suggestions


def _setup_approve(monkeypatch, tmp_path, suggestion, taxonomy_data, skills_data):
    monkeypatch.setattr(suggestions_mod, "load_all", lambda: [suggestion])
    monkeypatch.setattr(suggestions_mod, "decide", lambda sid, approved, decided_by: {"id": sid, "status": "approved"})
    monkeypatch.setattr(taxonomy_mod, "load_taxonomy", lambda: taxonomy_data)
    # Redirect the module paths BEFORE writing: these are real files under
    # data/reference/, and writing to them here used to wipe the shipped data
    # whenever this test module ran.
    monkeypatch.setattr(admin_mod, "IT_SKILLS_PATH", _ensure_it_skills(tmp_path))
    monkeypatch.setattr(taxonomy_mod, "TAXONOMY_PATH", _ensure_taxonomy(tmp_path))
    admin_mod.IT_SKILLS_PATH.write_text(json.dumps(skills_data), encoding="utf-8")
    taxonomy_mod.TAXONOMY_PATH.write_text(json.dumps(taxonomy_data), encoding="utf-8")


def test_admin_skills_approve(monkeypatch, tmp_path):
    _setup_approve(
        monkeypatch, tmp_path,
        {"id": "s1", "skill": "go", "status": "pending"},
        {"categories": {"backend": {"skills": []}}},
        [],
    )
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/skills/suggestions/s1/approve", json={"category": "backend"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["category"] == "backend"
    assert "go" in json.loads(admin_mod.IT_SKILLS_PATH.read_text(encoding="utf-8"))


def test_admin_skills_approve_unknown_category(monkeypatch, tmp_path):
    _setup_approve(
        monkeypatch, tmp_path,
        {"id": "s1", "skill": "go", "status": "pending"},
        {"categories": {"backend": {"skills": []}}},
        [],
    )
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/skills/suggestions/s1/approve", json={"category": "missing"})
    assert r.status_code == 400


def test_admin_skills_approve_not_pending(monkeypatch, tmp_path):
    _setup_approve(
        monkeypatch, tmp_path,
        {"id": "s1", "skill": "go", "status": "approved"},
        {"categories": {"backend": {"skills": []}}},
        [],
    )
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/skills/suggestions/s1/approve", json={"category": "backend"})
    assert r.status_code == 404


def test_admin_skills_reject(monkeypatch, tmp_path):
    monkeypatch.setattr(suggestions_mod, "decide", lambda sid, approved, decided_by: {"id": sid, "status": "rejected"})
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/skills/suggestions/s1/reject")
    assert r.status_code == 200
    assert r.json()["suggestion"]["status"] == "rejected"


def test_admin_skills_reject_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(suggestions_mod, "decide", lambda sid, approved, decided_by: None)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/skills/suggestions/s1/reject")
    assert r.status_code == 404


# ---------- scheduler ----------


def test_scheduler_status(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "cfg")
    monkeypatch.setattr(bg_collector_mod, "scheduler_status", lambda: {"running": False})
    client = _admin_client(monkeypatch, tmp_path)
    r = client.get("/admin/scheduler/status")
    assert r.status_code == 200
    assert r.json()["running"] is False


def test_scheduler_settings(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "cfg")
    monkeypatch.setattr(
        bg_collector_mod, "save_scheduler_settings",
        lambda patch: captured.__setitem__("patch", patch),
    )
    monkeypatch.setattr(bg_collector_mod, "scheduler_status", lambda: {"running": True})
    client = _admin_client(monkeypatch, tmp_path)
    r = client.put("/admin/scheduler/settings", json={"collector_enabled": True})
    assert r.status_code == 200
    assert r.json()["running"] is True
    assert captured["patch"] == {"collector_enabled": True}


def test_scheduler_collect_now(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "cfg")
    real_create_task = asyncio.create_task

    def fake_create_task(coro, *, name=None):
        real_create_task(_fake_coroutine())
        return None

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/scheduler/collect/now")
    assert r.status_code == 200
    assert r.json()["job"] == "collect"


def test_scheduler_collect_now_busy(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "cfg")
    monkeypatch.setattr(bg_collector_mod, "is_scheduler_busy", lambda: True)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/scheduler/collect/now")
    assert r.status_code == 409


def test_scheduler_gap_now(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "cfg")
    real_create_task = asyncio.create_task

    def fake_create_task(coro, *, name=None):
        real_create_task(_fake_coroutine())
        return None

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/scheduler/gap/now")
    assert r.status_code == 200
    assert r.json()["job"] == "gap"


def test_scheduler_gap_now_busy(monkeypatch, tmp_path):
    monkeypatch.setattr(config_mod, "DATA_DIR", tmp_path / "cfg")
    monkeypatch.setattr(bg_collector_mod, "is_scheduler_busy", lambda: True)
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/admin/scheduler/gap/now")
    assert r.status_code == 409


# ---------- frontend log ----------


def test_frontend_log(monkeypatch, tmp_path):
    client = _admin_client(monkeypatch, tmp_path)
    r = client.post("/log", json={"action": "click", "detail": "save"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
