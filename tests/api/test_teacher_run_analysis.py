"""Запуск teacher-analysis кнопкой из кабинета преподавателя.

Регрессия: эндпоинт создавал pipeline_run без владельца и без направления.
Статус читается через /teacher/rpd/status/{run_id}, а `_require_run_access`
пускает владельца только при заполненном owner_id (иначе fail closed - 403).
В результате каждый опрос статуса отдавал 403, фронт писал «нет связи» и
кнопка «Запустить анализ» висела до таймаута.

Проверяем, что запуск пишет owner_id + direction_code и что статус этого
запуска доступен тому же преподавателю.
"""
import uuid

from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.rpd as rpd_mod
import src.api_pkg.routers.zun as zun_mod
import src.database as database_mod
import src.pipeline.db_writer as db_writer
from src.api_pkg import create_app

OWN = "09.03.02"
TEACHER_UID = "u-42"
RUN_ID = str(uuid.uuid4())


class FakeRun:
    def __init__(self, direction_code=None, owner_id=None, status="running"):
        self.id = RUN_ID
        self.action = "teacher-analysis"
        self.status = status
        self.direction_code = direction_code
        self.owner_id = owner_id
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.stats = {"stage": "analysis"}


class FakeSession:
    def __init__(self, run):
        self._run = run

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, _model, _run_id):
        return self._run


def _client(monkeypatch, created: dict, run):
    app = create_app()

    async def _fake_user(_request):
        return {"uid": TEACHER_UID, "u": "t@example.com", "r": "teacher", "t": None}

    async def _fake_dirs(_user):
        return [OWN]

    async def _fake_create(action, owner_id=None, direction_code=None):
        created.update(action=action, owner_id=owner_id, direction_code=direction_code)
        return RUN_ID

    async def _fake_update(_run_id, _stats):
        return None

    async def _fake_run_cli(*_a, **_kw):
        return 0, "OK"

    async def _fake_complete(*_a, **_kw):
        return None

    def _fake_session_factory():
        return FakeSession(run)

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    monkeypatch.setattr(zun_mod, "_user_dir_codes", _fake_dirs)
    monkeypatch.setattr(database_mod, "async_session_factory", _fake_session_factory)
    monkeypatch.setattr(db_writer, "create_pipeline_run", _fake_create)
    monkeypatch.setattr(rpd_mod, "_update_run", _fake_update)
    monkeypatch.setattr(rpd_mod, "_run_cli", _fake_run_cli)
    monkeypatch.setattr(db_writer, "complete_pipeline_run", _fake_complete)
    return TestClient(app)


def test_run_analysis_records_owner_and_direction(monkeypatch):
    created: dict = {}
    client = _client(monkeypatch, created, FakeRun(direction_code=OWN, owner_id=TEACHER_UID))
    r = client.post("/api/teacher/krm/run-analysis", params={"dir_code": OWN})
    assert r.status_code == 200, r.text
    assert r.json()["run_id"] == RUN_ID
    assert created["action"] == "teacher-analysis"
    # Без этих двух полей опрос статуса отдаёт 403 и кнопка «висит».
    assert created["owner_id"] == TEACHER_UID
    assert created["direction_code"] == OWN


def test_run_analysis_status_readable_by_owner(monkeypatch):
    created: dict = {}
    run = FakeRun(direction_code=OWN, owner_id=TEACHER_UID)
    client = _client(monkeypatch, created, run)
    client.post("/api/teacher/krm/run-analysis", params={"dir_code": OWN})
    r = client.get(f"/api/teacher/rpd/status/{RUN_ID}")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "running"


def test_run_analysis_invalid_direction_rejected(monkeypatch):
    created: dict = {}
    client = _client(monkeypatch, created, FakeRun())
    r = client.post("/api/teacher/krm/run-analysis", params={"dir_code": "../etc"})
    assert r.status_code == 400, r.text
    assert not created, "запуск не должен создаваться при невалидном коде"
