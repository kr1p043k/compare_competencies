"""IDOR-guard в RPD: запуск/статус/отмена чужого направления или чужого запуска.

Проверяет guard'ы напрямую, как test_zun_idor.py:
  * upload/collect чужого направления -> 403 (guard на входе в эндпоинт);
  * status/cancel legacy-запуска (без направления и владельца) -> 403 у teacher,
    у admin проходит (fail closed только для не-админа);
  * status/cancel запуска своего направления teacher'ом -> не 403;
  * запуск без направления, но с чужим владельцем -> 403; со своим -> не 403.

Подменяются текущий пользователь, список доступных направлений и фабрика
сессий БД. Реальные подпроцессы/БД не запускаются.
"""
import uuid

from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.zun as zun_mod
import src.database as database_mod
from src.api_pkg import create_app

OWN = "09.03.02"
FOREIGN = "01.03.02"  # есть в YANDEX_COVERED, поэтому collect идёт до 400-coverage
RUN_ID = str(uuid.uuid4())
TEACHER_UID = "u-1"


class FakeRun:
    def __init__(self, direction_code=None, owner_id=None, status="started"):
        self.id = RUN_ID
        self.action = "rpd-import"
        self.status = status
        self.direction_code = direction_code
        self.owner_id = owner_id
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.stats = None


class FakeSession:
    """Асинхронная сессия, возвращающая заданный run по любому id."""

    def __init__(self, run: FakeRun | None):
        self._run = run

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, _model, _run_id):
        return self._run


def _make_client(monkeypatch, *, run=None, role="teacher", dirs=(OWN,)):
    app = create_app()

    async def _fake_user(_request):
        return {"uid": TEACHER_UID, "u": f"{role}@example.com", "r": role, "t": None}

    async def _fake_dirs(_user):
        return list(dirs)

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    monkeypatch.setattr(zun_mod, "_user_dir_codes", _fake_dirs)
    if run is not None:
        def _fake_session_factory():
            return FakeSession(run)
        monkeypatch.setattr(database_mod, "async_session_factory", _fake_session_factory)
    return TestClient(app)


# ------------------------------------------------------------ create-run guard

def test_upload_foreign_direction_403(monkeypatch):
    client = _make_client(monkeypatch)
    r = client.post(
        "/api/teacher/rpd/upload",
        data={"dir_code": FOREIGN},
        files={"file": ("a.pdf", b"%PDF-1.4 test", "application/pdf")},
    )
    assert r.status_code == 403, r.text
    assert "запрещён" in r.json()["detail"]


def test_collect_foreign_direction_403(monkeypatch):
    client = _make_client(monkeypatch)
    r = client.post("/api/teacher/rpd/collect", data={"dir_code": FOREIGN})
    assert r.status_code == 403, r.text
    assert "запрещён" in r.json()["detail"]


# ------------------------------------------------------------ legacy run access

def test_status_legacy_teacher_403(monkeypatch):
    client = _make_client(monkeypatch, run=FakeRun())
    r = client.get(f"/api/teacher/rpd/status/{RUN_ID}")
    assert r.status_code == 403, r.text


def test_cancel_legacy_teacher_403(monkeypatch):
    client = _make_client(monkeypatch, run=FakeRun(status="started"))
    r = client.post(f"/api/teacher/rpd/cancel/{RUN_ID}")
    assert r.status_code == 403, r.text


def test_status_legacy_admin_not_403(monkeypatch):
    client = _make_client(monkeypatch, run=FakeRun(), role="admin")
    r = client.get(f"/api/teacher/rpd/status/{RUN_ID}")
    assert r.status_code == 200, r.text


def test_cancel_legacy_admin_not_403(monkeypatch):
    # finished run -> отдача статуса без записи в БД (не 403).
    client = _make_client(monkeypatch, run=FakeRun(status="completed"), role="admin")
    r = client.post(f"/api/teacher/rpd/cancel/{RUN_ID}")
    assert r.status_code == 200, r.text


# ------------------------------------------------------------ own-direction run

def test_status_own_direction_teacher_not_403(monkeypatch):
    client = _make_client(monkeypatch, run=FakeRun(direction_code=OWN))
    r = client.get(f"/api/teacher/rpd/status/{RUN_ID}")
    assert r.status_code == 200, r.text


def test_cancel_own_direction_teacher_not_403(monkeypatch):
    client = _make_client(monkeypatch, run=FakeRun(direction_code=OWN, status="completed"))
    r = client.post(f"/api/teacher/rpd/cancel/{RUN_ID}")
    assert r.status_code == 200, r.text


# ------------------------------------------------------------ owner-only runs

def test_status_owned_by_other_teacher_403(monkeypatch):
    client = _make_client(monkeypatch, run=FakeRun(owner_id="u-other"))
    r = client.get(f"/api/teacher/rpd/status/{RUN_ID}")
    assert r.status_code == 403, r.text


def test_status_owned_by_self_not_403(monkeypatch):
    client = _make_client(monkeypatch, run=FakeRun(owner_id=TEACHER_UID))
    r = client.get(f"/api/teacher/rpd/status/{RUN_ID}")
    assert r.status_code == 200, r.text
