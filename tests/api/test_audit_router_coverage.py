"""Покрытие роутера audit: список записей и словарь действий.

Сессию SQLAlchemy подменяем фейком: реальная БД не нужна. Проверяем фильтры,
пагинацию, разбор дат и мягкую обработку отсутствующей таблицы.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import ProgrammingError

import src.api_pkg.routers.audit as audit


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class FakeSession:
    def __init__(self, *, total=0, rows=None, actions=None, raise_exc=None):
        self._total = total
        self._rows = rows or []
        self._actions = actions or []
        self._raise = raise_exc
        self.queries: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def scalar(self, _stmt):
        if self._raise:
            raise self._raise
        return self._total

    async def execute(self, stmt):
        if self._raise:
            raise self._raise
        self.queries.append(stmt)
        # Различаем выборку записей и словарь действий по числу колонок.
        if self._actions:
            return FakeResult(self._actions)
        return FakeResult(self._rows)


def _row():
    return SimpleNamespace(
        id=uuid.uuid4(),
        actor_email="a@b.c",
        actor_role="admin",
        action="auth.login",
        target_type="user",
        target_id="1",
        target=None,
        meta={"k": "v"},
        ip="127.0.0.1",
        request_id="r1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _client(monkeypatch, session, *, role="admin"):
    from src.api_pkg import create_app
    from tests.conftest import open_all_gates

    app = create_app()
    open_all_gates(app, role=role)
    monkeypatch.setattr(audit, "async_session_factory", lambda: session)
    lim = getattr(audit, "limiter", None)
    if lim is not None:
        lim.reset()
    return TestClient(app)


# ---------- чистые хелперы ----------


def test_parse_iso_naive_gets_utc():
    dt = audit._parse_iso("2026-01-01T10:00:00", "since")
    assert dt.tzinfo is UTC


def test_parse_iso_with_z():
    dt = audit._parse_iso("2026-01-01T10:00:00Z", "since")
    assert dt.year == 2026 and dt.tzinfo is not None


def test_parse_iso_bad_raises_400():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        audit._parse_iso("not-a-date", "since")
    assert e.value.status_code == 400


def test_is_undefined_table_by_sqlstate():
    exc = ProgrammingError("stmt", {}, Exception())
    exc.orig = SimpleNamespace(sqlstate="42P01")
    assert audit._is_undefined_table(exc) is True


def test_is_undefined_table_false_for_other():
    assert audit._is_undefined_table(RuntimeError("nope")) is False


def test_serialize_audit():
    out = audit._serialize_audit(_row())
    assert out["actor_email"] == "a@b.c"
    assert out["created_at"] == "2026-01-01T00:00:00+00:00"


# ---------- /admin/audit ----------


def test_list_audit_returns_entries(monkeypatch):
    session = FakeSession(total=1, rows=[_row()])
    c = _client(monkeypatch, session)
    r = c.get("/api/admin/audit")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    assert body["entries"][0]["action"] == "auth.login"


def test_list_audit_with_filters(monkeypatch):
    session = FakeSession(total=0, rows=[])
    c = _client(monkeypatch, session)
    r = c.get(
        "/api/admin/audit?action=auth&actor=a@b.c&target_type=user"
        "&target_id=1&since=2026-01-01&until=2026-02-01"
    )
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_list_audit_bad_date_400(monkeypatch):
    c = _client(monkeypatch, FakeSession())
    assert c.get("/api/admin/audit?since=nope").status_code == 400


def test_list_audit_missing_table_returns_empty(monkeypatch):
    exc = ProgrammingError("stmt", {}, Exception())
    exc.orig = SimpleNamespace(sqlstate="42P01")
    c = _client(monkeypatch, FakeSession(raise_exc=exc))
    r = c.get("/api/admin/audit")
    assert r.status_code == 200
    assert r.json() == {"entries": [], "total": 0, "limit": 100, "offset": 0}


def test_list_audit_limit_validation(monkeypatch):
    c = _client(monkeypatch, FakeSession())
    assert c.get("/api/admin/audit?limit=0").status_code == 422
    assert c.get("/api/admin/audit?limit=501").status_code == 422


# ---------- /admin/audit/actions ----------


def test_list_actions(monkeypatch):
    c = _client(monkeypatch, FakeSession(actions=["auth.login", "auth.logout"]))
    r = c.get("/api/admin/audit/actions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2
    assert "auth.login" in body["actions"]


def test_list_actions_missing_table(monkeypatch):
    exc = ProgrammingError("stmt", {}, Exception())
    exc.orig = SimpleNamespace(sqlstate="42P01")
    c = _client(monkeypatch, FakeSession(raise_exc=exc))
    r = c.get("/api/admin/audit/actions")
    assert r.status_code == 200
    assert r.json() == {"actions": [], "total": 0}
