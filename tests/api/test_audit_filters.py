"""Фильтры и барьеры GET /admin/audit (durable-журнал аудита).

БД не нужна: сессия подменяется фейком, который запоминает текст SQL.
Проверяем построение фильтров, валидацию пагинации, устойчивость к
отсутствующей таблице и разбор since/until.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

import src.api_pkg.routers.audit as audit_mod
from tests.conftest import open_all_gates


class _UndefinedTableError(Exception):
    pgcode = "42P01"


class _FakeSession:
    def __init__(self, *, error: BaseException | None = None, scalar: int = 0, rows=None):
        self.queries: list[str] = []
        self.error = error
        self._scalar = scalar
        self._rows = list(rows or [])

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    @staticmethod
    def _compile(stmt) -> str:
        try:
            return str(stmt.compile(compile_kwargs={"literal_binds": True}))
        except Exception:
            return str(stmt)

    async def scalar(self, stmt):
        self.queries.append(self._compile(stmt))
        if self.error is not None:
            raise self.error
        return self._scalar

    async def execute(self, stmt):
        self.queries.append(self._compile(stmt))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self._rows))


def _app(*, gated: bool = False) -> FastAPI:
    app = FastAPI()
    app.state.limiter = audit_mod.limiter
    app.include_router(audit_mod.router, prefix="/api")
    return open_all_gates(app) if gated else app


def _patch_session(monkeypatch, session: _FakeSession) -> None:
    monkeypatch.setattr(audit_mod, "async_session_factory", lambda: session)
    with contextlib.suppress(Exception):
        audit_mod.limiter._storage.reset()


def test_audit_requires_admin(monkeypatch):
    """Без гейта (аноним) журнал аудита недоступен."""
    client = TestClient(_app(), raise_server_exceptions=False)
    assert client.get("/api/admin/audit").status_code == 401
    assert client.get("/api/admin/audit/actions").status_code == 401


def test_audit_builds_all_filters(monkeypatch):
    session = _FakeSession(scalar=3, rows=[])
    _patch_session(monkeypatch, session)
    client = TestClient(_app(gated=True), raise_server_exceptions=False)

    r = client.get(
        "/api/admin/audit",
        params={
            "action": "admin.user",
            "actor": "a@b.c",
            "target_type": "user",
            "target_id": "zzz",
            "since": "2026-01-01T00:00:00+00:00",
            "until": "2026-02-01T00:00:00Z",
            "limit": 50,
            "offset": 10,
        },
    )
    assert r.status_code == 200, r.text
    sql = "\n".join(session.queries)
    assert "audit_logs.action LIKE" in sql
    # SQLAlchemy рендерит префикс как 'admin.user' || '%' ESCAPE '/'.
    assert "'admin.user' || '%'" in sql
    assert "audit_logs.actor_email =" in sql
    assert "a@b.c" in sql
    assert "audit_logs.target_type =" in sql
    assert "audit_logs.target_id =" in sql
    assert "audit_logs.created_at >=" in sql
    assert "audit_logs.created_at <=" in sql
    assert "LIMIT 50" in sql
    assert "OFFSET 10" in sql
    body = r.json()
    assert body["total"] == 3
    assert body["limit"] == 50 and body["offset"] == 10


@pytest.mark.parametrize("qs", ["limit=0", "limit=501", "offset=-1", "offset=100001"])
def test_audit_rejects_bad_pagination(monkeypatch, qs):
    _patch_session(monkeypatch, _FakeSession())
    client = TestClient(_app(gated=True), raise_server_exceptions=False)
    assert client.get(f"/api/admin/audit?{qs}").status_code == 422


def test_audit_tolerates_missing_table(monkeypatch):
    error = SQLAlchemyError("relation does not exist")
    error.orig = _UndefinedTableError()  # type: ignore[attr-defined]
    _patch_session(monkeypatch, _FakeSession(error=error))
    client = TestClient(_app(gated=True), raise_server_exceptions=False)

    r = client.get("/api/admin/audit")
    assert r.status_code == 200, r.text
    assert r.json() == {"entries": [], "total": 0, "limit": 100, "offset": 0}


def test_audit_actions_lists_distinct(monkeypatch):
    session = _FakeSession(rows=["admin.user.create", "auth.2fa.login"])
    _patch_session(monkeypatch, session)
    client = TestClient(_app(gated=True), raise_server_exceptions=False)

    r = client.get("/api/admin/audit/actions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["actions"] == ["admin.user.create", "auth.2fa.login"]
    assert body["total"] == 2
    assert any("DISTINCT" in q.upper() for q in session.queries)


def test_parse_iso_naive_and_z_and_invalid():
    naive = audit_mod._parse_iso("2026-03-01T10:00:00", "since")
    assert naive.tzinfo is not None and naive.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    zulu = audit_mod._parse_iso("2026-03-01T10:00:00Z", "until")
    assert zulu.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    with pytest.raises(HTTPException) as exc:
        audit_mod._parse_iso("not-a-date", "since")
    assert exc.value.status_code == 400
    assert "since" in str(exc.value.detail)
