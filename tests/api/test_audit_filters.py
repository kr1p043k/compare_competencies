"""Фильтры и барьеры GET /admin/audit (durable-журнал аудита).

БД не нужна: сессия подменяется фейком, который запоминает текст SQL.
Проверяем построение фильтров, валидацию пагинации, устойчивость к
отсутствующей таблице и разбор since/until.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
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
        self.params: list[dict] = []
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

    def _record(self, stmt) -> None:
        self.queries.append(self._compile(stmt))
        try:
            self.params.append(dict(stmt.compile().params))
        except Exception:
            self.params.append({})

    async def scalar(self, stmt):
        self._record(stmt)
        if self.error is not None:
            raise self.error
        return self._scalar

    async def execute(self, stmt):
        self._record(stmt)
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


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("since", "not-a-date"),
        ("until", "not-a-date"),
        ("since", "2026-02-30T00:00:00"),
        ("until", "31.12.2026 00:00"),
    ],
)
def test_audit_endpoint_rejects_broken_datetimes_with_400(monkeypatch, param, value):
    """Битая since/until даёт 400 от самого эндпоинта (не 500 и не 422).

    Для since/until нет Query-констрейнтов сверх max_length=40, поэтому
    короткая мусорная строка доходит до _parse_iso и падает как 400.
    """
    _patch_session(monkeypatch, _FakeSession())
    client = TestClient(_app(gated=True), raise_server_exceptions=False)

    r = client.get("/api/admin/audit", params={param: value})
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"Invalid {param}: expected ISO 8601 datetime"


def test_audit_endpoint_treats_naive_datetimes_as_utc(monkeypatch):
    """Эндпоинт принимает даты без таймзоны: в SQL уходят aware-значения UTC.

    Проверяем сами bind-параметры скомпилированных запросов (count + select),
    а не только внутренний _parse_iso: naive since/until не должен дойти
    до DateTime(timezone=True)-сравнения без tzinfo.
    """
    session = _FakeSession(scalar=0)
    _patch_session(monkeypatch, session)
    client = TestClient(_app(gated=True), raise_server_exceptions=False)

    r = client.get(
        "/api/admin/audit",
        params={"since": "2026-03-01T10:00:00", "until": "2026-03-02T10:00:00"},
    )
    assert r.status_code == 200, r.text

    dts = [v for p in session.params for v in p.values() if isinstance(v, datetime)]
    assert dts, "datetime-фильтры должны попасть в параметры SQL"
    assert all(v.utcoffset() is not None for v in dts), f"naive datetime прошёл в SQL: {dts}"
    assert all(v.utcoffset().total_seconds() == 0 for v in dts)


@pytest.mark.parametrize(
    ("qs", "limit", "offset"),
    [
        ("limit=500&offset=0", 500, 0),
        ("limit=1&offset=100000", 1, 100000),
        ("", 100, 0),  # значения по умолчанию - тоже внутри границ
    ],
)
def test_audit_accepts_pagination_at_bounds(monkeypatch, qs, limit, offset):
    """Зажим limit/offset включителен: границы (500, 100000) и дефолты проходят."""
    session = _FakeSession(scalar=0)
    _patch_session(monkeypatch, session)
    client = TestClient(_app(gated=True), raise_server_exceptions=False)

    r = client.get(f"/api/admin/audit{('?' + qs) if qs else ''}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["limit"] == limit
    assert body["offset"] == offset
    sql = "\n".join(session.queries)
    assert f"LIMIT {limit}" in sql
    assert f"OFFSET {offset}" in sql


def test_audit_actions_tolerates_missing_table(monkeypatch):
    """Без таблицы audit_logs /admin/audit/actions отдаёт пустой список, а не 500."""
    error = SQLAlchemyError("relation does not exist")
    error.orig = _UndefinedTableError()  # type: ignore[attr-defined]
    _patch_session(monkeypatch, _FakeSession(error=error))
    client = TestClient(_app(gated=True), raise_server_exceptions=False)

    r = client.get("/api/admin/audit/actions")
    assert r.status_code == 200, r.text
    assert r.json() == {"actions": [], "total": 0}
