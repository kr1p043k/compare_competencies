"""Pagination for GET /api/admin/users (LIMIT/OFFSET + total)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api_pkg.routers.admin as admin_mod
import src.api_pkg.routers.auth as auth_mod
import src.database as database_mod


def _user(i: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"00000000-0000-0000-0000-{i:012d}",
        email=f"user{i}@example.com",
        role="student",
        full_name=f"User {i}",
        is_active=True,
    )


class _FakeResult:
    def __init__(self, rows: list, *, scalar: object = None):
        self._rows = rows
        self._scalar = scalar

    def fetchall(self) -> list:
        return self._rows

    def all(self) -> list:
        return self._rows

    def scalar_one(self):
        return self._scalar


class _FakeSession:
    def __init__(self, users: list, directions: list[tuple[str, str]]):
        self._users = users
        self._directions = directions
        self.queries: list[str] = []

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, stmt, params: dict | None = None):
        sql = str(stmt)
        self.queries.append(sql)
        if "COUNT(*)" in sql:
            return _FakeResult([], scalar=len(self._users))
        if "FROM users" in sql and "LIMIT" in sql:
            limit = int(params["limit"])
            offset = int(params["offset"])
            return _FakeResult(self._users[offset:offset + limit])
        if "user_directions" in sql:
            return _FakeResult(self._directions)
        return _FakeResult([])


@pytest.fixture(autouse=True)
def _reset_limiter():
    with contextlib.suppress(Exception):
        admin_mod.limiter._storage.reset()
    yield


async def _fake_current_user(_request):
    return {"u": "admin@example.com", "r": "admin"}


def _client(monkeypatch, users: list, directions: list[tuple[str, str]] | None = None) -> TestClient:
    monkeypatch.setattr(auth_mod, "get_current_user", _fake_current_user)
    monkeypatch.setattr(database_mod, "async_session_factory", lambda: _FakeSession(users, directions or []))
    monkeypatch.setattr(admin_mod, "get_logs_by_user", lambda: {})

    app = FastAPI()
    app.state.limiter = admin_mod.limiter
    app.include_router(admin_mod.router)
    return TestClient(app, raise_server_exceptions=False)


def test_default_limit_100(monkeypatch):
    client = _client(monkeypatch, [_user(i) for i in range(3)])
    r = client.get("/admin/users")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["limit"] == 100
    assert body["offset"] == 0
    assert body["total"] == 3
    assert len(body["users"]) == 3
    assert body["users"][0]["username"] == "user0@example.com"


def test_explicit_limit_offset(monkeypatch):
    client = _client(monkeypatch, [_user(i) for i in range(5)])
    r = client.get("/admin/users?limit=2&offset=1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert body["total"] == 5
    assert [u["username"] for u in body["users"]] == ["user1@example.com", "user2@example.com"]


@pytest.mark.parametrize("qs", ["limit=501", "limit=0", "offset=-1"])
def test_invalid_pagination_returns_422(monkeypatch, qs):
    client = _client(monkeypatch, [_user(0)])
    r = client.get(f"/admin/users?{qs}")
    assert r.status_code == 422, r.text


def test_total_is_full_count_ignoring_limit(monkeypatch):
    client = _client(monkeypatch, [_user(i) for i in range(7)])
    r = client.get("/admin/users?limit=2&offset=0")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 7
    assert len(body["users"]) == 2