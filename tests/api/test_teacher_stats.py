"""GET /api/teacher/stats: aggregate discipline/competency/skill counts (no live DB).

Covers `src/api_pkg/routers/teacher.teacher_stats`. The endpoint contract:
two DB reads (one aggregate fetchrow + one vacancies fetchval), both
swallowed on failure, so the answer always carries the four documented keys.

- happy path: fake pool returns DB values, response shape/keys asserted and
  both documented queries verified against the pool;
- empty DB: fetchrow -> None and fetchval -> None still give the documented
  zero defaults, not a 500;
- pool unavailable: get_pool raising is swallowed (the endpoint degrades to
  its zero-default contract);
- role guard: the admin/teacher/rop gate stays real - anonymous -> 401,
  student -> 403 (only get_current_user is stubbed).

Auth/TestClient conventions follow tests/api/test_teacher_history.py, the
fake-pool convention follows tests/api/test_teacher_students_pagination.py.
The pool fake is installed on the router module itself because teacher.py
binds get_pool at import time (`from src.db import get_pool`).
"""

from __future__ import annotations

import contextlib

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.teacher as teacher_mod
from src.api_pkg import create_app
from tests.conftest import open_all_gates, route_gate_roles

ENDPOINT = "/api/teacher/stats"
ZERO_DEFAULTS = {"disciplines": 0, "competencies": 0, "skills": 0, "vacancies": 0}


class FakePool:
    """asyncpg-like pool: answers the two queries teacher_stats issues."""

    def __init__(self, row=None, vacancies=None):
        self.row = row
        self.vacancies = vacancies
        self.fetchrow_sql = None
        self.fetchval_sql = None

    async def fetchrow(self, sql, *args):
        self.fetchrow_sql = sql
        return self.row

    async def fetchval(self, sql, *args):
        self.fetchval_sql = sql
        return self.vacancies


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Shared module-level limiter: keep this file self-contained (tests/api convention)."""
    with contextlib.suppress(Exception):
        teacher_mod.limiter._storage.reset()
    yield


def _client(monkeypatch, pool):
    """Gates stubbed via open_all_gates (role=teacher): endpoint logic only."""
    app = open_all_gates(create_app(), role="teacher")
    monkeypatch.setattr(teacher_mod, "get_pool", lambda: pool)
    return TestClient(app, raise_server_exceptions=False)


def test_happy_path_returns_db_counts(monkeypatch):
    pool = FakePool(
        row={"disciplines": 12, "competencies": 340, "skills": 1024},
        vacancies=5533,
    )
    r = _client(monkeypatch, pool).get(ENDPOINT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"disciplines", "competencies", "skills", "vacancies"}
    assert body == ZERO_DEFAULTS | {
        "disciplines": 12,
        "competencies": 340,
        "skills": 1024,
        "vacancies": 5533,
    }
    # Both documented queries really reached the pool.
    assert "COUNT(DISTINCT disc.id) AS disciplines" in pool.fetchrow_sql
    assert "COUNT(DISTINCT c.id) AS competencies" in pool.fetchrow_sql
    assert "COUNT(DISTINCT k.id) AS skills" in pool.fetchrow_sql
    assert "SELECT COUNT(*) FROM vacancies" in pool.fetchval_sql


def test_empty_db_returns_zero_defaults(monkeypatch):
    pool = FakePool(row=None, vacancies=None)
    r = _client(monkeypatch, pool).get(ENDPOINT)
    assert r.status_code == 200, r.text
    assert r.json() == ZERO_DEFAULTS


def test_pool_unavailable_degrades_to_zero_defaults(monkeypatch):
    """A dead pool must not turn /teacher/stats into a 500: zeros are the contract."""

    def _no_pool():
        raise RuntimeError("Database pool not initialized")

    app = open_all_gates(create_app(), role="teacher")
    monkeypatch.setattr(teacher_mod, "get_pool", _no_pool)
    r = TestClient(app, raise_server_exceptions=False).get(ENDPOINT)
    assert r.status_code == 200, r.text
    assert r.json() == ZERO_DEFAULTS


def test_route_declares_teacher_role_gate():
    roles = route_gate_roles(create_app(), ENDPOINT, "GET")
    assert ("admin", "teacher", "rop") in roles, f"actual gates: {roles}"


def test_anonymous_request_gets_401():
    # No open_all_gates here: the real gate must reject a tokenless request
    # before the handler touches the pool.
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(ENDPOINT)
    assert r.status_code == 401, r.text


def test_student_role_gets_403(monkeypatch):
    # Same trick as test_admin_guards: substitute get_current_user only, so
    # the REAL require_any_role checks the role and answers 403 itself.
    async def _student(_request):
        return {"uid": "s-1", "u": "student@example.com", "r": "student", "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _student)
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(ENDPOINT)
    assert r.status_code == 403, r.text
