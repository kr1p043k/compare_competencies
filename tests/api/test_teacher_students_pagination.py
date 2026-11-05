"""Pagination of /teacher/students (fake pool, auth overridden - no real DB)."""
from fastapi.testclient import TestClient

from src.api_pkg import create_app


class FakePool:
    """Reacts to the two queries teacher_students issues: COUNT + page fetch."""

    def __init__(self, rows=None, total=0):
        self.rows = rows or []
        self.total = total
        self.count_sql = None
        self.fetch_sql = None
        self.fetch_args = None

    async def fetchval(self, sql, *args):
        self.count_sql = sql
        return self.total

    async def fetch(self, sql, *args):
        self.fetch_sql = sql
        self.fetch_args = args
        return self.rows


def _client(monkeypatch, pool):
    from tests.conftest import open_all_gates

    import src.api_pkg.routers.teacher as t

    app = create_app()
    open_all_gates(app, role="teacher")
    monkeypatch.setattr(t, "get_pool", lambda: pool)
    return TestClient(app)


def _row(email="s1@test.local", full_name="S One"):
    return {"email": email, "full_name": full_name}


def test_default_limit_100(monkeypatch):
    pool = FakePool(rows=[_row()], total=1)
    r = _client(monkeypatch, pool).get("/api/teacher/students")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["limit"] == 100
    assert body["offset"] == 0
    assert pool.fetch_args == (100, 0)
    assert "ORDER BY email, id LIMIT $1 OFFSET $2" in pool.fetch_sql


def test_explicit_limit_offset(monkeypatch):
    pool = FakePool(rows=[_row()], total=1)
    r = _client(monkeypatch, pool).get("/api/teacher/students?limit=10&offset=20")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["limit"] == 10
    assert body["offset"] == 20
    assert pool.fetch_args == (10, 20)


def test_limit_over_500_is_422(monkeypatch):
    pool = FakePool()
    r = _client(monkeypatch, pool).get("/api/teacher/students?limit=501")
    assert r.status_code == 422, r.text


def test_negative_offset_is_422(monkeypatch):
    pool = FakePool()
    r = _client(monkeypatch, pool).get("/api/teacher/students?offset=-1")
    assert r.status_code == 422, r.text


def test_total_reflects_full_count(monkeypatch):
    rows = [_row("a@test.local"), _row("b@test.local")]
    pool = FakePool(rows=rows, total=55)
    r = _client(monkeypatch, pool).get("/api/teacher/students?limit=2")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 55
    assert len(body["students"]) == 2
    assert "COUNT(*)" in pool.count_sql