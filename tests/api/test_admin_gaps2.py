"""Admin monitoring gaps2: session activity window + KRM discipline detail."""
import os
import urllib.parse

import pytest

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)


def _req(path="/api/x"):
    from starlette.requests import Request
    scope = {"type": "http", "method": "GET", "path": path,
             "query_string": b"", "headers": []}
    return Request(scope)


class FakePool:
    def __init__(self):
        self.fetchval_queries = []

    async def fetchval(self, q, *a):
        self.fetchval_queries.append(q)
        if "last_activity" in q:
            return 1
        if "logged_out_at IS NULL" in q:
            return 42
        return 0

    async def fetch(self, q, *a):
        return []

    async def fetchrow(self, q, *a):
        return None


def _summary_fn():
    from src.api_pkg.routers import admin as adminmod
    return getattr(adminmod.admin_monitoring_summary, "__wrapped__",
                   adminmod.admin_monitoring_summary)


async def test_summary_sessions_use_activity_window(monkeypatch):
    import src.db as dbmod
    fake = FakePool()
    monkeypatch.setattr(dbmod, "pool", fake)
    out = await _summary_fn()(_req("/api/admin/monitoring/summary"))
    assert out["sessions_active"] == 1
    assert out["sessions_total_open"] == 42
    active_q, total_q = fake.fetchval_queries[0], fake.fetchval_queries[1]
    assert "last_activity" in active_q and "15 minutes" in active_q
    assert "logged_out_at IS NULL" in total_q
    assert "last_activity" not in total_q


def test_routes_registered():
    from src.api_pkg import create_app
    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/admin/monitoring/summary" in paths
    assert "/api/teacher/krm/disciplines/{discipline_name:path}" in paths


async def _connect():
    try:
        import asyncpg
    except ImportError:
        pytest.skip("asyncpg not installed")
    try:
        pool = await asyncpg.create_pool(
            TEST_DB_URL, min_size=1, max_size=2, command_timeout=30)
    except Exception:
        pytest.skip("local postgres unavailable")
    return pool


async def test_sessions_count_respects_activity_window():
    import src.db as dbmod
    pool = await _connect()
    old = dbmod.pool
    dbmod.pool = pool
    tag = "t-gap2-%d" % os.getpid()
    try:
        uid = await pool.fetchval("SELECT id FROM users LIMIT 1")
        if uid is None:
            pytest.skip("no seed users")
        active_q = ("SELECT COUNT(*) FROM sessions WHERE logged_out_at IS NULL "
                    "AND last_activity > NOW() - INTERVAL '15 minutes'")
        total_q = "SELECT COUNT(*) FROM sessions WHERE logged_out_at IS NULL"
        base_active = await pool.fetchval(active_q)
        base_total = await pool.fetchval(total_q)
        await pool.execute(
            "INSERT INTO sessions (user_id, token_hash, last_activity) VALUES "
            "($1, $2, NOW()), ($1, $3, NOW() - INTERVAL '2 hours')",
            uid, tag + "-fresh", tag + "-stale")
        out = await _summary_fn()(_req("/api/admin/monitoring/summary"))
        assert out["sessions_active"] == base_active + 1
        assert out["sessions_total_open"] == base_total + 2
    finally:
        await pool.execute("DELETE FROM sessions WHERE token_hash IN ($1, $2)",
                           tag + "-fresh", tag + "-stale")
        dbmod.pool = old
        await pool.close()


async def test_krm_discipline_detail_returns_200():
    import src.db as dbmod
    pool = await _connect()
    old = dbmod.pool
    dbmod.pool = pool
    try:
        row = await pool.fetchrow(
            "SELECT disc.name FROM disciplines disc "
            "JOIN directions d ON d.id = disc.direction_id "
            "WHERE d.code = $1 ORDER BY disc.name LIMIT 1", "09.03.02")
        if not row:
            pytest.skip("no seed disciplines")
        from httpx import AsyncClient, ASGITransport
        from src.api_pkg import create_app
        from tests.conftest import open_all_gates
        app = open_all_gates(create_app())
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://t") as c:
            r = await c.get(
                "/api/teacher/krm/disciplines/" + urllib.parse.quote(row["name"]),
                params={"dir_code": "09.03.02"})
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert body["name"] == row["name"]
        assert isinstance(body.get("competencies"), list) and body["competencies"]
        c0 = body["competencies"][0]
        assert {"id", "code", "skills", "ksa"} <= set(c0)
        assert {"knowledge", "abilities", "skills"} <= set(c0["ksa"])
    finally:
        dbmod.pool = old
        await pool.close()


async def test_krm_coverage_no_tz_crash():
    """GET /api/teacher/krm/coverage must not 500 on tz-aware analysis_date."""
    import src.db as dbmod
    pool = await _connect()
    old = dbmod.pool
    dbmod.pool = pool
    try:
        from httpx import AsyncClient, ASGITransport
        from src.api_pkg import create_app
        from tests.conftest import open_all_gates
        app = open_all_gates(create_app())
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://t") as c:
            r = await c.get("/api/teacher/krm/coverage")
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert isinstance(body.get("disciplines"), list)
    finally:
        dbmod.pool = old
        await pool.close()


async def test_krm_tree_no_tz_crash():
    """GET /api/teacher/krm/competencies/tree must not 500 (tz strip)."""
    import src.db as dbmod
    pool = await _connect()
    old = dbmod.pool
    dbmod.pool = pool
    try:
        from httpx import AsyncClient, ASGITransport
        from src.api_pkg import create_app
        from tests.conftest import open_all_gates
        app = open_all_gates(create_app())
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://t") as c:
            r = await c.get("/api/teacher/krm/competencies/tree",
                            params={"dir_code": "09.03.02"})
        assert r.status_code == 200, r.text[:300]
    finally:
        dbmod.pool = old
        await pool.close()
