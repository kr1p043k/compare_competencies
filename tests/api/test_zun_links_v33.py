"""v33: skill-link + competency-search endpoints (fake pool, auth overridden)."""
import uuid

import pytest
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from src.api_pkg.routers import zun as zun_module


class FakePool:
    def __init__(self, script):
        self.script = script
        self.exec_log: list[str] = []

    async def fetchrow(self, sql, *a):
        self.exec_log.append(sql)
        for key, val in self.script.get("fetchrow", []):
            if key in sql:
                return val() if callable(val) else val
        return None

    async def fetchval(self, sql, *a):
        self.exec_log.append(sql)
        for key, val in self.script.get("fetchval", []):
            if key in sql:
                return val() if callable(val) else val
        return None

    async def fetch(self, sql, *a):
        self.exec_log.append(sql)
        for key, val in self.script.get("fetch", []):
            if key in sql:
                return val
        return []

    async def execute(self, sql, *a):
        self.exec_log.append(sql)
        return "OK"


def _client(pool):
    app = create_app()
    for route in app.routes:
        dep = getattr(route, "dependant", None)
        for sub in (getattr(dep, "dependencies", None) or []):
            app.dependency_overrides[sub.call] = lambda: {"r": "admin"}
    import src.api_pkg.routers.zun as z
    z.get_pool = lambda: pool
    return TestClient(app)


CID = str(uuid.uuid4())
SID = str(uuid.uuid4())


def _pool_ok(skill_row=None, dup=None):
    return FakePool({
        "fetchrow": [
            ("FROM competencies WHERE id", {"id": CID}),
            ("FROM skills WHERE LOWER(name)", skill_row),
            ("FROM competency_skills cs", dup),
        ],
        "fetchval": [("INSERT INTO skills", SID)],
    })


def test_link_creates_skill_and_link():
    pool = _pool_ok()
    r = _client(pool).post(f"/api/teacher/zun/competencies/{CID}/skills",
                           json={"skill_name": "docker", "ksa_type": "skills"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["match_type"] == "exact" and body["skill_name"] == "docker"
    assert any("INSERT INTO skills" in s for s in pool.exec_log)
    assert any("INSERT INTO competency_skills" in s for s in pool.exec_log)
    assert any("UPDATE competencies SET updated_at" in s for s in pool.exec_log)


def test_link_reuses_rpd_skill_row():
    pool = _pool_ok(skill_row={"id": SID, "source": "rpd_skills"})
    r = _client(pool).post(f"/api/teacher/zun/competencies/{CID}/skills",
                           json={"skill_name": "docker"})
    assert r.status_code == 201, r.text
    assert r.json()["skill_id"] == SID
    assert not any("INSERT INTO skills" in s for s in pool.exec_log)


def test_link_market_row_forks_rpd_row():
    pool = _pool_ok(skill_row={"id": "market-row", "source": "market"})
    r = _client(pool).post(f"/api/teacher/zun/competencies/{CID}/skills",
                           json={"skill_name": "docker"})
    assert r.status_code == 201, r.text
    assert r.json()["skill_id"] == SID  # new rpd_skills row, market row untouched
    assert any("INSERT INTO skills" in s for s in pool.exec_log)


def test_link_conflicts_and_validations():
    pool = _pool_ok(dup={"1": 1})
    c = _client(pool)
    assert c.post(f"/api/teacher/zun/competencies/{CID}/skills",
                  json={"skill_name": "docker"}).status_code == 409
    assert c.post(f"/api/teacher/zun/competencies/{CID}/skills",
                  json={"skill_name": "x", "ksa_type": "nope"}).status_code == 400
    assert c.post(f"/api/teacher/zun/competencies/{CID}/skills",
                  json={"skill_name": "  "}).status_code == 400
    bad = str(uuid.uuid4())
    pool2 = _pool_ok()
    c2 = _client(pool2)
    # unknown competency -> 404 (competency lookup returns None)
    pool2.script["fetchrow"] = [(k, v) for k, v in pool2.script["fetchrow"]
                                if "FROM competencies WHERE id" not in k]
    assert c2.post(f"/api/teacher/zun/competencies/{bad}/skills",
                   json={"skill_name": "docker"}).status_code == 404


def test_competency_search_shape():
    rows = [{"id": CID, "code": "PK-1", "discipline_name": " networks"}]
    pool = FakePool({"fetch": [("FROM competencies c", rows)]})
    r = _client(pool).get("/api/teacher/zun/competencies/search?q=pk&dir_code=09.03.02")
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert items[0]["code"] == "PK-1" and "id" in items[0]
