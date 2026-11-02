"""IDOR guard on mutating /teacher/zun endpoints.

A teacher who owns direction 09.03.02 must not be able to mutate rows that belong
to another direction (01.03.01 here). Admin bypasses the direction check.

Unit level: a fake pool resolves the direction of the target row, and
_user_directions is patched so the test never touches real KRM files.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from src.api_pkg.routers import krm_teacher

OTHER_DIR = "01.03.01"
OWN_DIR = "09.03.02"


class FakePool:
    """Minimal asyncpg stand-in: resolves the target row's direction."""

    def __init__(self, dir_code=OTHER_DIR):
        self.dir_code = dir_code
        self.writes = []

    async def fetchrow(self, sql, *a):
        if "d.code AS dir_code" in sql:
            return {"id": "row-1", "dir_code": self.dir_code}
        if "FROM discipline_scope" in sql:
            return None
        if "SELECT 1 FROM" in sql:
            return None
        return None

    async def fetch(self, sql, *a):
        if "FROM disciplines d2" in sql:
            return [{"name": "Math"}]
        if "FROM user_directions" in sql:
            return [{"dir_code": OWN_DIR}]
        return []

    async def fetchval(self, sql, *a):
        if "COALESCE(MAX" in sql:
            return 1
        if "INSERT INTO" in sql:
            return uuid.uuid4()
        return None

    async def execute(self, sql, *a):
        self.writes.append(sql)
        return "OK"


def _make_client(pool, role="teacher"):
    app = create_app()
    for route in app.routes:
        dep = getattr(route, "dependant", None)
        for sub in (getattr(dep, "dependencies", None) or []):
            app.dependency_overrides[sub.call] = lambda: {"r": role}

    import src.api_pkg.routers.auth as auth
    import src.api_pkg.routers.zun as z

    async def fake_user(_request):
        return {"uid": "u-1", "u": "t@example.local", "r": role}

    async def fake_audit(*_a, **_kw):
        return None

    async def fake_dirs(user):
        if user.get("r") == "admin":
            return [OWN_DIR, OTHER_DIR]
        return [OWN_DIR]

    z.get_pool = lambda: pool
    z.get_current_user = fake_user
    z.audit_action = fake_audit
    krm_teacher.get_pool = lambda: pool
    krm_teacher._user_directions = fake_dirs
    # zun imported the guard by value: rebind it to use the patched helpers.
    z._ensure_dir_access = krm_teacher._ensure_dir_access
    assert auth  # imported for symmetry, keeps ruff quiet
    return TestClient(app)


ROW = "11111111-1111-1111-1111-111111111111"


@pytest.mark.parametrize("method,path,payload", [
    ("post", f"/api/teacher/zun/competencies/{ROW}/entries",
     {"ksa_type": "knowledge", "text": "x"}),
    ("post", f"/api/teacher/zun/disciplines/{ROW}/competencies",
     {"code": "ПК-1", "name": "n"}),
    ("post", f"/api/teacher/zun/competencies/{ROW}/skills",
     {"skill_name": "Python", "ksa_type": "skills"}),
    ("patch", f"/api/teacher/zun/entries/{ROW}", {"text": "x"}),
    ("delete", f"/api/teacher/zun/entries/{ROW}", None),
    ("put", "/api/teacher/zun/scope",
     {"dir_code": OTHER_DIR, "changes": [{"discipline_name": "Math", "included": False}]}),
    ("post", f"/api/teacher/zun/import/{OTHER_DIR}", None),
])
def test_teacher_cannot_write_foreign_direction(method, path, payload):
    client = _make_client(FakePool(dir_code=OTHER_DIR), role="teacher")
    fn = getattr(client, method)
    r = fn(path, json=payload) if payload is not None else fn(path)
    assert r.status_code == 403, f"{method} {path} -> {r.status_code} {r.text[:200]}"


def test_admin_passes_direction_guard():
    pool = FakePool(dir_code=OTHER_DIR)
    client = _make_client(pool, role="admin")
    r = client.post(f"/api/teacher/zun/competencies/{ROW}/entries",
                    json={"ksa_type": "knowledge", "text": "admin write"})
    assert r.status_code == 201, r.text[:300]


def test_teacher_can_write_own_direction():
    pool = FakePool(dir_code=OWN_DIR)
    client = _make_client(pool, role="teacher")
    r = client.post(f"/api/teacher/zun/competencies/{ROW}/entries",
                    json={"ksa_type": "knowledge", "text": "own write"})
    assert r.status_code == 201, r.text[:300]
