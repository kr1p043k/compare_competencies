"""P0 security gates: admin-only user creation, authed export."""
from fastapi.testclient import TestClient

from src.api_pkg import create_app


def _route_deps(path: str, method: str = "GET"):
    for r in create_app().routes:
        if getattr(r, "path", "") == path and method in getattr(r, "methods", set()):
            return list(getattr(r, "dependencies", []) or [])
    raise AssertionError(f"route {method} {path} not found")


def _dep_roles(dep) -> tuple:
    fn = getattr(dep, "dependency", None)
    closure = getattr(fn, "__closure__", None) or []
    for cell in closure:
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if isinstance(val, tuple) and all(isinstance(x, str) for x in val):
            return val
    raise AssertionError("roles not introspectable")


def test_users_create_admin_only():
    deps = _route_deps("/api/admin/users/create", "POST")
    assert deps, "users/create must carry auth dependency"
    roles = {_dep_roles(d) for d in deps}
    assert ("admin",) in roles, f"no admin-only gate, have {roles}"


def test_export_requires_auth():
    deps = _route_deps("/api/teacher/export/vacancies", "GET")
    assert deps, "export must carry auth dependency"


def test_no_token_unauthorized():
    client = TestClient(create_app(), raise_server_exceptions=False)
    r1 = client.post("/api/admin/users/create",
                     json={"email": "x@y.z", "password": "pw", "role": "teacher"})
    assert r1.status_code in (401, 422)
    r2 = client.get("/api/teacher/export/vacancies")
    assert r2.status_code == 401
