"""P0 security gates: admin-only user creation, authed export, no default creds."""
import json
from pathlib import Path

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


def test_seed_refuses_defaults():
    from seed_users import _resolve_password, BLOCKED_PASSWORDS
    assert "admin" in BLOCKED_PASSWORDS and "teacher123" in BLOCKED_PASSWORDS
    assert _resolve_password("a@b.c", {"password": "admin"}) is None
    assert _resolve_password("a@b.c", {"password": "  Teacher123 "}) is None
    assert _resolve_password("a@b.c", {"password": "s3cure!X9q"}) == "s3cure!X9q"


def test_seed_env_override(monkeypatch):
    from seed_users import _resolve_password
    monkeypatch.setenv("SEED_ADMIN_PASSWORD", "env-only-pw")
    assert _resolve_password("a@b.c", {"password_env": "SEED_ADMIN_PASSWORD",
                                       "password": "admin"}) == "env-only-pw"


def test_example_has_no_inline_passwords():
    tpl = json.loads((Path(__file__).resolve().parents[2]
                      / "users.example.json").read_text(encoding="utf-8"))
    assert tpl, "template must not be empty"
    for email, info in tpl.items():
        assert "password" not in info, f"{email} carries inline password"
        assert info.get("password_env"), f"{email} lacks password_env"
