"""Admin console v1: route registration, auth guards, role validation and audit utilities.

All tests run without a live DB pool: routes are checked via TestClient
without the ASGI lifespan, and the few handlers reached are wired so that
they fail before touching PostgreSQL.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from starlette.requests import Request

from src.api_pkg import create_app
from src.api_pkg import request_logger as rl
from src.api_pkg.routers import auth as auth_mod
from src.models.krm_models import AuditLog
from tests.conftest import all_route_paths, open_all_gates, route_gate_roles

ADMIN_TOKEN_USER = {"uid": "00000000-0000-0000-0000-000000000001", "u": "admin@t.local", "r": "admin"}


def _client(app=None):
    return TestClient(app if app is not None else create_app())


def test_admin_routes_are_registered():
    paths = all_route_paths(create_app())
    wanted = [
        "/api/admin/users",
        "/api/admin/users/{user_id}",
        "/api/admin/sessions",
        "/api/admin/audit",
        "/api/admin/audit/actions",
        "/api/admin/monitoring",
        "/api/admin/monitoring/summary",
    ]
    for path in wanted:
        assert path in paths, f"missing {path}"
        v1 = path.replace("/api/", "/api/v1/")
        assert v1 in paths, f"missing {v1}"


def test_admin_endpoints_require_admin_role():
    app = create_app()
    for path, method in [
        ("/api/admin/users", "GET"),
        ("/api/admin/sessions", "GET"),
        ("/api/admin/audit", "GET"),
        ("/api/admin/monitoring", "GET"),
        ("/api/admin/monitoring/summary", "GET"),
    ]:
        roles = route_gate_roles(app, path, method)
        assert roles, f"no gate roles for {method} {path}"
        assert any("admin" in t for t in roles), f"admin missing from {method} {path}: {roles}"


def test_unauthenticated_admin_paths_return_401():
    client = _client()
    for path in [
        "/api/admin/users",
        "/api/admin/sessions",
        "/api/admin/audit",
        "/api/admin/monitoring",
    ]:
        assert client.get(path).status_code == 401, path


def test_patch_invalid_role_returns_400_before_db():
    app = open_all_gates(create_app())
    client = _client(app)
    bogus_uuid = "00000000-0000-0000-0000-000000000099"
    r = client.patch(f"/api/admin/users/{bogus_uuid}", json={"role": "superuser"})
    assert r.status_code == 400, r.text
    assert "Invalid role" in r.json().get("detail", "")


def test_audit_limit_zero_rejected_at_validation():
    app = open_all_gates(create_app())
    client = _client(app)
    r = client.get("/api/admin/audit?limit=0")
    assert r.status_code == 422, r.text


def test_audit_action_filter_compiles_to_like_prefix():
    expr = AuditLog.action.startswith("admin.user", autoescape=True)
    sql = str(
        select(1).where(expr).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    # literal_binds экранирует % как %% для psycopg, поэтому сверяем префикс и ESCAPE,
    # а не точный текст "%".
    assert "LIKE 'admin.user' || '" in sql, sql
    assert "ESCAPE '/'" in sql, sql


class _FakePool:
    def __init__(self, confirmed: bool = False, locked_until=None):
        self._confirmed = confirmed
        self._locked_until = locked_until

    async def fetchrow(self, query: str, *args):
        if "admin_totp" in query:
            return {
                "secret": "secret",
                "confirmed": self._confirmed,
                "recovery_codes": None,
                "failed_attempts": 0,
                "locked_until": self._locked_until,
                "retry_in": 0,
            }
        if "FROM users" in query:
            return {"email": "admin@t.local", "role": "admin", "full_name": "Admin"}
        return None


def test_auth_me_contract_reflects_totp_state(monkeypatch):
    app = create_app()
    monkeypatch.setattr(auth_mod, "get_current_user", AsyncMock(return_value=dict(ADMIN_TOKEN_USER)))
    monkeypatch.setattr(auth_mod, "get_pool", lambda: _FakePool(confirmed=True))
    client = _client(app)
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer x.y"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["username"] == "admin@t.local"
    assert body["role"] == "admin"
    assert "totp_enabled" in body
    assert "totp_locked_until" in body
    assert body["totp_enabled"] is True
    assert body["totp_locked_until"] is None


def test_audit_action_tolerates_missing_pool(monkeypatch):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/admin/users/x",
        "query_string": b"",
        "headers": [],
    }
    monkeypatch.setattr("src.db.get_pool", lambda: (_ for _ in ()).throw(RuntimeError("no pool")))
    monkeypatch.setattr(auth_mod, "get_current_user", AsyncMock(return_value=None))
    out = asyncio.run(rl.audit_action(Request(scope), "admin.user.probe", "target=none"))
    assert out["action"] == "admin.user.probe"
    assert out["actor_email"] == "anonymous"
    assert out["target"] == "target=none"
