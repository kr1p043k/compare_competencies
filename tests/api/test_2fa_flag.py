"""Роут-уровневые тесты глобального флага 2FA (без live БД, где возможно)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.startup as startup_mod
import src.security_settings as security_mod
from src.api_pkg import create_app
from src.api_pkg.routers import admin as admin_mod
from tests.conftest import open_all_gates

ADMIN_TOKEN_USER = {
    "uid": "00000000-0000-0000-0000-000000000001",
    "u": "admin@t.local",
    "r": "admin",
}


@pytest.fixture(autouse=True)
def _silence_startup(monkeypatch):
    """Отключаем startup, требующий PostgreSQL: тесты - роут-уровневые."""
    monkeypatch.setattr(startup_mod, "run_startup", AsyncMock())
    monkeypatch.setattr(startup_mod, "_run_bootstrap_safely", AsyncMock())


def _client(app=None):
    return TestClient(create_app() if app is None else app)


def test_anonymous_security_settings_unauthorized():
    """1. Анонимный GET/PUT /api/admin/security/settings -> 401."""
    client = _client()
    assert client.get("/api/admin/security/settings").status_code == 401
    assert (
        client.put("/api/admin/security/settings", json={"two_factor_enabled": True}).status_code
        == 401
    )


def test_security_settings_get_shape_and_put_audit(monkeypatch):
    """2. GET отдает нужную форму; PUT вызывает audit_action с нужным action."""
    app = open_all_gates(create_app())
    monkeypatch.setattr(security_mod, "load_security_settings", lambda: {"two_factor_enabled": False})
    monkeypatch.setattr(
        admin_mod, "save_security_settings", lambda patch: {"two_factor_enabled": patch["two_factor_enabled"]}
    )
    mock_audit = AsyncMock()
    monkeypatch.setattr(admin_mod, "audit_action", mock_audit)

    client = _client(app)
    r = client.get("/api/admin/security/settings")
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"two_factor_enabled", "enrolled_admins", "active_admins"} <= set(body)
    assert body["two_factor_enabled"] is False

    r = client.put("/api/admin/security/settings", json={"two_factor_enabled": True})
    assert r.status_code == 200, r.text
    assert r.json()["two_factor_enabled"] is True
    mock_audit.assert_awaited_once()
    assert mock_audit.await_args.args[1] == "admin.security.settings"


def test_security_settings_put_validation_errors():
    """3. PUT с лишним ключом -> 422; пустое тело -> 400 'Нет изменений'."""
    app = open_all_gates(create_app())
    client = _client(app)

    r = client.put(
        "/api/admin/security/settings", json={"two_factor_enabled": True, "foo": "bar"}
    )
    assert r.status_code == 422, r.text

    r = client.put("/api/admin/security/settings", json={})
    assert r.status_code == 400, r.text
    assert "Нет изменений" in r.json()["detail"]


def test_2fa_endpoints_409_when_flag_off(monkeypatch):
    """4. При выключенном флаге setup/confirm/verify/recovery отвечают 409 до БД."""
    monkeypatch.setattr(auth_mod, "two_factor_enabled", lambda: False)
    monkeypatch.setattr(auth_mod, "get_pool", lambda: object())

    client = _client(create_app())
    cases = [
        ("/api/auth/2fa/setup", {}),
        ("/api/auth/2fa/confirm", {"code": "123456"}),
        ("/api/auth/2fa/verify", {"challenge_id": "x", "code": "123456"}),
        ("/api/auth/2fa/recovery", {"challenge_id": "x", "code": "123456"}),
    ]
    for path, body in cases:
        r = client.post(path, json=body)
        assert r.status_code == 409, f"{path}: {r.text}"
        assert "выключена" in r.json().get("detail", "").lower()


class _MissingTotpPool:
    """Фейковый пул: fetchrow падает только на admin_totp."""

    async def fetchrow(self, query: str, *args):
        if "admin_totp" in query:
            raise RuntimeError("table missing")
        if "FROM users" in query:
            return {
                "email": ADMIN_TOKEN_USER["u"],
                "role": "admin",
                "full_name": "Admin",
            }
        return None


def test_auth_me_totp_missing_table(monkeypatch):
    """5. При включенном флаге и отсутствующей таблице /auth/me работает и totp_enabled=False."""
    monkeypatch.setattr(
        auth_mod, "get_current_user", AsyncMock(return_value=dict(ADMIN_TOKEN_USER))
    )
    monkeypatch.setattr(auth_mod, "two_factor_enabled", lambda: True)
    monkeypatch.setattr(auth_mod, "get_pool", lambda: _MissingTotpPool())

    client = _client(create_app())
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer x.y"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["totp_enabled"] is False
    assert body["totp_locked_until"] is None
    assert body["two_factor_enabled"] is True
