"""Рубильник второго фактора: включён/выключен без переписывания кода.

Проверяем ровно то, что обещано владельцу:
  * по умолчанию выключено;
  * при выключенном флаге подключить второй фактор нельзя (404);
  * у уже подключённых он продолжает требоваться (флаг - не способ обойти защиту);
  * /auth/2fa/status честно сообщает enabled/available/method/reason.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api_pkg.routers import auth


def _request(headers=None) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/auth/2fa/status",
        "raw_path": b"/api/auth/2fa/status",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("127.0.0.1", 1),
        "server": ("t", 80),
    }
    return Request(scope)


@pytest.fixture()
def flag(monkeypatch):
    def _set(enabled: bool, method: str = "totp"):
        monkeypatch.setattr(auth.config, "TWO_FACTOR_ENABLED", enabled, raising=False)
        monkeypatch.setattr(auth.config, "TWO_FACTOR_METHOD", method, raising=False)

    return _set


# ---------- availability ----------

def test_two_factor_is_off_by_default(flag):
    flag(False)
    assert auth._two_factor_available() is False
    assert "отключена" in (auth._two_factor_unavailable_reason() or "")


def test_two_factor_available_when_enabled(flag):
    flag(True)
    assert auth._two_factor_available() is True
    assert auth._two_factor_unavailable_reason() is None


def test_unimplemented_channel_reports_reason(flag):
    # Задел: значение принимается, но канал не реализован - и это видно в статусе.
    flag(True, "email")
    assert auth._two_factor_available() is True
    reason = auth._two_factor_unavailable_reason()
    assert reason is not None and "email" in reason


def test_implemented_channel_has_no_reason(flag):
    flag(True, "TOTP")  # регистр не должен ломать проверку
    assert auth._two_factor_unavailable_reason() is None


# ---------- enrolment gate ----------

async def test_setup_rejected_when_disabled(flag, monkeypatch):
    flag(False)

    async def fake_admin(_request):
        return {"uid": "u-1", "r": "admin"}

    monkeypatch.setattr(auth, "_require_admin", fake_admin)
    with pytest.raises(HTTPException) as exc:
        await auth.totp_setup(_request())
    assert exc.value.status_code == 404
    assert "отключена" in exc.value.detail


async def test_confirm_rejected_when_disabled(flag, monkeypatch):
    flag(False)

    async def fake_admin(_request):
        return {"uid": "u-1", "r": "admin"}

    monkeypatch.setattr(auth, "_require_admin", fake_admin)
    with pytest.raises(HTTPException) as exc:
        await auth.totp_confirm(SimpleNamespace(code="123456"), _request())
    assert exc.value.status_code == 404


async def test_setup_still_admin_only_when_enabled(flag, monkeypatch):
    # Владелец: подключать может только admin. Проверяем, что гейт роли не снят.
    flag(True)

    async def fake_user(_request):
        return {"uid": "u-2", "r": "student"}

    monkeypatch.setattr(auth, "get_current_user", fake_user)
    with pytest.raises(HTTPException) as exc:
        await auth.totp_setup(_request())
    assert exc.value.status_code == 403


# ---------- status endpoint ----------

async def test_status_disabled(flag, monkeypatch):
    flag(False)

    async def fake_user(_request):
        return {"uid": "u-1", "r": "admin"}

    monkeypatch.setattr(auth, "get_current_user", fake_user)
    monkeypatch.setattr(auth, "get_pool", lambda: None)
    body = await auth.totp_status(_request())
    assert body["enabled"] is False
    assert body["available"] is False
    assert "отключена" in body["reason"]


async def test_status_enabled_without_totp(flag, monkeypatch):
    flag(True)

    async def fake_user(_request):
        return {"uid": "u-1", "r": "admin"}

    async def fake_state(_pool, _uid):
        return None

    monkeypatch.setattr(auth, "get_current_user", fake_user)
    monkeypatch.setattr(auth, "get_pool", lambda: object())
    monkeypatch.setattr(auth, "_totp_state", fake_state)
    body = await auth.totp_status(_request())
    assert body["enabled"] is False
    assert body["available"] is True
    assert body["method"] == "totp"


async def test_status_enabled_and_enrolled(flag, monkeypatch):
    flag(True)

    async def fake_user(_request):
        return {"uid": "u-1", "r": "admin"}

    async def fake_state(_pool, _uid):
        return {"confirmed": True}

    monkeypatch.setattr(auth, "get_current_user", fake_user)
    monkeypatch.setattr(auth, "get_pool", lambda: object())
    monkeypatch.setattr(auth, "_totp_state", fake_state)
    body = await auth.totp_status(_request())
    assert body["enabled"] is True
    assert body["available"] is True


async def test_status_requires_auth(flag, monkeypatch):
    flag(True)

    async def no_user(_request):
        return None

    monkeypatch.setattr(auth, "get_current_user", no_user)
    with pytest.raises(HTTPException) as exc:
        await auth.totp_status(_request())
    assert exc.value.status_code == 401
