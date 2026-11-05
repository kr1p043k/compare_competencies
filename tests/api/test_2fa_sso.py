"""SSO не должен обходить второй фактор у администратора.

Регрессия на CRITICAL-находку: `/auth/sso` выпускал полноценный токен любому,
включая admin, ни разу не заглянув в admin_totp. При включённом SSO это делало
второй фактор декоративным: достаточно войти через hub.sfedu.ru.

Внешний academic-api подменяется фейковым httpx-клиентом: сеть в тестах не нужна.
"""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api_pkg.routers import academic, auth as auth_mod


def _request() -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/auth/sso",
        "raw_path": b"/api/auth/sso",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("t", 80),
    }
    return Request(scope)


class _Pool:
    def __init__(self, role: str = "admin"):
        self.role = role
        self.executed: list[str] = []

    async def fetchrow(self, sql, *args):
        return {
            "id": "11111111-1111-1111-1111-111111111111",
            "email": "adm@sfedu.ru",
            "role": self.role,
            "full_name": "Admin",
        }

    async def execute(self, sql, *args):
        self.executed.append(sql)
        return "OK"


class _Response:
    status_code = 200
    text = "ok"

    def json(self):
        return {"email": "adm@sfedu.ru", "sub": "1", "username": "adm"}


class _Client:
    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *a, **kw):
        return _Response()


@pytest.fixture()
def sso_on(monkeypatch):
    monkeypatch.setattr(academic.config, "ACADEMIC_SSO_ENABLED", True, raising=False)
    monkeypatch.setattr(academic.config, "ACADEMIC_EMAIL_DOMAIN", "sfedu.ru", raising=False)
    monkeypatch.setattr(academic, "httpx", type("M", (), {"AsyncClient": _Client, "HTTPError": Exception}))
    monkeypatch.setattr(academic, "_jwt_username", lambda _t: "adm", raising=False)
    monkeypatch.setattr(academic, "client_ip", lambda _r: "127.0.0.1", raising=False)


async def test_sso_admin_with_confirmed_totp_gets_challenge(sso_on, monkeypatch):
    """Админ с подключённым 2FA не получает токен - только challenge."""
    pool = _Pool("admin")
    monkeypatch.setattr(academic, "get_pool", lambda: pool)

    async def fake_state(_pool, _uid):
        return {"confirmed": True}

    async def fake_challenge(_uid):
        return "chal-1"

    monkeypatch.setattr(auth_mod, "_totp_state", fake_state)
    monkeypatch.setattr(auth_mod, "_issue_challenge", fake_challenge)

    body = await academic.sso_login(academic.SsoRequest(token="hub.jwt.token"), _request())

    assert body.get("status") == "totp_required"
    assert "token" not in body
    assert not any("INSERT INTO sessions" in sql for sql in pool.executed), \
        "сессия не должна создаваться до прохождения второго фактора"


async def test_sso_admin_without_totp_gets_token(sso_on, monkeypatch):
    """Без подключённого 2FA вход через SSO работает как раньше."""
    pool = _Pool("admin")
    monkeypatch.setattr(academic, "get_pool", lambda: pool)

    async def fake_state(_pool, _uid):
        return None

    monkeypatch.setattr(auth_mod, "_totp_state", fake_state)

    body = await academic.sso_login(academic.SsoRequest(token="hub.jwt.token"), _request())

    assert body.get("token")
    assert any("INSERT INTO sessions" in sql for sql in pool.executed)


async def test_sso_non_admin_gets_token_even_with_totp_row(sso_on, monkeypatch):
    """Не-admin не блокируется проверкой второго фактора."""
    pool = _Pool("teacher")
    monkeypatch.setattr(academic, "get_pool", lambda: pool)

    async def fake_state(_pool, _uid):
        raise AssertionError("для teacher проверка 2FA не должна вызываться")

    monkeypatch.setattr(auth_mod, "_totp_state", fake_state)

    body = await academic.sso_login(academic.SsoRequest(token="hub.jwt.token"), _request())
    assert body.get("token")


async def test_sso_disabled_returns_503(monkeypatch):
    monkeypatch.setattr(academic.config, "ACADEMIC_SSO_ENABLED", False, raising=False)
    with pytest.raises(HTTPException) as exc:
        await academic.sso_login(academic.SsoRequest(token="x"), _request())
    assert exc.value.status_code == 503
