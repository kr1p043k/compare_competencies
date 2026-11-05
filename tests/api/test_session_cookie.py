"""Сессионная cookie: HttpOnly, SameSite, Secure по флагу.

Токен больше не хранится в localStorage: его кладёт сервер в HttpOnly-cookie,
поэтому JS на странице его не читает и XSS не даёт угнать сессию.
"""
from __future__ import annotations

import os
import socket
import urllib.parse

import pytest
from starlette.responses import Response

import src.api_pkg.routers.auth as auth_mod


def _set_cookie_header() -> str:
    response = Response()
    auth_mod._set_session_cookie(response, "test-token")
    return response.headers["set-cookie"].lower()


class TestCookieAttributes:
    def test_httponly_is_set(self):
        """Без HttpOnly токен читается из document.cookie - это и был риск."""
        assert "httponly" in _set_cookie_header()

    def test_samesite_lax(self):
        assert "samesite=lax" in _set_cookie_header()

    def test_path_is_root(self):
        assert "path=/" in _set_cookie_header()

    def test_secure_off_by_default(self):
        """На localhost по http cookie с Secure не сохранилась бы."""
        assert "secure" not in _set_cookie_header()

    def test_secure_on_when_configured(self, monkeypatch):
        monkeypatch.setattr(auth_mod.config, "COOKIE_SECURE", True)
        assert "secure" in _set_cookie_header()

    def test_value_is_the_token(self):
        assert "token=test-token" in _set_cookie_header()


TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)


def _db_reachable() -> bool:
    try:
        parts = urllib.parse.urlsplit(TEST_DB_URL)
        with socket.create_connection(
            (parts.hostname or "localhost", parts.port or 5432), timeout=2
        ):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _db_reachable(), reason="local postgres unavailable")
@pytest.mark.asyncio
async def test_login_sets_httponly_cookie():
    """Настоящий вход ставит cookie, а не только отдаёт токен в теле."""
    import asyncpg
    from httpx import ASGITransport, AsyncClient

    import src.db as dbmod
    from src.api_pkg import create_app

    email = "cookie-check@example.test"
    password = "Cookie_Check_123"
    old = dbmod.pool
    pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2, command_timeout=30)
    dbmod.pool = pool
    try:
        await pool.execute("DELETE FROM users WHERE lower(email) = $1", email)
        await pool.execute(
            """INSERT INTO users (email, password_hash, role, full_name, is_active)
               VALUES ($1, crypt($2, gen_salt('bf', 10)), 'teacher', 'Cookie Check', true)""",
            email, password,
        )
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://t"
        ) as client:
            r = await client.post(
                "/api/auth/login", json={"email": email, "password": password}
            )
            assert r.status_code == 200, r.text
            cookies = r.headers.get_list("set-cookie")
            token_cookie = [c for c in cookies if c.startswith("token=")]
            assert token_cookie, "вход обязан ставить cookie сессии"
            assert "httponly" in token_cookie[0].lower()
    finally:
        await pool.execute("DELETE FROM users WHERE lower(email) = $1", email)
        dbmod.pool = old
        await pool.close()
