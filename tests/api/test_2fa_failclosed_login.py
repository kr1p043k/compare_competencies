"""E2E: при сбое чтения флага 2FA вход не должен молча выдавать токен.

Мастер-флаг читается из app_settings. Если чтение падает (таблица/строка
недоступны), helper обязан трактовать состояние как «включено» (fail-closed):
лучше лишний раз спросить фактор, чем тихо его отключить. Проверяем это на
живом входе, а не на заглушке гейта: патчим низкоуровневый read_two_factor,
чтобы настоящий get_two_factor_enabled сам вычислил True.
"""
from __future__ import annotations

import os
import socket
import urllib.parse
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.security_settings as ss
from src.api_pkg import create_app

DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)
EMAIL = f"2fa-failclosed-{os.getpid()}@example.test"
PASSWORD = "FailClosed_Password_123"


def _db_reachable() -> bool:
    try:
        parts = urllib.parse.urlsplit(DB_URL)
        with socket.create_connection(
            (parts.hostname or "localhost", parts.port or 5432), timeout=2
        ):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="local postgres unavailable")


async def _drop(pool: asyncpg.Pool) -> None:
    await pool.execute(
        "DELETE FROM admin_totp WHERE user_id IN (SELECT id FROM users WHERE lower(email) = $1)",
        EMAIL,
    )
    await pool.execute("DELETE FROM users WHERE lower(email) = $1", EMAIL)


@pytest.fixture()
async def env(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    import src.db as dbmod

    old = dbmod.pool
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2, command_timeout=30)
    dbmod.pool = pool
    await _drop(pool)
    uid = await pool.fetchval(
        """INSERT INTO users (email, password_hash, role, full_name, is_active)
           VALUES ($1, crypt($2, gen_salt('bf', 10)), 'admin', 'Fail Closed', true)
        RETURNING id""",
        EMAIL,
        PASSWORD,
    )
    # Подключённый TOTP: при включённом флаге вход обязан требовать код.
    await ss._ensure_table(pool)
    await pool.execute(
        """INSERT INTO admin_totp (user_id, secret, confirmed, failed_attempts)
           VALUES ($1, 'dummy', true, 0)""",
        uid,
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://t"
        ) as client:
            yield {"client": client, "pool": pool}
    finally:
        await _drop(pool)
        dbmod.pool = old
        await pool.close()


async def _login(client: AsyncClient) -> Any:
    return await client.post(
        "/api/auth/login", json={"email": EMAIL, "password": PASSWORD}
    )


@pytest.mark.asyncio
async def test_read_error_denies_login(env, monkeypatch):
    """read_two_factor=ERROR -> флаг считается включённым, токен не выдаём."""

    async def _error(_pool: Any) -> ss.TwoFactorRead:
        return ss.TwoFactorRead(False, ss.TwoFactorReadState.ERROR)

    monkeypatch.setattr(ss, "read_two_factor", _error)
    # Прямая проверка helper-а: на error он обязан сказать True.
    assert await auth_mod.get_two_factor_enabled(env["pool"]) is True

    r = await _login(env["client"])
    body = r.json()
    assert body.get("status") == "totp_required", body
    assert not body.get("token"), "при сбое чтения флага токен выдавать нельзя"


@pytest.mark.asyncio
async def test_flag_read_ok_and_off_returns_token(env, monkeypatch):
    """Контроль: успешное чтение «выключено» -> обычный вход по паролю."""

    async def _ok_off(_pool: Any) -> ss.TwoFactorRead:
        return ss.TwoFactorRead(False, ss.TwoFactorReadState.OK)

    monkeypatch.setattr(ss, "read_two_factor", _ok_off)
    r = await _login(env["client"])
    assert r.status_code == 200, r.text
    assert r.json().get("token"), r.text
