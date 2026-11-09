"""Сброс 2FA требует подтверждения и для СВОЕГО аккаунта.

Раньше step-up (свой пароль) спрашивался только при сбросе 2FA ДРУГОМУ
админу. Угнанная админ-сессия могла снять собственный второй фактор одним
запросом. Теперь пароль нужен всегда для админского аккаунта.
"""
from __future__ import annotations

import contextlib
import os
import socket
import urllib.parse
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

import src.api_pkg.routers.auth as auth_mod
from src.api_pkg import create_app

DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)
EMAIL = f"2fa-reset-stepup-{os.getpid()}@example.test"
PASSWORD = "Reset_Stepup_Password_123"


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
        "DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE lower(email) = $1)",
        EMAIL,
    )
    await pool.execute("DELETE FROM admin_totp WHERE user_id IN (SELECT id FROM users WHERE lower(email) = $1)", EMAIL)
    await pool.execute("DELETE FROM users WHERE lower(email) = $1", EMAIL)


@pytest.fixture()
async def env(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    import src.db as dbmod

    # Флаг выключен: вход одношаговый, тест проверяет именно step-up сброса.
    async def _flag_off(_pool: Any) -> bool:
        return False

    monkeypatch.setattr(auth_mod, "get_two_factor_enabled", _flag_off)
    monkeypatch.delenv("TWO_FACTOR", raising=False)

    old = dbmod.pool
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2, command_timeout=30)
    dbmod.pool = pool
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()
    await _drop(pool)
    uid = await pool.fetchval(
        """INSERT INTO users (email, password_hash, role, full_name, is_active)
           VALUES ($1, crypt($2, gen_salt('bf', 10)), 'admin', 'Reset Stepup', true)
        RETURNING id""",
        EMAIL,
        PASSWORD,
    )
    # Строка 2FA, чтобы reset было что удалять.
    await pool.execute(
        """INSERT INTO admin_totp (user_id, secret, confirmed, failed_attempts)
           VALUES ($1, 'dummy', true, 0)""",
        uid,
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://t"
        ) as client:
            yield {"client": client, "pool": pool, "uid": str(uid)}
    finally:
        await _drop(pool)
        dbmod.pool = old
        await pool.close()


async def _token(client: AsyncClient) -> str:
    r = await client.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 200, r.text
    token = r.json().get("token")
    assert token, r.text
    return token


@pytest.mark.asyncio
async def test_self_reset_requires_password(env):
    client = env["client"]
    uid = env["uid"]
    token = await _token(client)
    headers = {"Authorization": f"Bearer {token}"}

    # без пароля - отказ
    no_pw = await client.post(f"/api/admin/users/{uid}/2fa/reset", headers=headers)
    assert no_pw.status_code == 403, no_pw.text

    # неверный пароль - отказ
    bad = await client.post(
        f"/api/admin/users/{uid}/2fa/reset",
        json={"caller_password": "definitely-wrong"},
        headers=headers,
    )
    assert bad.status_code == 403, bad.text

    # строка 2FA на месте
    assert await env["pool"].fetchval(
        "SELECT count(*) FROM admin_totp WHERE user_id = $1", __import__("uuid").UUID(uid)
    ) == 1

    # верный пароль - сброс проходит
    ok = await client.post(
        f"/api/admin/users/{uid}/2fa/reset",
        json={"caller_password": PASSWORD},
        headers=headers,
    )
    assert ok.status_code == 200, ok.text
    assert await env["pool"].fetchval(
        "SELECT count(*) FROM admin_totp WHERE user_id = $1", __import__("uuid").UUID(uid)
    ) == 0
