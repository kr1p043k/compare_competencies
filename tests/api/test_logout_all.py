"""Выход отовсюду гасит все сессии пользователя, не только текущую.

Сценарий угона: у злоумышленника свой токен. Выход на своём устройстве его не
затрагивает, поэтому нужен явный «закрыть все сессии». Проверяем, что после
POST /auth/logout-all не работает ни одна ранее выданная сессия.
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
EMAIL = f"logout-all-{os.getpid()}@example.test"
PASSWORD = "Logout_All_Password_123"


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
    await pool.execute("DELETE FROM users WHERE lower(email) = $1", EMAIL)


@pytest.fixture()
async def env(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    import src.db as dbmod

    # Флаг 2FA выключен: вход одношаговый, тест про сессии.
    async def _flag_off(_pool: Any) -> bool:
        return False

    monkeypatch.setattr(auth_mod, "get_two_factor_enabled", _flag_off)

    old = dbmod.pool
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=3, command_timeout=30)
    dbmod.pool = pool
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()
    await _drop(pool)
    await pool.execute(
        """INSERT INTO users (email, password_hash, role, full_name, is_active)
           VALUES ($1, crypt($2, gen_salt('bf', 10)), 'admin', 'Logout All', true)""",
        EMAIL,
        PASSWORD,
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


async def _login(client: AsyncClient) -> str:
    r = await client.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 200, r.text
    token = r.json().get("token")
    assert token, r.text
    return token


@pytest.mark.asyncio
async def test_logout_all_revokes_every_session(env):
    client = env["client"]

    first = await _login(client)
    second = await _login(client)
    assert first != second

    h1 = {"Authorization": f"Bearer {first}"}
    h2 = {"Authorization": f"Bearer {second}"}
    assert (await client.get("/api/auth/me", headers=h1)).status_code == 200
    assert (await client.get("/api/auth/me", headers=h2)).status_code == 200

    out = await client.post("/api/auth/logout-all", headers=h1)
    assert out.status_code == 200, out.text
    assert out.json()["revoked"] >= 2

    # ни одна сессия больше не действует, включая ту, что делала запрос
    assert (await client.get("/api/auth/me", headers=h1)).status_code == 401
    assert (await client.get("/api/auth/me", headers=h2)).status_code == 401


@pytest.mark.asyncio
async def test_logout_all_requires_auth(env):
    r = await env["client"].post("/api/auth/logout-all")
    assert r.status_code == 401, r.text
