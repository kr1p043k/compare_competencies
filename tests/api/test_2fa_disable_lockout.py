"""Отключение 2FA: неверные коды считаются, после пяти - блокировка.

Раньше диалог отключения принимал неограниченные попытки подбора кода (лимит
запросов 10/мин не считал именно неверные коды). Теперь disable использует тот
же разбор, что и вход.
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

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)
EMAIL = f"2fa-disable-lock-{os.getpid()}@example.test"
PASSWORD = "Disable_Lock_Password_123"
MAX_FAILURES = 5


def _db_reachable() -> bool:
    try:
        parts = urllib.parse.urlsplit(TEST_DB_URL)
        with socket.create_connection(
            (parts.hostname or "localhost", parts.port or 5432), timeout=2
        ):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="local postgres unavailable")


async def _drop_user(pool: asyncpg.Pool) -> None:
    await pool.execute(
        "DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE lower(email) = $1)",
        EMAIL,
    )
    await pool.execute(
        "DELETE FROM admin_totp WHERE user_id IN (SELECT id FROM users WHERE lower(email) = $1)",
        EMAIL,
    )
    await pool.execute("DELETE FROM users WHERE lower(email) = $1", EMAIL)


async def _enrolled(client: AsyncClient) -> tuple[str, str]:
    """Подключить TOTP, вернуть (сессия, секрет)."""
    data = (
        await client.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    ).json()
    assert data.get("status") == "totp_setup_required", data
    enroll = data["enroll_id"]
    setup = await client.post("/api/auth/2fa/setup", json={"enroll_id": enroll})
    assert setup.status_code == 200, setup.text
    secret = setup.json()["secret"]
    confirm = await client.post(
        "/api/auth/2fa/confirm",
        json={"code": auth_mod._totp_now(secret), "enroll_id": enroll},
    )
    assert confirm.status_code == 200, confirm.text
    return confirm.json()["token"], secret


@pytest.fixture()
async def env(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    import src.db as dbmod

    async def _flag_on(_pool: Any) -> bool:
        return True

    monkeypatch.setattr(auth_mod, "get_two_factor_enabled", _flag_on)

    old = dbmod.pool
    try:
        pool = await asyncpg.create_pool(
            TEST_DB_URL, min_size=1, max_size=2, command_timeout=30
        )
    except Exception:  # pragma: no cover
        pytest.skip("local postgres unavailable")
    dbmod.pool = pool
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()
    await auth_mod._ensure_totp_table(pool)
    await _drop_user(pool)
    await pool.execute(
        """INSERT INTO users (email, password_hash, role, full_name, is_active)
           VALUES ($1, crypt($2, gen_salt('bf', 10)), 'admin', 'Disable Lock', true)""",
        EMAIL,
        PASSWORD,
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://t"
        ) as client:
            yield {"client": client, "pool": pool}
    finally:
        await _drop_user(pool)
        dbmod.pool = old
        await pool.close()


@pytest.mark.asyncio
async def test_wrong_disable_codes_lock_the_account(env):
    client = env["client"]
    token, secret = await _enrolled(client)
    headers = {"Authorization": f"Bearer {token}"}

    # 4 неверных - обычная ошибка кода, без блокировки
    for attempt in range(MAX_FAILURES - 1):
        r = await client.post("/api/auth/2fa/disable", json={"code": "000000"}, headers=headers)
        assert r.status_code == 400, f"попытка {attempt + 1}: {r.status_code} {r.text}"

    # 5-я неверная - срабатывает блокировка
    locked = await client.post("/api/auth/2fa/disable", json={"code": "000000"}, headers=headers)
    assert locked.status_code == 429, locked.text
    assert "Retry-After" in locked.headers

    # пока действует блокировка, даже верный код не проходит
    still = await client.post(
        "/api/auth/2fa/disable",
        json={"code": auth_mod._totp_now(secret)},
        headers=headers,
    )
    assert still.status_code == 429, still.text

    # 2FA не отключилась
    rows = await env["pool"].fetchval(
        "SELECT count(*) FROM admin_totp WHERE user_id = (SELECT id FROM users WHERE lower(email) = $1)",
        EMAIL,
    )
    assert rows == 1
