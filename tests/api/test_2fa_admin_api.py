"""2FA: админский API - единый переключатель и отзыв сессий при включении.

Живая локальная PostgreSQL, тот же приём, что в tests/api/test_2fa.py: пул
создаётся на тест и подменяет src.db.pool.
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
from src.api_pkg.security_settings import set_two_factor_enabled

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)

# Уникальный email на процесс: тесты идут в общей БД.
EMAIL = f"2fa-admin-api-{os.getpid()}@example.test"
PASSWORD = "AdminApi_Password_123"


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


async def _login(client: AsyncClient) -> Any:
    return await client.post(
        "/api/auth/login", json={"email": EMAIL, "password": PASSWORD}
    )


async def _enrolled_token(client: AsyncClient) -> tuple[str, str]:
    """Подключить TOTP, вернуть (сессия, секрет). Секрет нужен для входа B."""
    data = (await _login(client)).json()
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


async def _second_session(client: AsyncClient, secret: str) -> str:
    """Второе устройство того же админа: login -> challenge -> verify."""
    data = (await _login(client)).json()
    assert data.get("status") == "totp_required", data
    verify = await client.post(
        "/api/auth/2fa/verify",
        json={"challenge_id": data["challenge_id"], "code": auth_mod._totp_now(secret)},
    )
    assert verify.status_code == 200, verify.text
    return verify.json()["token"]


@pytest.fixture()
async def env(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    import src.db as dbmod

    # Флаг подменяем в модуле, а не в общей БД: соседний воркер на той же базе
    # может переключить app_settings посреди теста. Админский эндпоинт читает
    # флаг из БД сам, но проверки входа идут через эту подмену.
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
           VALUES ($1, crypt($2, gen_salt('bf', 10)), 'admin', '2FA Admin API', true)""",
        EMAIL,
        PASSWORD,
    )
    app = create_app()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield {"client": client, "pool": pool}
    finally:
        with contextlib.suppress(Exception):
            await set_two_factor_enabled(pool, False)
        await _drop_user(pool)
        dbmod.pool = old
        await pool.close()


@pytest.mark.asyncio
async def test_dead_feature_flag_endpoints_are_removed(env):
    """Второй переключатель 2FA удалён: /admin/features больше не существует.

    Раньше здесь был флаг 2FA в отдельном файле features.json, который login
    не читал: тумблер переключался, а второй фактор не включался. Теперь
    единственный источник правды - app_settings, а мёртвые эндпоинты убраны.
    """
    client = env["client"]
    token, _secret = await _enrolled_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    about = await client.get("/api/admin/features", headers=headers)
    assert about.status_code == 404, about.text

    put = await client.put(
        "/api/admin/features/totp_2fa_enabled",
        json={"enabled": True},
        headers=headers,
    )
    assert put.status_code == 404, put.text


@pytest.mark.asyncio
async def test_enabling_2fa_revokes_other_admin_sessions(env):
    """Сессия, добытая до включения второго фактора, перестаёт работать."""
    client = env["client"]

    # сессия A: подключает TOTP и получает токен; запоминаем секрет
    token_a, secret = await _enrolled_token(client)

    # сессия B: тот же админ входит вторым устройством
    token_b = await _second_session(client, secret)
    assert token_b and token_b != token_a

    # включаем 2FA из сессии A
    on = await client.put(
        "/api/admin/security/2fa/settings",
        json={"enabled": True},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert on.status_code == 200, on.text
    assert on.json()["enabled"] is True

    me_a = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token_a}"})
    me_b = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token_b}"})
    assert me_a.status_code == 200, "сессия, включившая 2FA, должна остаться"
    assert me_b.status_code == 401, "чужая админ-сессия должна отозваться"
