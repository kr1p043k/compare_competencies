"""Сброс 2FA требует подтверждения и для СВОЕГО аккаунта.

Раньше step-up (свой пароль) спрашивался только при сбросе 2FA ДРУГОМУ
админу. Угнанная админ-сессия могла снять собственный второй фактор одним
запросом. Теперь пароль нужен всегда для админского аккаунта.

Тест намеренно не ходит через вход и таблицу `sessions`: параллельный воркер
на общей БД может отозвать админ-сессии (тесты включения 2FA), и тогда вход
становится недостоверным. Вызывающего подставляет monkeypatch, а сам предмет
проверки - step-up по паролю из БД.
"""
from __future__ import annotations

import contextlib
import os
import socket
import urllib.parse
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

import src.api_pkg.routers.admin as admin_mod
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
    # Лимитер общий на процесс: сбрасываем, чтобы чужие тесты в этом воркере
    # не оставили нам 429.
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()
    # Эндпоинт ходит через общий кэшированный движок (src.database._engine),
    # который привязан к чужому event loop/fоновым задачам и под нагрузкой
    # роняет запрос в asyncpg «another operation is in progress». Даём тесту
    # собственный движок на текущий loop: патчим фабрику сессий.
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import src.database as database_mod
    from src.config import settings

    own_engine = create_async_engine(settings.DATABASE_URL)
    own_maker = async_sessionmaker(own_engine, expire_on_commit=False)
    monkeypatch.setattr(database_mod, "async_session_factory", lambda: own_maker())
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

    caller = {"uid": str(uid), "u": EMAIL, "r": "admin", "t": None}

    async def _current_user(_request: Any) -> dict[str, Any]:
        return caller

    # admin.py импортирует имя на уровне модуля, зависимость роли - из auth:
    # патчим оба, чтобы запрос не трогал таблицу sessions.
    monkeypatch.setattr(auth_mod, "get_current_user", _current_user)
    monkeypatch.setattr(admin_mod, "get_current_user", _current_user)

    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://t"
        ) as client:
            yield {"client": client, "pool": pool, "uid": str(uid)}
    finally:
        await _drop(pool)
        dbmod.pool = old
        await pool.close()
        await own_engine.dispose()


@pytest.mark.asyncio
async def test_self_reset_requires_password(env):
    client = env["client"]
    uid = env["uid"]
    row_count = "SELECT count(*) FROM admin_totp WHERE user_id = $1"

    # без пароля - отказ
    no_pw = await client.post(f"/api/admin/users/{uid}/2fa/reset")
    assert no_pw.status_code == 403, no_pw.text

    # неверный пароль - отказ
    bad = await client.post(
        f"/api/admin/users/{uid}/2fa/reset",
        json={"caller_password": "definitely-wrong"},
    )
    assert bad.status_code == 403, bad.text

    # строка 2FA на месте
    assert await env["pool"].fetchval(row_count, UUID(uid)) == 1

    # верный пароль - сброс проходит
    ok = await client.post(
        f"/api/admin/users/{uid}/2fa/reset",
        json={"caller_password": PASSWORD},
    )
    assert ok.status_code == 200, ok.text
    assert await env["pool"].fetchval(row_count, UUID(uid)) == 0
