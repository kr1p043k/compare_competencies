"""Admin user management: role changes, session revocation and filters (live DB)."""
from __future__ import annotations

import hashlib
import os
import socket
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

import src.db as dbmod
from src import config
from src.api_pkg import create_app
from tests.conftest import open_all_gates

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)


def _db_reachable() -> bool:
    try:
        parts = __import__("urllib").parse.urlsplit(TEST_DB_URL)
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="local postgres unavailable")


def _async_url(url: str) -> str:
    """SQLAlchemy в приложении работает через asyncpg, а asyncpg-пул - через plain DSN."""
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://"):]
    return url


def _reset_db_caches() -> None:
    """_engine/_sessionmaker закешированы через lru_cache: без сброса тест возьмёт старый URL."""
    import src.database as database_mod

    database_mod._engine.cache_clear()
    database_mod._sessionmaker.cache_clear()


def _unique_tag() -> str:
    return f"pytest-{os.getpid()}-{uuid.uuid4().hex[:8]}"


async def _seed_user(pool: asyncpg.Pool, email: str, role: str) -> str:
    return await pool.fetchval(
        """INSERT INTO users (email, password_hash, role, full_name, is_active)
           VALUES ($1, crypt($2, gen_salt('bf', 8)), $3, $4, true) RETURNING id""",
        email, "Test_Pass_1234!", role, email,
    )


async def _insert_session(pool: asyncpg.Pool, user_id: str) -> None:
    await pool.execute(
        "INSERT INTO sessions (user_id, token_hash, ip_address, user_agent) VALUES ($1, $2, $3, $4)",
        user_id, hashlib.sha256(b"pytest-token").hexdigest(), "127.0.0.1", "pytest",
    )


async def _cleanup_users(pool: asyncpg.Pool, user_ids: list[str]) -> None:
    if not user_ids:
        return
    await pool.execute("DELETE FROM user_directions WHERE user_id = ANY($1)", user_ids)
    await pool.execute("DELETE FROM admin_totp WHERE user_id = ANY($1)", user_ids)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1)", user_ids)
    await pool.execute("DELETE FROM users WHERE id = ANY($1)", user_ids)


async def _deactivate_other_admins(pool: asyncpg.Pool, target_id: str) -> list[asyncpg.Record]:
    rows = await pool.fetch(
        "SELECT id, is_active FROM users WHERE role = 'admin' AND is_active = true AND id <> $1",
        target_id,
    )
    await pool.execute(
        "UPDATE users SET is_active = false WHERE role = 'admin' AND is_active = true AND id <> $1",
        target_id,
    )
    return rows


async def _restore_admins(pool: asyncpg.Pool, rows: list[asyncpg.Record]) -> None:
    for r in rows:
        await pool.execute("UPDATE users SET is_active = $1 WHERE id = $2", r["is_active"], r["id"])


@pytest.fixture()
async def env() -> AsyncIterator[dict[str, Any]]:
    tag = _unique_tag()
    admin_email = f"ac-admin-{tag}@example.test"
    teacher_email = f"ac-teacher-{tag}@example.test"

    old_db_url = dbmod.DATABASE_URL
    old_settings_url = config.settings.DATABASE_URL
    dbmod.DATABASE_URL = TEST_DB_URL
    config.DATABASE_URL = TEST_DB_URL
    config.settings.DATABASE_URL = _async_url(TEST_DB_URL)
    _reset_db_caches()

    pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2, command_timeout=30)
    admin_id = await _seed_user(pool, admin_email, "admin")
    teacher_id = await _seed_user(pool, teacher_email, "teacher")
    await _insert_session(pool, teacher_id)

    app = open_all_gates(create_app())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        yield {
            "client": client,
            "pool": pool,
            "admin_id": str(admin_id),
            "teacher_id": str(teacher_id),
            "admin_email": admin_email,
            "teacher_email": teacher_email,
        }

    await _cleanup_users(pool, [str(admin_id), str(teacher_id)])
    dbmod.DATABASE_URL = old_db_url
    config.DATABASE_URL = old_db_url
    config.settings.DATABASE_URL = old_settings_url
    _reset_db_caches()
    await pool.close()


async def test_list_users_with_filters(env):
    client = env["client"]
    r = await client.get("/api/admin/users", params={"q": env["teacher_email"], "role": "teacher", "is_active": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    assert body["users"][0]["username"] == env["teacher_email"]
    assert body["users"][0]["role"] == "teacher"
    assert body["users"][0]["is_active"] is True

    r2 = await client.get("/api/admin/users", params={"role": "superuser"})
    assert r2.status_code == 400, r2.text


async def test_user_detail_returns_sessions_and_totp(env):
    r = await env["client"].get(f"/api/admin/users/{env['teacher_id']}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["id"] == env["teacher_id"]
    assert body["user"]["role"] == "teacher"
    session_ids = {s["id"] for s in body["sessions"]}
    assert session_ids, "expected at least one session"


async def test_patch_role_round_trip(env):
    client = env["client"]
    url = f"/api/admin/users/{env['teacher_id']}"
    r = await client.patch(url, json={"role": "student"})
    assert r.status_code == 200, r.text
    assert r.json()["user"]["role"] == "student"

    r2 = await client.get(url)
    assert r2.json()["user"]["role"] == "student"

    await client.patch(url, json={"role": "teacher"})


async def test_self_demotion_is_blocked(env, monkeypatch):
    # admin.py импортирует get_current_user напрямую, поэтому патчим ссылку в его модуле.
    import src.api_pkg.routers.admin as admin_mod

    monkeypatch.setattr(
        admin_mod,
        "get_current_user",
        AsyncMock(return_value={"uid": env["admin_id"], "u": env["admin_email"], "r": "admin"}),
    )
    r = await env["client"].patch(
        f"/api/admin/users/{env['admin_id']}", json={"role": "teacher"}
    )
    assert r.status_code == 400, r.text
    assert "Нельзя изменить собственную роль" in r.json().get("detail", "")


async def test_last_active_admin_guard(env):
    rows = await _deactivate_other_admins(env["pool"], env["admin_id"])
    try:
        r = await env["client"].patch(
            f"/api/admin/users/{env['admin_id']}", json={"role": "teacher"}
        )
        assert r.status_code == 409, r.text
        assert "Нельзя оставить систему без активного администратора" in r.json().get("detail", "")
    finally:
        await _restore_admins(env["pool"], rows)


async def test_password_reset_revokes_other_sessions(env):
    pool = env["pool"]
    sess = await pool.fetchrow(
        "SELECT id, logged_out_at FROM sessions WHERE user_id = $1", env["teacher_id"]
    )
    assert sess and sess["logged_out_at"] is None

    r = await env["client"].post(
        f"/api/admin/users/{env['teacher_id']}/password",
        json={"new_password": "NewSecure_1234"},
    )
    assert r.status_code == 200, r.text

    after = await pool.fetchrow("SELECT logged_out_at FROM sessions WHERE id = $1", sess["id"])
    assert after["logged_out_at"] is not None


async def test_session_revoke_endpoint_closes_sessions(env):
    pool = env["pool"]
    await _insert_session(pool, env["teacher_id"])
    open_before = await pool.fetchval(
        "SELECT COUNT(*) FROM sessions WHERE user_id = $1 AND logged_out_at IS NULL",
        env["teacher_id"],
    )
    assert open_before >= 2

    r = await env["client"].post(f"/api/admin/users/{env['teacher_id']}/sessions/revoke")
    assert r.status_code == 200, r.text
    assert r.json()["revoked"] >= 1

    open_after = await pool.fetchval(
        "SELECT COUNT(*) FROM sessions WHERE user_id = $1 AND logged_out_at IS NULL",
        env["teacher_id"],
    )
    assert open_after == 0


async def test_sessions_list_filter_by_email(env):
    r = await env["client"].get("/api/admin/sessions", params={"user": env["teacher_email"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] >= 1
    assert all(s["email"] == env["teacher_email"] for s in body["sessions"])


async def test_2fa_reset_guard_requires_configured_totp(env):
    r = await env["client"].post(f"/api/admin/users/{env['teacher_id']}/2fa/reset")
    assert r.status_code == 400, r.text
    assert "2FA не настроена" in r.json().get("detail", "")
