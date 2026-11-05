"""TOTP 2FA для админа: настройка, вход, recovery, отключение.

Живая локальная PostgreSQL, тот же приём, что в tests/api/test_auth_bruteforce.py:
пул создаётся на тест и подменяет ``src.db.pool`` (иначе asyncpg-пул привязан к
чужому event loop). Каждый тест использует уникального администратора, чтобы
тесты не зависели от порядка запуска и не мутировали общую строку пользователя.
"""
from __future__ import annotations

import contextlib
import os
import socket
import urllib.parse
import uuid
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

_BASE_EMAIL = "totp-admin-{suffix}@example.test"
PASSWORD = "Totp_Test_Password_123"


def _db_reachable() -> bool:
    try:
        parts = urllib.parse.urlsplit(TEST_DB_URL)
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="local postgres unavailable")


def _totp(secret: str) -> str:
    return auth_mod._totp_now(secret)


def _bad_code(secret: str) -> str:
    return "000000" if _totp(secret) != "000000" else "111111"


async def _drop_user(pool: asyncpg.Pool, email: str) -> None:
    row = await pool.fetchrow("SELECT id FROM users WHERE lower(email) = $1", email)
    if row is not None:
        await pool.execute("DELETE FROM admin_totp WHERE user_id = $1", row["id"])
        await pool.execute("DELETE FROM sessions WHERE user_id = $1", row["id"])
    await pool.execute("DELETE FROM users WHERE lower(email) = $1", email)


async def _make_admin(pool: asyncpg.Pool, email: str, password: str) -> None:
    await _drop_user(pool, email)
    await pool.execute(
        """INSERT INTO users (email, password_hash, role, full_name, is_active)
           VALUES ($1, crypt($2, gen_salt('bf', 10)), 'admin', 'TOTP Admin', true)""",
        email, password,
    )


@pytest.fixture()
async def env() -> AsyncIterator[dict[str, Any]]:
    import src.db as dbmod
    from src.api_pkg import limiter as pkg_limiter

    old = dbmod.pool
    email = _BASE_EMAIL.format(suffix=uuid.uuid4().hex[:8])
    try:
        pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2, command_timeout=30)
    except Exception:  # pragma: no cover
        pytest.skip("local postgres unavailable")
    dbmod.pool = pool

    # Сбрасываем общее состояние, которое иначе протекает между тестами.
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()
    with contextlib.suppress(Exception):
        pkg_limiter._storage.reset()
    auth_mod._totp_challenges.clear()

    await auth_mod._ensure_totp_table(pool)
    await _make_admin(pool, email, PASSWORD)

    app = create_app()
    app.state.limiter = auth_mod.limiter
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield {"client": client, "pool": pool, "email": email, "password": PASSWORD}
    finally:
        await _drop_user(pool, email)
        dbmod.pool = old
        await pool.close()


async def _login(client: AsyncClient, email: str, password: str):
    return await client.post("/api/auth/login", json={"email": email, "password": password})


async def _admin_token(client: AsyncClient, email: str, password: str) -> str:
    r = await _login(client, email, password)
    assert r.status_code == 200, r.text
    return r.json()["token"]


async def _enable(client: AsyncClient, token: str) -> tuple[str, list[str]]:
    setup = await client.post("/api/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
    assert setup.status_code == 200, setup.text
    secret = setup.json()["secret"]
    confirm = await client.post(
        "/api/auth/2fa/confirm",
        json={"code": _totp(secret)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert confirm.status_code == 200, confirm.text
    return secret, confirm.json()["recovery_codes"]


# ------------------------------------------------------------------ setup

async def test_setup_returns_uri_and_secret(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    r = await client.post("/api/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["otpauth_uri"].startswith("otpauth://totp/")
    assert "secret=" in body["otpauth_uri"]
    assert len(body["secret"]) >= 16


async def test_setup_conflicts_when_already_confirmed(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    await _enable(client, token)
    r = await client.post("/api/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 409, r.text


async def test_setup_requires_auth(env):
    r = await env["client"].post("/api/auth/2fa/setup")
    assert r.status_code == 401, r.text


async def test_setup_forbidden_for_teacher(env):
    """Роль из БД: не-админ получает 403."""
    pool = env["pool"]
    await pool.execute("UPDATE users SET role = 'teacher' WHERE lower(email) = $1", env["email"])
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    r = await client.post("/api/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403, r.text


# ---------------------------------------------------------------- confirm

async def test_confirm_wrong_code_400(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    setup = await client.post("/api/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
    secret = setup.json()["secret"]
    r = await client.post(
        "/api/auth/2fa/confirm",
        json={"code": _bad_code(secret)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400, r.text


async def test_confirm_returns_ten_recovery_codes(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    _, codes = await _enable(client, token)
    assert len(codes) == 10
    assert len(set(codes)) == 10


# ------------------------------------------------------------ login flow

async def test_login_requires_totp_and_issues_no_token(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    await _enable(client, token)

    r = await _login(client, env["email"], env["password"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "totp_required"
    assert "token" not in body, "одного пароля недостаточно"
    assert body["challenge_id"]


async def test_verify_issues_token(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    secret, _ = await _enable(client, token)

    challenge = (await _login(client, env["email"], env["password"])).json()["challenge_id"]
    r = await client.post(
        "/api/auth/2fa/verify", json={"challenge_id": challenge, "code": _totp(secret)}
    )
    assert r.status_code == 200, r.text
    assert r.json().get("token")
    assert r.json()["role"] == "admin"


async def test_wrong_code_locks_after_five(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    secret, _ = await _enable(client, token)
    bad = _bad_code(secret)

    statuses = []
    for _ in range(5):
        challenge = (await _login(client, env["email"], env["password"])).json()["challenge_id"]
        r = await client.post(
            "/api/auth/2fa/verify", json={"challenge_id": challenge, "code": bad}
        )
        statuses.append(r.status_code)

    assert statuses[0] == 401, statuses
    assert statuses[-1] == 429, statuses

    challenge = (await _login(client, env["email"], env["password"])).json()["challenge_id"]
    blocked = await client.post(
        "/api/auth/2fa/verify", json={"challenge_id": challenge, "code": bad}
    )
    assert blocked.status_code == 429, blocked.text


async def test_challenge_is_single_use(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    secret, _ = await _enable(client, token)

    challenge = (await _login(client, env["email"], env["password"])).json()["challenge_id"]
    ok = await client.post(
        "/api/auth/2fa/verify", json={"challenge_id": challenge, "code": _totp(secret)}
    )
    assert ok.status_code == 200, ok.text
    again = await client.post(
        "/api/auth/2fa/verify", json={"challenge_id": challenge, "code": _totp(secret)}
    )
    assert again.status_code == 401, again.text


async def test_unknown_challenge_401(env):
    r = await env["client"].post(
        "/api/auth/2fa/verify", json={"challenge_id": "nope-nope-nope", "code": "123456"}
    )
    assert r.status_code == 401, r.text


# --------------------------------------------------------------- recovery

async def test_recovery_code_single_use(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    _, codes = await _enable(client, token)

    challenge = (await _login(client, env["email"], env["password"])).json()["challenge_id"]
    first = await client.post(
        "/api/auth/2fa/recovery", json={"challenge_id": challenge, "code": codes[0]}
    )
    assert first.status_code == 200, first.text
    assert first.json().get("token")

    challenge2 = (await _login(client, env["email"], env["password"])).json()["challenge_id"]
    second = await client.post(
        "/api/auth/2fa/recovery", json={"challenge_id": challenge2, "code": codes[0]}
    )
    assert second.status_code == 401, second.text


async def test_unknown_recovery_code_401(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    await _enable(client, token)
    challenge = (await _login(client, env["email"], env["password"])).json()["challenge_id"]
    r = await client.post(
        "/api/auth/2fa/recovery", json={"challenge_id": challenge, "code": "deadbeef"}
    )
    assert r.status_code == 401, r.text


# ------------------------------------------------------------- regression

async def test_admin_without_2fa_logs_in_as_before(env):
    client = env["client"]
    r = await _login(client, env["email"], env["password"])
    assert r.status_code == 200, r.text
    assert r.json().get("token"), "обычный вход обязан работать"


async def test_disable_returns_to_password_only(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    secret, _ = await _enable(client, token)

    off = await client.post(
        "/api/auth/2fa/disable",
        json={"code": _totp(secret)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert off.status_code == 200, off.text
    r = await _login(client, env["email"], env["password"])
    assert r.status_code == 200
    assert r.json().get("token")


async def test_disable_wrong_code_400(env):
    client = env["client"]
    token = await _admin_token(client, env["email"], env["password"])
    secret, _ = await _enable(client, token)
    r = await client.post(
        "/api/auth/2fa/disable",
        json={"code": _bad_code(secret)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400, r.text
