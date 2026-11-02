"""Integration tests for the per-account brute-force lockout on POST /api/auth/login.

Live local PostgreSQL, same pattern as tests/api/test_admin_gaps2.py: a pool is
created per test and swapped into ``src.db.pool`` so the asyncpg pool is bound to
the test's event loop. All tests are skipped (pytest.mark.skipif) when the DB is
unreachable at import time.

Credentials live only in module constants; nothing prints or asserts password
values. Password hashes are generated per-row with pgcrypto
``crypt($1, gen_salt('bf', 6))`` ($2a$06$), exactly like tests/api/test_student_view.py.
"""
from __future__ import annotations

import os
import socket
import urllib.parse
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request as StarletteRequest

from src.api_pkg import create_app
from tests.conftest import open_all_gates

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)

EMAIL = "test.bruteforce@compare-competencies.local"
PASSWORD = "Brute_Force_Test_123"
FULL_NAME = "Brute Force Test"
LOCKOUT_THRESHOLD = 5


def _db_reachable() -> bool:
    try:
        parts = urllib.parse.urlsplit(TEST_DB_URL)
        host = parts.hostname or "localhost"
        port = parts.port or 5432
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


DB_REACHABLE = _db_reachable()

pytestmark = pytest.mark.skipif(not DB_REACHABLE, reason="local postgres unavailable")


def _raw_request(peer: str = "127.0.0.1") -> StarletteRequest:
    """Минимальный ASGI-scope для прямого вызова эндпоинта (без middleware)."""
    scope: dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/auth/login",
        "raw_path": b"/api/auth/login",
        "query_string": b"",
        "headers": [(b"user-agent", b"pytest")],
        "client": (peer, 12345),
        "server": ("t", 80),
    }
    return StarletteRequest(scope)


async def _drop_user(pool: asyncpg.Pool, email: str) -> None:
    await pool.execute(
        "DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE email = $1)",
        email,
    )
    await pool.execute("DELETE FROM users WHERE email = $1", email)


async def _make_user(pool: asyncpg.Pool, email: str) -> None:
    """Свежий пользователь со слабым хешем cost 6 (как исторические строки)."""
    await _drop_user(pool, email)
    pw_hash = await pool.fetchval("SELECT crypt($1, gen_salt('bf', 6))", PASSWORD)
    assert str(pw_hash).startswith("$2a$06$"), "unexpected hash scheme"
    await pool.execute(
        "INSERT INTO users (email, password_hash, full_name, role, is_active)"
        " VALUES ($1, $2, $3, $4, true)",
        email, pw_hash, FULL_NAME, "student",
    )


async def _reset_counters(pool: asyncpg.Pool, email: str) -> None:
    """Снять блокировку и обнулить счётчик неудач (до и после каждого теста)."""
    await pool.execute(
        "UPDATE users SET failed_login_count = 0, locked_until = NULL,"
        " last_failed_login_at = NULL WHERE lower(email) = $1",
        email,
    )


@pytest.fixture()
async def env() -> AsyncIterator[dict[str, Any]]:
    """Живой пул + ASGI-клиент; пользователь и счётчики чисты до и после теста."""
    import src.db as dbmod
    from src.api_pkg import limiter as app_limiter
    from src.api_pkg.routers import auth as authmod

    old = dbmod.pool
    try:
        pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2, command_timeout=30)
    except Exception:
        pytest.skip("local postgres unavailable")
    dbmod.pool = pool
    # Счётчики slowapi живут в памяти: сбрасываем, чтобы IP-лимит не протекал.
    app_limiter._storage.reset()
    authmod.limiter._storage.reset()

    app = open_all_gates(create_app())  # type: ignore[no-untyped-call]
    await _make_user(pool, EMAIL)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield {"client": client, "pool": pool}
    finally:
        await _reset_counters(pool, EMAIL)
        await _drop_user(pool, EMAIL)
        dbmod.pool = old
        await pool.close()


async def test_five_wrong_passwords_lock_account(env: dict[str, Any]) -> None:
    client = env["client"]
    statuses = []
    for _ in range(LOCKOUT_THRESHOLD):
        r = await client.post("/api/auth/login", json={"email": EMAIL, "password": "definitely-wrong"})
        statuses.append(r.status_code)
    assert statuses == [401] * LOCKOUT_THRESHOLD

    row = await env["pool"].fetchrow(
        "SELECT failed_login_count, locked_until FROM users WHERE lower(email) = $1", EMAIL
    )
    assert row is not None
    assert row["failed_login_count"] == LOCKOUT_THRESHOLD
    assert row["locked_until"] is not None
    assert row["locked_until"] > datetime.now(UTC)


async def test_correct_password_while_locked_returns_429(env: dict[str, Any]) -> None:
    await env["pool"].execute(
        "UPDATE users SET locked_until = NOW() + INTERVAL '5 minutes' WHERE lower(email) = $1",
        EMAIL,
    )
    r = await env["client"].post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 429
    # Тело именно про блокировку аккаунта, а не про общий IP-лимит slowapi.
    assert "Повторите через" in r.json()["detail"]


async def test_locked_response_carries_retry_after(env: dict[str, Any]) -> None:
    await env["pool"].execute(
        "UPDATE users SET locked_until = NOW() + INTERVAL '5 minutes' WHERE lower(email) = $1",
        EMAIL,
    )
    r = await env["client"].post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 429
    assert "retry-after" in {k.lower() for k in r.headers}


async def test_endpoint_sets_retry_after_before_app_handler(env: dict[str, Any]) -> None:
    """Прямой вызов эндпоинта: он кладёт Retry-After в HTTPException.headers."""
    from src.api_pkg.routers import auth as authmod

    await env["pool"].execute(
        "UPDATE users SET locked_until = NOW() + INTERVAL '5 minutes' WHERE lower(email) = $1",
        EMAIL,
    )
    fn = getattr(authmod.login, "__wrapped__", authmod.login)
    with pytest.raises(HTTPException) as excinfo:
        await fn(authmod.LoginRequest(email=EMAIL, password=PASSWORD), _raw_request())
    assert excinfo.value.status_code == 429
    assert (excinfo.value.headers or {}).get("Retry-After")


async def test_unknown_email_returns_401_without_lockout_row(env: dict[str, Any]) -> None:
    unknown = "nobody.bruteforce@compare-competencies.local"
    r = await env["client"].post(
        "/api/auth/login", json={"email": unknown, "password": "whatever"}
    )
    assert r.status_code == 401
    n = await env["pool"].fetchval("SELECT count(*) FROM users WHERE lower(email) = $1", unknown)
    assert n == 0


async def test_clearing_lock_allows_login_and_resets_counter(env: dict[str, Any]) -> None:
    pool = env["pool"]
    await pool.execute(
        "UPDATE users SET failed_login_count = $1, locked_until = NOW() + INTERVAL '5 minutes',"
        " last_failed_login_at = NOW() WHERE lower(email) = $2",
        LOCKOUT_THRESHOLD, EMAIL,
    )
    r = await env["client"].post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 429

    await _reset_counters(pool, EMAIL)
    r = await env["client"].post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 200, r.text

    row = await pool.fetchrow(
        "SELECT failed_login_count, locked_until FROM users WHERE lower(email) = $1", EMAIL
    )
    assert row is not None
    assert row["failed_login_count"] == 0
    assert row["locked_until"] is None


async def test_successful_login_upgrades_hash_cost_to_12(env: dict[str, Any]) -> None:
    pool = env["pool"]
    cost_q = "SELECT split_part(password_hash, '$', 3)::int FROM users WHERE lower(email) = $1"
    assert await pool.fetchval(cost_q, EMAIL) == 6
    r = await env["client"].post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert r.status_code == 200, r.text
    assert await pool.fetchval(cost_q, EMAIL) == 12
