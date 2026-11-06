"""SSO-вход и TOTP: admin через /auth/sso не должен обходить второй фактор.

Живой локальный PostgreSQL (как в tests/api/test_2fa.py) + подмена внешнего
academic-api: /verify-token всегда отвечает 200, чтобы не зависеть от сети.
Пул подменяет ``src.db.pool`` на время теста.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import socket
import urllib.parse
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

import src.api_pkg.routers.academic as academic_mod
import src.api_pkg.routers.auth as auth_mod
from src import config
from src.api_pkg import create_app

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies",
)

USERNAME = "sso-totp-admin"
EMAIL = f"{USERNAME}@{config.ACADEMIC_EMAIL_DOMAIN}"
# Пароль нужен только чтобы включить TOTP через /auth/2fa/* до проверки SSO.
PASSWORD = "Sso_Totp_Test_Password_123"


def _db_reachable() -> bool:
    try:
        parts = urllib.parse.urlsplit(TEST_DB_URL)
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason="local postgres unavailable")


def _totp_secret() -> str:
    """Случайный base32-секрет для прямой записи в admin_totp."""
    return auth_mod._generate_totp_secret()


def _totp(secret: str) -> str:
    return auth_mod._totp_now(secret)


def _jwt(username: str) -> str:
    """JWT-подобный токен хаба с username в payload (проверки подписи нет)."""
    payload = base64.urlsafe_b64encode(
        json.dumps({"username": username}).encode()
    ).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{payload}.c2ln"


class _FakeResponse:
    status_code = 200

    def json(self) -> dict:
        return {}


class _FakeAsyncClient:
    """Подменяет httpx.AsyncClient: /verify-token всегда 200."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def post(self, url: str, *, json: Any = None, headers: Any = None) -> _FakeResponse:
        return _FakeResponse()


@pytest.fixture()
def fake_sso(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(academic_mod.httpx, "AsyncClient", _FakeAsyncClient)


async def _drop_user(pool: asyncpg.Pool) -> None:
    row = await pool.fetchrow("SELECT id FROM users WHERE lower(email) = $1", EMAIL)
    if row is not None:
        await pool.execute("DELETE FROM admin_totp WHERE user_id = $1", row["id"])
        await pool.execute("DELETE FROM sessions WHERE user_id = $1", row["id"])
    await pool.execute("DELETE FROM users WHERE lower(email) = $1", EMAIL)


async def _make_user(pool: asyncpg.Pool, role: str = "admin") -> None:
    await _drop_user(pool)
    await pool.execute(
        """INSERT INTO users (email, password_hash, role, full_name, is_active)
           VALUES ($1, crypt($2, gen_salt('bf', 10)), $3, 'SSO User', true)""",
        EMAIL, PASSWORD, role,
    )


@pytest.fixture()
async def env() -> AsyncIterator[dict[str, Any]]:
    import src.db as dbmod

    old = dbmod.pool
    try:
        pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2, command_timeout=30)
    except Exception:  # pragma: no cover
        pytest.skip("local postgres unavailable")
    dbmod.pool = pool
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()
    with contextlib.suppress(Exception):
        academic_mod.limiter.reset()
    await auth_mod._ensure_totp_table(pool)
    await _make_user(pool)
    # 2FA включена: сценарии ниже проверяют именно включённый рубильник.
    from src.api_pkg.security_settings import set_two_factor_enabled
    await set_two_factor_enabled(pool, True)

    app = create_app()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield {"client": client, "pool": pool}
    finally:
        with contextlib.suppress(Exception):
            from src.api_pkg.security_settings import set_two_factor_enabled as _set

            await _set(pool, False)
        await _drop_user(pool)
        dbmod.pool = old
        await pool.close()


async def _login(client: AsyncClient):
    return await client.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})


async def _admin_token(client: AsyncClient) -> str:
    r = await _login(client)
    assert r.status_code == 200, r.text
    return r.json()["token"]


async def _enable(client: AsyncClient, token: str) -> tuple[str, list[str]]:
    setup = await client.post(
        "/api/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"}
    )
    assert setup.status_code == 200, setup.text
    secret = setup.json()["secret"]
    confirm = await client.post(
        "/api/auth/2fa/confirm",
        json={"code": _totp(secret)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert confirm.status_code == 200, confirm.text
    return secret, confirm.json()["recovery_codes"]


def _sso(client: AsyncClient, username: str = USERNAME):
    return client.post("/api/auth/sso", json={"token": _jwt(username)})


async def test_sso_admin_with_totp_returns_challenge(env, fake_sso):
    """Админ с подтверждённым TOTP: SSO выдаёт challenge, а не токен."""
    client = env["client"]
    pool = env["pool"]
    await pool.execute(
        "INSERT INTO admin_totp (user_id, secret, confirmed) "
        "SELECT id, $1, true FROM users WHERE lower(email) = lower($2) "
        "ON CONFLICT (user_id) DO UPDATE SET confirmed = true",
        _totp_secret(), EMAIL,
    )

    r = await _sso(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "totp_required", body
    assert "token" not in body, "SSO не должен выдавать токен админу мимо TOTP"
    assert body["challenge_id"]


async def test_sso_admin_without_totp_requires_enrolment(env, fake_sso):
    """Админ без TOTP при включённом рубильнике получает пропуск, не токен.

    Токен здесь означал бы обход второго фактора: у кого есть hub-JWT,
    тот вошёл бы без кода."""
    client = env["client"]
    r = await _sso(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "totp_setup_required", body
    assert "token" not in body
    assert body.get("enroll_id")


async def test_sso_non_admin_issues_token(env, fake_sso):
    """Не-админ: SSO поведение не меняется - токен выдаётся."""
    pool = env["pool"]
    await pool.execute("UPDATE users SET role = 'teacher' WHERE lower(email) = $1", EMAIL)
    client = env["client"]
    r = await _sso(client)
    assert r.status_code == 200, r.text
    assert r.json().get("token")
