"""TOTP 2FA для админа: настройка, вход, recovery, отключение.

Живая локальная PostgreSQL, тот же приём, что в tests/api/test_auth_bruteforce.py:
пул создаётся на тест и подменяет ``src.db.pool`` (иначе asyncpg-пул привязан к
чужому event loop). Пользователь и admin_totp чистятся до и после теста.
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

EMAIL = "totp-admin@example.test"
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


async def _drop_user(pool: asyncpg.Pool) -> None:
    row = await pool.fetchrow("SELECT id FROM users WHERE lower(email) = $1", EMAIL)
    if row is not None:
        await pool.execute("DELETE FROM admin_totp WHERE user_id = $1", row["id"])
        await pool.execute("DELETE FROM sessions WHERE user_id = $1", row["id"])
    await pool.execute("DELETE FROM users WHERE lower(email) = $1", EMAIL)


async def _make_admin(pool: asyncpg.Pool) -> None:
    await _drop_user(pool)
    await pool.execute(
        """INSERT INTO users (email, password_hash, role, full_name, is_active)
           VALUES ($1, crypt($2, gen_salt('bf', 10)), 'admin', 'TOTP Admin', true)""",
        EMAIL, PASSWORD,
    )


@pytest.fixture()
async def env(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    import src.db as dbmod

    # Эти тесты проверяют сам TOTP-поток, поэтому мастер-переключатель
    # считаем включённым. Поведение при выключенном флаге - в test_2fa_toggle.py.
    async def _enabled(_pool: Any) -> bool:
        return True

    monkeypatch.setattr(auth_mod, "get_two_factor_enabled", _enabled)

    old = dbmod.pool
    real_two_factor_enabled = auth_mod.two_factor_enabled
    try:
        pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2, command_timeout=30)
    except Exception:  # pragma: no cover
        pytest.skip("local postgres unavailable")
    dbmod.pool = pool
    _ENROLLED.clear()
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()
    await auth_mod._ensure_totp_table(pool)
    await _make_admin(pool)
    # 2FA теперь за глобальным рубильником (по умолчанию выключен): эти тесты
    # проверяют поведение включённой 2FA, поэтому флаг включаем точечно.
    auth_mod.two_factor_enabled = lambda: True

    app = create_app()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield {"client": client, "pool": pool}
    finally:
        await _drop_user(pool)
        dbmod.pool = old
        auth_mod.two_factor_enabled = real_two_factor_enabled
        await pool.close()


async def _login(client: AsyncClient):
    return await client.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})


#: Секрет и recovery-коды, полученные при настройке в текущем тесте: при
#: включённом 2FA логин без TOTP не выдаёт токен, поэтому настройка идёт первой.
_ENROLLED: dict[str, Any] = {}


async def _admin_token(client: AsyncClient) -> str:
    """Сессия админа при включённом мастер-переключателе 2FA.

    Если админ ещё не подключил TOTP, логин возвращает пропуск на настройку
    (status=totp_setup_required): проходим её тем же путём, что пользователь.
    """
    r = await _login(client)
    assert r.status_code == 200, r.text
    data = r.json()
    if data.get("token"):
        return data["token"]

    assert data.get("status") == "totp_setup_required", data
    enroll = data["enroll_id"]

    setup = await client.post("/api/auth/2fa/setup", json={"enroll_id": enroll})
    assert setup.status_code == 200, setup.text
    secret = setup.json()["secret"]

    confirm = await client.post(
        "/api/auth/2fa/confirm", json={"code": _totp(secret), "enroll_id": enroll}
    )
    assert confirm.status_code == 200, confirm.text
    token = confirm.json().get("token")
    assert token, "подтверждение настройки должно сразу выдавать сессию"
    _ENROLLED["secret"] = secret
    _ENROLLED["codes"] = confirm.json().get("recovery_codes", [])
    return token


async def _enable(client: AsyncClient, token: str) -> tuple[str, list[str]]:
    if _ENROLLED.get("secret"):
        return _ENROLLED["secret"], _ENROLLED.get("codes", [])
    setup = await client.post("/api/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
    assert setup.status_code == 200, setup.text
    secret = setup.json()["secret"]
    confirm = await client.post(
        "/api/auth/2fa/confirm",
        json={"code": _totp(secret)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert confirm.status_code == 200, confirm.text
    _ENROLLED["secret"] = secret
    _ENROLLED["codes"] = confirm.json().get("recovery_codes", [])
    return secret, confirm.json()["recovery_codes"]


# ------------------------------------------------------------------ setup

async def test_setup_returns_uri_and_secret(env):
    """Первичная настройка идёт по пропуску из логина (сессии ещё нет)."""
    client = env["client"]
    data = (await _login(client)).json()
    assert data.get("status") == "totp_setup_required", data
    r = await client.post("/api/auth/2fa/setup", json={"enroll_id": data["enroll_id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["otpauth_uri"].startswith("otpauth://totp/")
    assert "secret=" in body["otpauth_uri"]
    assert len(body["secret"]) >= 16


async def test_setup_conflicts_when_already_confirmed(env):
    client = env["client"]
    token = await _admin_token(client)
    await _enable(client, token)
    r = await client.post("/api/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 409, r.text


async def test_setup_requires_auth(env):
    r = await env["client"].post("/api/auth/2fa/setup")
    assert r.status_code == 401, r.text


async def test_setup_forbidden_for_teacher(env):
    """Роль из БД: не-админ получает 403."""
    pool = env["pool"]
    await pool.execute("UPDATE users SET role = 'teacher' WHERE lower(email) = $1", EMAIL)
    client = env["client"]
    token = await _admin_token(client)
    r = await client.post("/api/auth/2fa/setup", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403, r.text


# ---------------------------------------------------------------- confirm

async def test_confirm_wrong_code_400(env):
    """Неверный код при подтверждении настройки не принимается."""
    client = env["client"]
    data = (await _login(client)).json()
    assert data.get("status") == "totp_setup_required", data
    enroll = data["enroll_id"]
    setup = await client.post("/api/auth/2fa/setup", json={"enroll_id": enroll})
    secret = setup.json()["secret"]
    r = await client.post(
        "/api/auth/2fa/confirm",
        json={"code": _bad_code(secret), "enroll_id": enroll},
    )
    assert r.status_code == 400, r.text


async def test_confirm_returns_ten_recovery_codes(env):
    client = env["client"]
    token = await _admin_token(client)
    _, codes = await _enable(client, token)
    assert len(codes) == 10
    assert len(set(codes)) == 10


# ------------------------------------------------------------ login flow

async def test_login_requires_totp_and_issues_no_token(env):
    client = env["client"]
    token = await _admin_token(client)
    await _enable(client, token)

    r = await _login(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "totp_required"
    assert "token" not in body, "одного пароля недостаточно"
    assert body["challenge_id"]


async def test_verify_issues_token(env):
    client = env["client"]
    token = await _admin_token(client)
    secret, _ = await _enable(client, token)

    challenge = (await _login(client)).json()["challenge_id"]
    r = await client.post(
        "/api/auth/2fa/verify", json={"challenge_id": challenge, "code": _totp(secret)}
    )
    assert r.status_code == 200, r.text
    assert r.json().get("token")
    assert r.json()["role"] == "admin"


async def test_wrong_code_locks_after_five(env):
    client = env["client"]
    token = await _admin_token(client)
    secret, _ = await _enable(client, token)
    bad = _bad_code(secret)

    statuses = []
    for _ in range(5):
        challenge = (await _login(client)).json()["challenge_id"]
        r = await client.post(
            "/api/auth/2fa/verify", json={"challenge_id": challenge, "code": bad}
        )
        statuses.append(r.status_code)

    assert statuses[0] == 401, statuses
    assert statuses[-1] == 429, statuses

    challenge = (await _login(client)).json()["challenge_id"]
    blocked = await client.post(
        "/api/auth/2fa/verify", json={"challenge_id": challenge, "code": bad}
    )
    assert blocked.status_code == 429, blocked.text


async def test_challenge_is_single_use(env):
    client = env["client"]
    token = await _admin_token(client)
    secret, _ = await _enable(client, token)

    challenge = (await _login(client)).json()["challenge_id"]
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
    token = await _admin_token(client)
    _, codes = await _enable(client, token)

    challenge = (await _login(client)).json()["challenge_id"]
    first = await client.post(
        "/api/auth/2fa/recovery", json={"challenge_id": challenge, "code": codes[0]}
    )
    assert first.status_code == 200, first.text
    assert first.json().get("token")

    challenge2 = (await _login(client)).json()["challenge_id"]
    second = await client.post(
        "/api/auth/2fa/recovery", json={"challenge_id": challenge2, "code": codes[0]}
    )
    assert second.status_code == 401, second.text


async def test_unknown_recovery_code_401(env):
    client = env["client"]
    token = await _admin_token(client)
    await _enable(client, token)
    challenge = (await _login(client)).json()["challenge_id"]
    r = await client.post(
        "/api/auth/2fa/recovery", json={"challenge_id": challenge, "code": "deadbeef"}
    )
    assert r.status_code == 401, r.text


# ------------------------------------------------------------- regression

async def test_admin_without_2fa_must_enroll_while_switch_on(env):
    """Мастер включён: админ без TOTP получает пропуск на настройку, а не сессию."""
    client = env["client"]
    r = await _login(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "totp_setup_required", body
    assert body.get("enroll_id")
    assert not body.get("token"), "без второго фактора токен не выдаём"


async def test_login_is_password_only_when_switch_off(env, monkeypatch):
    """Мастер выключен: вход по паролю, как раньше (даже для подключённых)."""
    client = env["client"]
    await _admin_token(client)  # подключаем TOTP
    _ENROLLED.clear()

    async def _disabled(_pool):
        return False

    monkeypatch.setattr(auth_mod, "get_two_factor_enabled", _disabled)
    r = await _login(client)
    assert r.status_code == 200, r.text
    assert r.json().get("token"), "при выключенном тумблере код не спрашиваем"


async def test_disable_returns_to_enrollment_while_switch_on(env):
    client = env["client"]
    token = await _admin_token(client)
    secret, _ = await _enable(client, token)

    off = await client.post(
        "/api/auth/2fa/disable",
        json={"code": _totp(secret)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert off.status_code == 200, off.text
    # Мастер-переключатель всё ещё включён, поэтому настройку потребуют заново:
    # персональное отключение не обходит требование администратора.
    r = await _login(client)
    assert r.status_code == 200
    assert r.json().get("status") == "totp_setup_required", r.text


async def test_disable_wrong_code_400(env):
    client = env["client"]
    token = await _admin_token(client)
    secret, _ = await _enable(client, token)
    r = await client.post(
        "/api/auth/2fa/disable",
        json={"code": _bad_code(secret)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400, r.text
