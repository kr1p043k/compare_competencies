"""Эндпоинты переключателя 2FA: по умолчанию выключено, включается кнопкой.

Проверяем контракт, который видит фронт и от которого зависит вход
администратора: состояние флага, права доступа, аудит переключения и —
главное — что при выключенном флаге вход по паролю работает, даже если у
администратора остался подтверждённый TOTP.
"""

from __future__ import annotations

import contextlib
import uuid

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from src.api_pkg import twofa_policy
from src.api_pkg.routers import auth as auth_mod


def _resolve_test_db_url() -> str:
    """DSN для интеграционных тестов: только из окружения или .env.

    Литерал с паролем в тесте - это утёкший секрет в истории репозитория,
    поэтому здесь его нет: без переменной тесты просто пропускаются.
    """
    import os
    from pathlib import Path

    url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    if url:
        return _strip_driver(url)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(("TEST_DATABASE_URL=", "DATABASE_URL=")):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return _strip_driver(value)
    return ""


def _strip_driver(url: str) -> str:
    """asyncpg не понимает SQLAlchemy-драйвер в схеме URL."""
    for suffix in ("+asyncpg", "+psycopg2", "+psycopg"):
        url = url.replace(suffix, "")
    return url


TEST_DB_URL = _resolve_test_db_url()
PASSWORD = "Toggle-Test-Password-123"


def _db_reachable() -> bool:
    import asyncio

    async def _probe() -> bool:
        try:
            conn = await asyncpg.connect(TEST_DB_URL, timeout=5)
        except Exception:
            return False
        await conn.close()
        return True

    try:
        return asyncio.run(_probe())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not TEST_DB_URL or not _db_reachable(), reason="local postgres unavailable"
)


async def _make_user(pool, email: str, role: str) -> str:
    """Создать активного пользователя с известным паролем, вернуть id."""
    await pool.execute("DELETE FROM users WHERE lower(email) = $1", email)
    row = await pool.fetchrow(
        """INSERT INTO users (email, password_hash, role, full_name, is_active)
           VALUES ($1, crypt($2, gen_salt('bf', 6)), $3, $4, true)
           RETURNING id""",
        email, PASSWORD, role, email.split("@")[0],
    )
    return str(row["id"])


async def _confirm_totp(pool, user_id: str, secret: str) -> None:
    await pool.execute(
        """INSERT INTO admin_totp (user_id, secret, confirmed)
           VALUES ($1, $2, true)
           ON CONFLICT (user_id) DO UPDATE SET confirmed = true""",
        user_id, secret,
    )


def _code(secret: str) -> str:
    import base64
    import hashlib
    import hmac
    import struct
    import time

    counter = int(time.time()) // 30
    digest = hmac.new(
        base64.b32decode(secret + "=" * (-len(secret) % 8)),
        struct.pack(">Q", counter),
        hashlib.sha1,
    ).digest()
    offset = digest[-1] & 0x0F
    binary = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return f"{binary % 1_000_000:06d}"


SECRET = "JBSWY3DPEHPK3PXP"


@pytest.fixture()
async def env():
    import src.db as dbmod
    from src.api_pkg import create_app
    from src.api_pkg import limiter as pkg_limiter

    old_pool = dbmod.pool
    suffix = uuid.uuid4().hex[:8]
    admin_email = f"toggle-admin-{suffix}@example.test"
    teacher_email = f"toggle-teacher-{suffix}@example.test"

    try:
        pool = await asyncpg.create_pool(TEST_DB_URL, min_size=1, max_size=2, command_timeout=30)
    except Exception:  # pragma: no cover
        pytest.skip("local postgres unavailable")

    dbmod.pool = pool
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()
    with contextlib.suppress(Exception):
        pkg_limiter._storage.reset()
    auth_mod._totp_challenges.clear()

    await auth_mod._ensure_totp_table(pool)
    await twofa_policy.ensure_settings_table(pool)
    # Начинаем с выключенного состояния - это значение по умолчанию.
    await twofa_policy.set_twofa_enabled(pool, False, actor="test-setup")

    admin_id = await _make_user(pool, admin_email, "admin")
    await _make_user(pool, teacher_email, "teacher")

    app = create_app()
    app.state.limiter = auth_mod.limiter
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield {
                "client": client,
                "pool": pool,
                "admin_email": admin_email,
                "teacher_email": teacher_email,
                "admin_id": admin_id,
            }
    finally:
        with contextlib.suppress(Exception):
            await twofa_policy.set_twofa_enabled(pool, False, actor="test-teardown")
        with contextlib.suppress(Exception):
            await pool.execute("DELETE FROM admin_totp WHERE user_id = $1", admin_id)
        with contextlib.suppress(Exception):
            await pool.execute(
                "DELETE FROM users WHERE lower(email) = ANY($1::text[])",
                [admin_email, teacher_email],
            )
        dbmod.pool = old_pool
        await pool.close()


async def _login(client, email: str, password: str = PASSWORD):
    return await client.post("/api/auth/login", json={"email": email, "password": password})


async def _token(client, email: str) -> str:
    r = await _login(client, email)
    assert r.status_code == 200, r.text
    return r.json()["token"]


class TestDefaultState:
    async def test_flag_is_off_by_default(self, env):
        client, pool = env["client"], env["pool"]
        assert await twofa_policy.is_twofa_enabled(pool) is False

        token = await _token(client, env["admin_email"])
        r = await client.get(
            "/api/admin/security/2fa", headers={"Authorization": f"Bearer {token}"}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["enabled"] is False
        assert body["role"] == "admin"
        assert "total_admins" in body and "enrolled_admins" in body


class TestToggle:
    async def test_toggle_roundtrip_through_api(self, env):
        client = env["client"]
        token = await _token(client, env["admin_email"])
        headers = {"Authorization": f"Bearer {token}"}

        r = await client.put("/api/admin/security/2fa", json={"enabled": True}, headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["enabled"] is True

        r = await client.get("/api/admin/security/2fa", headers=headers)
        assert r.json()["enabled"] is True

        r = await client.put("/api/admin/security/2fa", json={"enabled": False}, headers=headers)
        assert r.status_code == 200, r.text
        r = await client.get("/api/admin/security/2fa", headers=headers)
        assert r.json()["enabled"] is False

    async def test_teacher_cannot_read_or_flip(self, env):
        client = env["client"]
        token = await _token(client, env["teacher_email"])
        headers = {"Authorization": f"Bearer {token}"}

        read = await client.get("/api/admin/security/2fa", headers=headers)
        assert read.status_code == 403, read.text
        flip = await client.put(
            "/api/admin/security/2fa", json={"enabled": True}, headers=headers
        )
        assert flip.status_code == 403, flip.text

    async def test_anonymous_is_rejected(self, env):
        client = env["client"]
        read = await client.get("/api/admin/security/2fa")
        assert read.status_code in (401, 403)
        flip = await client.put("/api/admin/security/2fa", json={"enabled": True})
        assert flip.status_code in (401, 403)


class TestEnrolmentCounts:
    async def test_counts_reflect_admin_totp_rows(self, env):
        client, pool = env["client"], env["pool"]
        await _confirm_totp(pool, env["admin_id"], SECRET)

        token = await _token(client, env["admin_email"])
        headers = {"Authorization": f"Bearer {token}"}
        body = (await client.get("/api/admin/security/2fa", headers=headers)).json()

        assert body["enrolled_admins"] >= 1
        assert body["total_admins"] >= 1
        assert body["enrolled_admins"] <= body["total_admins"]

    async def test_admins_list_marks_enrolled(self, env):
        client, pool = env["client"], env["pool"]
        await _confirm_totp(pool, env["admin_id"], SECRET)

        token = await _token(client, env["admin_email"])
        r = await client.get(
            "/api/admin/security/2fa/admins",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        admins = {a["email"]: a for a in r.json()["admins"]}
        assert env["admin_email"] in admins
        assert admins[env["admin_email"]]["enrolled"] is True
        assert all(a["role"] == "admin" for a in admins.values())


class TestLoginContract:
    """Главный контракт: флаг решает, спрашивать ли код."""

    async def test_flag_off_lets_admin_in_with_password_even_with_totp(self, env):
        client, pool = env["client"], env["pool"]
        await _confirm_totp(pool, env["admin_id"], SECRET)
        await twofa_policy.set_twofa_enabled(pool, False, actor="test")

        r = await _login(client, env["admin_email"])
        assert r.status_code == 200, r.text
        body = r.json()
        assert "token" in body, "при выключенном флаге токен выдаётся сразу"
        assert body.get("status") != "totp_required"

    async def test_flag_on_requires_code_and_issues_no_token(self, env):
        client, pool = env["client"], env["pool"]
        await _confirm_totp(pool, env["admin_id"], SECRET)
        await twofa_policy.set_twofa_enabled(pool, True, actor="test")

        r = await _login(client, env["admin_email"])
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("status") == "totp_required"
        assert "token" not in body
        assert body.get("challenge_id")

    async def test_flag_on_does_not_affect_teacher_login(self, env):
        client, pool = env["client"], env["pool"]
        await twofa_policy.set_twofa_enabled(pool, True, actor="test")

        r = await _login(client, env["teacher_email"])
        assert r.status_code == 200, r.text
        assert "token" in r.json(), "флаг применяется только к роли admin"


class TestStatusForUi:
    async def test_status_reports_flag_and_own_enrolment(self, env):
        client, pool = env["client"], env["pool"]
        token = await _token(client, env["admin_email"])
        headers = {"Authorization": f"Bearer {token}"}

        body = (await client.get("/api/auth/2fa/status", headers=headers)).json()
        assert body["enabled_globally"] is False
        assert body["confirmed"] is False

        await _confirm_totp(pool, env["admin_id"], SECRET)
        body = (await client.get("/api/auth/2fa/status", headers=headers)).json()
        assert body["configured"] is True
        assert body["confirmed"] is True


class TestPolicyUnit:
    def test_applies_only_to_admin(self):
        assert twofa_policy.applies_to_role("admin") is True
        assert twofa_policy.applies_to_role("ADMIN") is True
        for role in ("teacher", "rop", "student", "", None):
            assert twofa_policy.applies_to_role(role) is False
