"""Тесты переключателя 2FA: по умолчанию выключено, включается кнопкой.

Главный контракт: пока флаг выключен, администратор входит по паролю, даже
если у него в БД остался подтверждённый TOTP. Раньше 2FA была включена всегда,
и без UI администратор вообще не мог войти.
"""

from __future__ import annotations

import contextlib

import pytest

from src.api_pkg import twofa_policy


class FakePool:
    """Минимальная замена asyncpg-пула для таблицы настроек."""

    def __init__(self, store: dict[str, str] | None = None):
        self.store = store if store is not None else {}
        self.queries: list[str] = []

    async def execute(self, sql: str, *args):
        self.queries.append(sql)
        if "INSERT INTO app_settings" in sql and len(args) >= 2:
            self.store[args[0]] = args[1]
        return "OK"

    async def fetchrow(self, sql: str, *args):
        self.queries.append(sql)
        if "FROM app_settings" in sql and args:
            value = self.store.get(args[0])
            return {"value": value} if value is not None else None
        return None


class BrokenPool:
    """БД недоступна / нет таблицы."""

    async def execute(self, *a, **k):
        raise RuntimeError("relation app_settings does not exist")

    async def fetchrow(self, *a, **k):
        raise RuntimeError("relation app_settings does not exist")


@pytest.mark.asyncio
async def test_default_is_disabled_when_no_row():
    pool = FakePool()
    assert await twofa_policy.is_twofa_enabled(pool) is False


@pytest.mark.asyncio
async def test_enable_then_disable_roundtrip():
    pool = FakePool()
    await twofa_policy.set_twofa_enabled(pool, True, actor="t@t")
    assert await twofa_policy.is_twofa_enabled(pool) is True

    await twofa_policy.set_twofa_enabled(pool, False, actor="t@t")
    assert await twofa_policy.is_twofa_enabled(pool) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
async def test_truthy_values(value):
    pool = FakePool({twofa_policy.TWOFA_ADMIN_KEY: value})
    assert await twofa_policy.is_twofa_enabled(pool) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
async def test_falsy_values(value):
    pool = FakePool({twofa_policy.TWOFA_ADMIN_KEY: value})
    assert await twofa_policy.is_twofa_enabled(pool) is False


@pytest.mark.asyncio
async def test_unreadable_database_fails_open():
    """Недоступная БД не должна запирать администратора снаружи."""
    assert await twofa_policy.is_twofa_enabled(BrokenPool()) is False


@pytest.mark.asyncio
async def test_none_pool_is_disabled():
    assert await twofa_policy.is_twofa_enabled(None) is False


@pytest.mark.asyncio
async def test_should_require_only_for_admin_role():
    pool = FakePool()
    await twofa_policy.set_twofa_enabled(pool, True, actor="t")

    assert await twofa_policy.should_require_totp(pool, "admin") is True
    assert await twofa_policy.should_require_totp(pool, "teacher") is False
    assert await twofa_policy.should_require_totp(pool, "student") is False
    assert await twofa_policy.should_require_totp(pool, None) is False


@pytest.mark.asyncio
async def test_disabled_flag_never_requires_totp():
    pool = FakePool()
    await twofa_policy.set_twofa_enabled(pool, False, actor="t")
    assert await twofa_policy.should_require_totp(pool, "admin") is False


def test_set_twofa_creates_table_before_writing():
    """Страховка на случай запуска до миграции."""
    import asyncio

    pool = FakePool()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        twofa_policy.set_twofa_enabled(pool, True, actor="t")
    )
    assert any("CREATE TABLE IF NOT EXISTS app_settings" in q for q in pool.queries)
