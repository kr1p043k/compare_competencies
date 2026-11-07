"""Fail-closed чтение мастер-выключателя 2FA из таблицы app_settings.

Проверяет, что get_two_factor_enabled:
- возвращает False, когда строки нет (по умолчанию выключено);
- читает {"enabled": true} и как dict, и как json-строку (jsonb может вернуться str);
- бросает TwoFactorSettingUnavailable на любой ошибке чтения (state unknown, deny);
- возвращает False на asyncpg.UndefinedTableError (свежий деплой до миграции);
- возвращает False при pool=None (без БД вход всё равно невозможен).
"""
from __future__ import annotations

import asyncpg
import pytest

from src.api_pkg import security_settings
from src.api_pkg.security_settings import TwoFactorSettingUnavailableError


@pytest.fixture(autouse=True)
def reset_table_ready():
    """Каждый тест стартует с False, чтобы _ensure_table выполнил DDL заново."""
    security_settings._table_ready = False
    yield
    security_settings._table_ready = False


class MissingRowPool:
    """Строка с ключом two_factor отсутствует: fetchval возвращает None."""

    async def execute(self, *_a, **_kw):
        return "OK"

    async def fetchval(self, *_a, **_kw):
        return None


class EnabledDictPool:
    """jsonb-колонка отдана как dict (кодек asyncpg)."""

    async def execute(self, *_a, **_kw):
        return "OK"

    async def fetchval(self, *_a, **_kw):
        return {"enabled": True}


class EnabledStrPool:
    """jsonb-колонка отдана как строка (registry без codec) -> json.loads."""

    async def execute(self, *_a, **_kw):
        return "OK"

    async def fetchval(self, *_a, **_kw):
        return '{"enabled": true}'


class RuntimeErrorPool:
    """Любая ошибка чтения, кроме UndefinedTableError, -> deny."""

    async def execute(self, *_a, **_kw):
        return "OK"

    async def fetchval(self, *_a, **_kw):
        raise RuntimeError("connection lost")


class MissingTablePool:
    """Свежий деплой до миграции: CREATE падает с UndefinedTableError."""

    async def execute(self, *_a, **_kw):
        raise asyncpg.UndefinedTableError('relation "app_settings" does not exist')

    async def fetchval(self, *_a, **_kw):
        raise asyncpg.UndefinedTableError('relation "app_settings" does not exist')


@pytest.mark.asyncio
async def test_missing_row_returns_false():
    assert await security_settings.get_two_factor_enabled(MissingRowPool()) is False


@pytest.mark.asyncio
async def test_enabled_dict_returns_true():
    assert await security_settings.get_two_factor_enabled(EnabledDictPool()) is True


@pytest.mark.asyncio
async def test_enabled_json_str_returns_true():
    assert await security_settings.get_two_factor_enabled(EnabledStrPool()) is True


@pytest.mark.asyncio
async def test_read_error_raises_unavailable():
    with pytest.raises(TwoFactorSettingUnavailableError):
        await security_settings.get_two_factor_enabled(RuntimeErrorPool())


@pytest.mark.asyncio
async def test_missing_table_returns_false():
    assert await security_settings.get_two_factor_enabled(MissingTablePool()) is False


@pytest.mark.asyncio
async def test_none_pool_returns_false():
    assert await security_settings.get_two_factor_enabled(None) is False
