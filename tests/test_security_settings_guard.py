"""Барьеры хранилища настроек безопасности (app_settings).

Проверяем гонку при создании таблицы и защиту от не-UUID updated_by:
раньше два первых параллельных запроса могли уронить путь записи 500.
"""

from __future__ import annotations

import asyncio
import weakref

import pytest

import src.api_pkg.security_settings as ss


class _FakePool:
    def __init__(self, *, fail_first: bool = False, regclass: bool = True):
        self.calls = 0
        self.insert_args: list[tuple] = []
        self.fail_first = fail_first
        self.regclass = regclass

    async def execute(self, _sql: str, *args):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            # Так выглядит проигравший гонку CREATE TABLE: ошибка каталога pg_type.
            raise RuntimeError("duplicate key value violates unique constraint pg_type_typname_nsp_index")
        if args:
            self.insert_args.append(args)

    async def fetchval(self, _sql: str, *args):
        return self.regclass


@pytest.fixture(autouse=True)
def _reset_state():
    ss._table_ready = False
    ss._ensure_locks = weakref.WeakKeyDictionary()
    yield
    ss._table_ready = False
    ss._ensure_locks = weakref.WeakKeyDictionary()


async def test_ensure_table_created_once_under_concurrency():
    pool = _FakePool()
    await asyncio.gather(ss._ensure_table(pool), ss._ensure_table(pool))  # type: ignore[arg-type]
    assert pool.calls == 1, "CREATE TABLE выполнился дважды - замок не работает"
    assert ss._table_ready is True


async def test_ensure_table_tolerates_catalog_race():
    """Проигравший гонку не падает, если таблица реально существует."""
    pool = _FakePool(fail_first=True, regclass=True)
    await ss._ensure_table(pool)  # type: ignore[arg-type]
    assert ss._table_ready is True


async def test_ensure_table_reraises_when_table_really_absent():
    pool = _FakePool(fail_first=True, regclass=False)
    with pytest.raises(RuntimeError):
        await ss._ensure_table(pool)  # type: ignore[arg-type]
    assert ss._table_ready is False


async def test_set_two_factor_drops_non_uuid_actor():
    """Не-UUID updated_by превращается в NULL, а не в 500 DataError."""
    pool = _FakePool()
    await ss.set_two_factor_enabled(pool, True, "not-a-uuid")  # type: ignore[arg-type]
    assert pool.insert_args, "INSERT не выполнен"
    assert pool.insert_args[-1][2] is None

    await ss.set_two_factor_enabled(pool, False, "00000000-0000-0000-0000-000000000042")  # type: ignore[arg-type]
    assert pool.insert_args[-1][2] == "00000000-0000-0000-0000-000000000042"


async def test_get_two_factor_enabled_is_fail_open_without_pool():
    assert await ss.get_two_factor_enabled(None) is False
