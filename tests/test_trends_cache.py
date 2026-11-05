"""Предрасчёт /competency-trends: in-memory кэш + БД-кэш trend_precomputed.

Эндпоинт считает тренды по всем компетенциям на каждый промах (0.9-1.2 c,
O(навыки x месяцы)), поэтому результат кэшируется в два слоя:
  * _CT_CACHE (in-memory, TTL 600 c) - как раньше;
  * trend_precomputed (БД) - переживает рестарт процесса.

Тесты БЕЗ реальной БД: asyncpg-пул (src.db.get_pool) и SQLAlchemy-сессия
(src.database.async_session_factory) подменяются заглушками.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.api_pkg.routers import trends as tr
from src.cache_bus import invalidate

# ---------------------------------------------------------------------------
# Заглушки данных: 2 снимка (текущий и предыдущий) и 1 компетенция с навыком.
# ---------------------------------------------------------------------------


def _snap(date: datetime, freq: dict) -> SimpleNamespace:
    return SimpleNamespace(snapshot_date=date, skill_freq=freq)


SNAPS = [
    _snap(datetime(2026, 11, 8), {"python": 100}),
    _snap(datetime(2026, 11, 1), {"python": 50}),
]

COMP_ROWS = [
    SimpleNamespace(id="c1", code="DEV", name="Разработка", skill_name="python"),
]


class _SnapResult:
    def __init__(self, snaps):
        self._snaps = snaps

    def scalars(self):
        return self

    def all(self):
        return self._snaps


class _CompResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    """SQLAlchemy-сессия-заглушка: 1-й execute - снимки, 2-й - компетенции."""

    def __init__(self, snaps, comp_rows):
        self._snaps = snaps
        self._comp_rows = comp_rows
        self.executes = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, stmt):
        self.executes += 1
        if self.executes == 1:
            return _SnapResult(self._snaps)
        return _CompResult(self._comp_rows)


class _FakePool:
    """asyncpg-пул-заглушка: SQL по trend_precomputed идёт сюда."""

    def __init__(self, *, fetchrow_error=None, execute_error=None, row=None):
        self.execute_calls: list[tuple[str, tuple]] = []
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.fetchrow_error = fetchrow_error
        self.execute_error = execute_error
        self.row = row

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))
        if self.execute_error:
            raise self.execute_error
        return None

    async def fetchrow(self, sql, *args):
        self.fetchrow_calls.append((sql, args))
        if self.fetchrow_error:
            raise self.fetchrow_error
        return self.row


class _Harness:
    """Подменяет get_pool и async_session_factory, считает создания сессии."""

    def __init__(self, monkeypatch, pool: _FakePool):
        self.pool = pool
        self.session_creations = 0
        monkeypatch.setattr(tr, "get_pool", lambda: pool)

        def factory():
            self.session_creations += 1
            return _FakeSession(SNAPS, COMP_ROWS)

        self.factory = factory
        monkeypatch.setattr(tr, "async_session_factory", factory)

    def call(self, direction=None, limit=200) -> dict:
        return asyncio.run(
            tr.get_competency_trends(
                request=SimpleNamespace(), direction=direction, limit=limit
            )
        )


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch) -> None:
    """Изоляция глобального кэша и отключение rate-limit обёртки."""
    tr._CT_CACHE.clear()
    tr._sw.cache_clear()
    # обёртка slowapi требует starlette.Request при enabled=True; в тестах
    # вызываем функцию напрямую, поэтому лимитер отключаем.
    monkeypatch.setattr(tr.limiter, "enabled", False, raising=False)
    yield
    tr._CT_CACHE.clear()
    tr._sw.cache_clear()


def _upserts(pool: _FakePool) -> list[tuple[str, tuple]]:
    return [c for c in pool.execute_calls if "INSERT INTO trend_precomputed" in c[0]]


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------


def test_first_call_computes_and_writes_db(monkeypatch):
    pool = _FakePool()
    h = _Harness(monkeypatch, pool)

    payload = h.call()

    assert payload["total"] == 1
    assert payload["trends"][0]["code"] == "DEV"
    assert h.session_creations == 1, "первый вызов обязан посчитать"

    upserts = _upserts(pool)
    assert len(upserts) == 1, "после расчёта результат пишется в БД (UPSERT)"
    assert "ON CONFLICT" in upserts[0][0]
    # ключ и payload переданы как аргументы
    assert upserts[0][1][0] == "|200"
    assert json.loads(upserts[0][1][1])["total"] == 1


def test_second_call_hits_memory_cache(monkeypatch):
    pool = _FakePool()
    h = _Harness(monkeypatch, pool)

    h.call()
    payload = h.call()

    assert payload["total"] == 1
    assert h.session_creations == 1, "второй вызов не должен пересчитывать"
    # fetchrow дёргался только один раз: при чтении БД-кэша на первом промахе;
    # in-memory hit на втором вызове в БД не ходит.
    assert len(pool.fetchrow_calls) == 1


def test_invalidate_resets_and_recomputes(monkeypatch):
    pool = _FakePool()
    h = _Harness(monkeypatch, pool)

    h.call()
    assert h.session_creations == 1

    invalidate("trends")
    assert tr._CT_CACHE == {}

    h.call()
    assert h.session_creations == 2, "после инвалидации кэш пересчитывается"
    assert len(_upserts(pool)) == 2, "новый расчёт снова пишется в БД"


def test_db_cache_survives_restart(monkeypatch):
    """После рестарта (пустой _CT_CACHE) payload подтягивается из БД без расчёта."""
    pool = _FakePool()
    h = _Harness(monkeypatch, pool)

    h.call()
    written = json.loads(_upserts(pool)[0][1][1])

    # имитация рестарта: in-memory кэш пуст, в БД лежит свежий payload
    tr._CT_CACHE.clear()
    pool.row = {"payload": json.dumps(written), "computed_at": datetime.now(UTC)}
    h.session_creations = 0

    payload = h.call()

    assert payload["total"] == 1
    assert h.session_creations == 0, "payload поднят из БД, расчёт не повторялся"
    assert pool.fetchrow_calls, "БД-кэш читался"


def test_missing_table_does_not_break_endpoint(monkeypatch):
    """pool.fetchrow/execute кидают (таблицы нет) - ответ всё равно корректен."""
    err = RuntimeError('relation "trend_precomputed" does not exist')
    pool = _FakePool(fetchrow_error=err, execute_error=err)
    h = _Harness(monkeypatch, pool)

    payload = h.call()

    assert payload["total"] == 1
    assert payload["trends"][0]["code"] == "DEV"
    assert h.session_creations == 1, "промах БД-кэша => честный расчёт"
