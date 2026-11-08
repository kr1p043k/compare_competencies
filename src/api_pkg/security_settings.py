"""Хранилище прикладных настроек в БД (таблица app_settings)."""

from __future__ import annotations

import asyncio
import json
import uuid
import weakref
from dataclasses import dataclass
from enum import Enum

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

TWO_FACTOR_KEY = "security.two_factor"

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by UUID NULL REFERENCES users(id) ON DELETE SET NULL
)
"""

_table_ready = False
#: Замки на создание таблицы, по одному на event loop. Иначе замок привязывается
#: к первому loop (в тестах их несколько) и повторное использование падает
#: RuntimeError "is bound to a different event loop".
_ensure_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _ensure_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _ensure_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _ensure_locks[loop] = lock
    return lock


def _is_uuid(value: str | None) -> bool:
    """updated_by ссылается на users(id) UUID: чужой формат даст DataError."""
    if not value:
        return False
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


async def _ensure_table(pool: asyncpg.Pool) -> None:
    """Страховка на случай деплоя без миграции (app_settings_table.py)."""
    global _table_ready

    if _table_ready:
        return
    async with _ensure_lock():
        if _table_ready:
            return
        try:
            await pool.execute(_TABLE_DDL)
        except Exception:
            # Проигравший гонку получил ошибку каталога: таблица уже есть -
            # проверяем фактическое наличие, иначе ошибку не глотаем.
            exists = await pool.fetchval(
                "SELECT to_regclass('public.app_settings') IS NOT NULL"
            )
            if not exists:
                raise
        _table_ready = True


class TwoFactorReadState(Enum):
    """Три состояния чтения мастер-флага 2FA.

    OK     - значение прочитано и распарсено;
    EMPTY  - значения нет (пула/таблицы/строки нет) - дефолт False;
    ERROR  - БД ответила ошибкой или значение не распарсить, флаг неизвестен
             (считаем False).
    """

    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"


@dataclass(frozen=True)
class TwoFactorRead:
    """Результат чтения мастер-флага: значение + состояние чтения."""

    enabled: bool
    state: TwoFactorReadState

    @property
    def read_ok(self) -> bool:
        """True при успешном чтении (ok/empty); False только при ошибке."""
        return self.state is not TwoFactorReadState.ERROR


async def read_two_factor(pool: asyncpg.Pool | None) -> TwoFactorRead:
    """Прочитать мастер-флаг 2FA с различением ok/empty/error.

    ok     - успешно прочитан {"enabled": ...};
    empty  - нет пула/таблицы/строки (значение по умолчанию False);
    error  - БД ответила ошибкой или вернула непарсящееся значение (например
             {"enabled": ...} подменён на список/строку): флаг неизвестен,
             считаем False, но read_ok=False, чтобы панель это не скрывала.
    """
    if pool is None:
        return TwoFactorRead(False, TwoFactorReadState.EMPTY)
    try:
        await _ensure_table(pool)
        raw = await pool.fetchval(
            "SELECT value FROM app_settings WHERE key = $1",
            TWO_FACTOR_KEY,
        )
        if raw is None:
            return TwoFactorRead(False, TwoFactorReadState.EMPTY)
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, dict):
            return TwoFactorRead(False, TwoFactorReadState.ERROR)
        return TwoFactorRead(bool(value.get("enabled")), TwoFactorReadState.OK)
    except Exception as exc:
        logger.warning("two_factor_setting_read_failed", error=str(exc))
        return TwoFactorRead(False, TwoFactorReadState.ERROR)


async def get_two_factor_enabled(pool: asyncpg.Pool | None) -> bool:
    """Мастер-выключатель 2FA (булево представление read_two_factor).

    Возвращает False, если пула нет, таблицы/строки нет или БД ответила ошибкой:
    фича выключена по умолчанию, поэтому сбой чтения не должен ломать вход.
    Когда чтение прошло успешно (ok), возвращает сохранённое значение.
    """
    return (await read_two_factor(pool)).enabled


async def set_two_factor_enabled(
    pool: asyncpg.Pool | None, enabled: bool, user_id: str | None = None
) -> None:
    """Включить/выключить 2FA. Пишет updated_at и updated_by."""
    if pool is None:
        return
    await _ensure_table(pool)
    await pool.execute(
        """
        INSERT INTO app_settings (key, value, updated_at, updated_by)
        VALUES ($1, $2::jsonb, now(), $3)
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value,
            updated_at = now(),
            updated_by = EXCLUDED.updated_by
        """,
        TWO_FACTOR_KEY,
        json.dumps({"enabled": enabled}),
        user_id if _is_uuid(user_id) else None,
    )
