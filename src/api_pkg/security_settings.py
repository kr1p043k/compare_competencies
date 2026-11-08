"""Хранилище прикладных настроек в БД (таблица app_settings)."""

from __future__ import annotations

import asyncio
import json
import uuid
import weakref

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


async def get_two_factor_status(pool: asyncpg.Pool | None) -> dict[str, bool]:
    """Состояние мастер-выключателя 2FA + признак, что чтение удалось.

    Нужен отдельно от get_two_factor_enabled: сам логин остаётся fail-open
    (ошибка чтения = выключено, чтобы сбой БД не запирал админов), но панель
    должна честно показать, что флаг прочитан по значению по умолчанию, иначе
    "тихое" отключение 2FA из-за сбоя никто не заметит.
    """
    if pool is None:
        return {"enabled": False, "read_ok": False}
    try:
        await _ensure_table(pool)
        raw = await pool.fetchval(
            "SELECT value FROM app_settings WHERE key = $1",
            TWO_FACTOR_KEY,
        )
        if raw is None:
            # Строки нет - это нормальное "выключено", чтение успешно.
            return {"enabled": False, "read_ok": True}
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, dict):
            return {"enabled": False, "read_ok": False}
        return {"enabled": bool(value.get("enabled")), "read_ok": True}
    except Exception as exc:
        logger.warning("two_factor_setting_read_failed", error=str(exc))
        return {"enabled": False, "read_ok": False}


async def get_two_factor_enabled(pool: asyncpg.Pool | None) -> bool:
    """Мастер-выключатель 2FA. По умолчанию выключено (fail-open для логина).

    Возвращает False, если пула нет, таблицы/строки нет или БД ответила ошибкой:
    фича выключена по умолчанию, поэтому сбой чтения не должен ломать вход.
    """
    return (await get_two_factor_status(pool))["enabled"]


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
