"""Хранилище прикладных настроек в БД (таблица app_settings)."""

from __future__ import annotations

import json

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


async def _ensure_table(pool: asyncpg.Pool) -> None:
    """Страховка на случай деплоя без миграции."""
    global _table_ready
    if _table_ready:
        return
    await pool.execute(_TABLE_DDL)
    _table_ready = True


async def get_two_factor_enabled(pool: asyncpg.Pool | None) -> bool:
    """Мастер-выключатель 2FA. По умолчанию выключено.

    Возвращает False, если пула нет, таблицы/строки нет или БД ответила ошибкой:
    фича выключена по умолчанию, поэтому сбой чтения не должен ломать вход.
    """
    if pool is None:
        return False
    try:
        await _ensure_table(pool)
        raw = await pool.fetchval(
            "SELECT value FROM app_settings WHERE key = $1",
            TWO_FACTOR_KEY,
        )
        if raw is None:
            return False
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, dict):
            return False
        return bool(value.get("enabled"))
    except Exception as exc:
        logger.warning("two_factor_setting_read_failed", error=str(exc))
        return False


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
        user_id,
    )
