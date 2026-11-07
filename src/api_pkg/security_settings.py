"""Хранилище прикладных настроек в БД (таблица app_settings)."""

from __future__ import annotations

import asyncio
import json
import uuid

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

TWO_FACTOR_KEY = "security.two_factor"


class TwoFactorSettingUnavailableError(RuntimeError):
    """Состояние 2FA не удалось прочитать из БД: "state unknown, deny".

    Означает fail-closed: вызывающий код обязан отклонить запрос (или вернуть
    503), а НЕ считать 2FA выключенной и пропускать админа без второго фактора.
    """

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by UUID NULL REFERENCES users(id) ON DELETE SET NULL
)
"""

_table_ready = False
#: Замок на создание таблицы: без него два первых параллельных запроса
#: выполняют CREATE TABLE IF NOT EXISTS одновременно и один из них падает
#: (гонка в каталоге pg_type), причём на пути записи это 500.
_ensure_lock: asyncio.Lock | None = None


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
    global _table_ready, _ensure_lock

    if _table_ready:
        return
    if _ensure_lock is None:
        _ensure_lock = asyncio.Lock()
    async with _ensure_lock:
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


async def get_two_factor_enabled(pool: asyncpg.Pool | None) -> bool:
    """Мастер-выключатель 2FA. По умолчанию выключено.

    Чтение fail-closed:
    - пула нет (pool is None) -> False: без БД аутентификация всё равно не пройдёт;
    - таблица ещё не создана (свежий деплой до миграции) -> warning и False;
    - любая другая ошибка чтения -> TwoFactorSettingUnavailable ("deny").
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
    except asyncpg.UndefinedTableError as exc:
        logger.warning("two_factor_table_missing", error=str(exc))
        return False
    except Exception as exc:
        logger.error("two_factor_setting_read_failed", error=str(exc))
        raise TwoFactorSettingUnavailableError(
            "Состояние 2FA неизвестно при чтении из БД (state unknown, deny): "
            f"{exc}"
        ) from exc


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
