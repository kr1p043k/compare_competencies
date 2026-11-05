"""Глобальная политика двухфакторной аутентификации.

2FA нужна не всегда: включённая по умолчанию, она блокирует вход, если у
администратора не настроен TOTP, а выключить её из интерфейса было нельзя.
Поэтому решение хранится отдельным флагом, а не выводится из наличия
подтверждённых TOTP-записей:

* флаг выключен (по умолчанию) - вход по паролю, TOTP не спрашивается,
  даже если у кого-то остались старые записи в ``admin_totp``;
* флаг включён - у администраторов с подтверждённым TOTP спрашивается код.

Почему fail-open при недоступной БД: значение по умолчанию - «выключено», и
если таблицу настроек не удалось прочитать, безопаснее не запереть всех
администраторов снаружи. Сбой чтения логируется как warning, чтобы это не
осталось незамеченным.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

#: Ключ настройки в таблице ``app_settings``.
TWOFA_ADMIN_KEY = "twofa_admin_enabled"

#: Роли, к которым применяется флаг.
TWOFA_ROLES = ("admin",)

_CREATE_SETTINGS = """
CREATE TABLE IF NOT EXISTS app_settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by  TEXT
)
"""


async def ensure_settings_table(pool) -> None:
    """Создать таблицу настроек, если миграция ещё не применена.

    Страховка на случай запуска до миграции: без неё чтение флага падало бы,
    и решение «выключено» принималось бы молча.
    """
    await pool.execute(_CREATE_SETTINGS)


async def is_twofa_enabled(pool) -> bool:
    """Включена ли 2FA для администраторов. По умолчанию - нет."""
    if pool is None:
        return False
    try:
        row = await pool.fetchrow(
            "SELECT value FROM app_settings WHERE key = $1", TWOFA_ADMIN_KEY
        )
    except Exception as exc:
        # Нет таблицы или БД недоступна: не запираем вход, но фиксируем сбой.
        logger.warning("twofa_flag_unreadable", error=str(exc)[:200])
        return False
    if row is None:
        return False
    return str(row["value"]).strip().lower() in ("1", "true", "yes", "on")


async def set_twofa_enabled(pool, enabled: bool, actor: str = "") -> None:
    """Записать флаг. ``actor`` - кто включил, для аудита."""
    await ensure_settings_table(pool)
    await pool.execute(
        """
        INSERT INTO app_settings (key, value, updated_at, updated_by)
        VALUES ($1, $2, now(), $3)
        ON CONFLICT (key) DO UPDATE
           SET value = EXCLUDED.value,
               updated_at = now(),
               updated_by = EXCLUDED.updated_by
        """,
        TWOFA_ADMIN_KEY,
        "1" if enabled else "0",
        actor or None,
    )
    logger.info("twofa_flag_changed", enabled=enabled, actor=actor)


def applies_to_role(role: str | None) -> bool:
    """Нужно ли спрашивать код у этой роли, если флаг включён."""
    return (role or "").strip().lower() in TWOFA_ROLES


async def should_require_totp(pool, role: str | None) -> bool:
    """Короткая проверка для эндпоинтов входа: роль + глобальный флаг."""
    if not applies_to_role(role):
        return False
    return await is_twofa_enabled(pool)
