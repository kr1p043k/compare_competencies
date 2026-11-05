"""Глобальный флаг двухфакторной аутентификации администраторов.

2FA была жёстко включена для роли admin: если у администратора подтверждён
TOTP, вход без кода невозможен, а выключить это из интерфейса было нельзя.
Флаг живёт в таблице ``app_settings`` (ключ-значение), а не в переменных
окружения, чтобы его можно было переключить кнопкой и увидеть эффект сразу.

По умолчанию строки нет - это значит «выключено».

Revision ID: twofa_toggle
Revises: trend_precomputed
"""

from collections.abc import Sequence

from alembic import op

revision: str = "twofa_toggle"
down_revision: str | None = "trend_precomputed"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by  TEXT
        )
        """
    )
    # Явная строка «выключено», чтобы состояние было видно в БД сразу после
    # миграции, а не только по отсутствию записи.
    op.execute(
        """
        INSERT INTO app_settings (key, value, updated_by)
        VALUES ('twofa_admin_enabled', '0', 'migration')
        ON CONFLICT (key) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DELETE FROM app_settings WHERE key = 'twofa_admin_enabled'")
