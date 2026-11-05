"""Глобальный мастер-выключатель 2FA: таблица app_settings.

Двухфакторная аутентификация остаётся в коде, но по умолчанию выключена.
Состояние тумблера ("2FA для админов") хранится в БД, а не в env или файле:
так админ переключает его по клику, и оно переживает рестарт процесса и
делится между воркерами. Здесь создаётся таблица key/value и сеется стартовая
строка ``security.two_factor`` = ``{"enabled": false}``.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "app_settings"
down_revision: str | None = "trend_precomputed"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key        TEXT PRIMARY KEY,
            value      JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_by UUID NULL REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    op.execute(
        """
        INSERT INTO app_settings (key, value)
        VALUES ('security.two_factor', '{"enabled": false}'::jsonb)
        ON CONFLICT (key) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS app_settings")
