"""TOTP 2FA for the admin role: admin_totp table.

Секрет шифруется Fernet-ключом, выведенным из SECRET_KEY (как hub-JWT в
academic.py), восстановительные коды хранятся хешами (jsonb). Счётчик неудач и
блокировка на 15 минут защищают verify/recovery от перебора - аналог
failed_login_count/locked_until у входа.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "add_admin_totp"
down_revision: str | None = "merge_owner_and_published_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_totp (
            user_id         UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            secret          TEXT NOT NULL,
            confirmed       BOOLEAN NOT NULL DEFAULT false,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_used_at    TIMESTAMPTZ,
            recovery_codes  JSONB,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until    TIMESTAMPTZ
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS admin_totp")
