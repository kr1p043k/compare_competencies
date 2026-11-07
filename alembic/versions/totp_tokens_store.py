"""Таблица totp_tokens: одноразовые 2FA-токены (challenge входа и настройка).

В БД лежит только SHA-256 сырого токена, а сам сырой токен отдаётся клиенту
ровно один раз. Переносим токены из per-process dict в Postgres, чтобы вход не
ломался при нескольких uvicorn-воркерах: single-use достигается атомарным
DELETE ... RETURNING при consume.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "totp_tokens_store"
down_revision: str | None = "add_chat_storage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS totp_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            purpose    TEXT NOT NULL CHECK (purpose IN ('login_challenge', 'enroll')),
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_totp_tokens_expires ON totp_tokens (expires_at)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS totp_tokens")
