"""Per-account brute-force protection: failed_login_count / locked_until.

Раньше единственной защитой входа был rate-limit по IP (10/мин), причём IP
брался из заголовка X-Forwarded-For, который клиент подставляет сам. Один
curl-цикл с меняющимся XFF обходил лимит полностью.
Эти колонки дают блокировку по аккаунту, не зависящую от IP.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "login_bruteforce_guard"
down_revision: str | None = "add_reparse_action"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS failed_login_count INTEGER NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_failed_login_at TIMESTAMPTZ"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_locked_until ON users (locked_until) "
        "WHERE locked_until IS NOT NULL"
    )
    # Логин ищет пользователя без учёта регистра email.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_email_lower ON users (lower(email))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_users_email_lower")
    op.execute("DROP INDEX IF EXISTS idx_users_locked_until")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS last_failed_login_at")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS locked_until")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS failed_login_count")
