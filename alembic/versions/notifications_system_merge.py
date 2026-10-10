"""merge branches + notifications: nullable subscription_id, severity column

Revision ID: notifications_system_merge
Revises: add_sessions_sso_token, remove_non_it_disciplines, add_subs_and_notifs
Create Date: 2026-08-08 00:00:00.000000

Мерджит три головы и делает changes для системных уведомлений:
- notifications.subscription_id -> nullable (системные ошибки без подписки)
- notifications.severity ('info' | 'warning' | 'error')
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "notifications_system_merge"
down_revision: str | Sequence[str] | None = (
    "add_sessions_sso_token",
    "remove_non_it_disciplines",
    "add_subs_and_notifs",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "notifications",
        "subscription_id",
        existing_type=UUID,
        nullable=True,
    )
    op.add_column(
        "notifications",
        sa.Column("severity", sa.String(20), nullable=False, server_default="info"),
    )


def downgrade() -> None:
    op.drop_column("notifications", "severity")
    op.alter_column(
        "notifications",
        "subscription_id",
        existing_type=UUID,
        nullable=False,
    )
