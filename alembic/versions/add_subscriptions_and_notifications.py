"""add subscriptions and notifications tables

Revision ID: add_subscriptions_and_notifications
Revises: merge_branches
Create Date: 2026-07-25 17:30:00.000000
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "add_subs_and_notifs"
down_revision: Union[str, None] = "merge_branches"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "subscriptions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False, index=True),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False, server_default="openalex+arxiv"),
        sa.Column("telegram_chat_id", sa.String(255), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("last_checked_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("source IN ('openalex', 'arxiv', 'openalex+arxiv')", name="ck_sub_source"),
    )
    op.create_index("idx_sub_user_active", "subscriptions", ["user_id", "is_active"])

    op.create_table(
        "notifications",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("subscription_id", UUID, sa.ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False, index=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("article_url", sa.Text(), nullable=True),
        sa.Column("article_source", sa.String(50), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("delivered_via", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_notif_user_unread", "notifications", ["user_id", "is_read"])
    op.create_index("idx_notif_created", "notifications", ["created_at"])


def downgrade() -> None:
    op.drop_table("notifications")
    op.drop_table("subscriptions")
