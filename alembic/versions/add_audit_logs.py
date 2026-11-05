"""Durable audit log: таблица audit_logs (переживает retention request_logs).

request_logs чистится src/cli/retention.py, поэтому события аудита дублируются
в отдельную таблицу audit_logs (best-effort из request_logger.audit_action).
Индексы покрывают фильтры GET /admin/audit: по времени, action+время,
actor_email, и пара target_type+target_id.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "add_audit_logs"
down_revision: str | None = "trend_precomputed"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_logs",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("actor_id", UUID, nullable=True, index=True),
        sa.Column("actor_email", sa.Text(), nullable=True),
        sa.Column("actor_role", sa.String(20), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("target_type", sa.String(50), nullable=True),
        sa.Column("target_id", sa.String(100), nullable=True),
        sa.Column("target", sa.Text(), nullable=True),
        sa.Column("meta", JSONB, nullable=True),
        sa.Column("ip", sa.String(45), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_audit_created", "audit_logs", ["created_at"])
    op.create_index("idx_audit_action_created", "audit_logs", ["action", "created_at"])
    op.create_index("idx_audit_actor", "audit_logs", ["actor_email"])
    op.create_index("idx_audit_target", "audit_logs", ["target_type", "target_id"])


def downgrade() -> None:
    # Индексы удаляются вместе с таблицей (в т.ч. автоиндекс ix_audit_logs_actor_id).
    op.drop_table("audit_logs")
