"""add cost_usd to llm_interactions

Revision ID: add_llm_cost
Revises: merge_all_heads
Create Date: 2026-11-07

Add a nullable cost_usd column to llm_interactions and an index to make
aggregation (SUM / AVG by model or date) cheaper.  No backfill needed –
existing rows stay NULL until a companion script populates them.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "add_llm_cost"
down_revision: tuple[str, ...] | str = ("add_chat_storage",)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "llm_interactions",
        sa.Column(
            "cost_usd",
            sa.Numeric(precision=12, scale=6),
            nullable=True,
        ),
    )
    op.create_index(
        op.f("ix_llm_interactions_cost_usd"),
        "llm_interactions",
        ["cost_usd"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_llm_interactions_cost_usd"), table_name="llm_interactions")
    op.drop_column("llm_interactions", "cost_usd")
