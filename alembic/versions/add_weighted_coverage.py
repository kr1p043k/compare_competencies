"""Add weighted_coverage to coverage_analyses (conservative level support).

Revision ID: add_weighted_coverage
Revises: add_llm_cache_tables
"""

from typing import Sequence, Union

from alembic import op

revision: str = "add_weighted_coverage"
down_revision: Union[str, None] = "add_llm_cache_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE coverage_analyses "
        "ADD COLUMN IF NOT EXISTS weighted_coverage DOUBLE PRECISION"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE coverage_analyses DROP COLUMN IF EXISTS weighted_coverage"
    )
