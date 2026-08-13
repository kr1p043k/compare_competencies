"""Add 'rpd-import' action to pipeline_runs CHECK constraint.

Used by the RPD upload/collect pipeline (src.api_pkg.routers.rpd).
"""

from typing import Sequence, Union

from alembic import op

revision: str = "add_rpd_import_action"
down_revision: Union[str, None] = "competency_vpk_subcodes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_action_check")
    op.execute("""
        ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_action_check
            CHECK (action IN (
                'full-cycle', 'rebuild', 'train-clusters', 'train-model',
                'gap-analysis', 'teacher-analysis', 'data-collection',
                'rpd-import'
            ))
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_action_check")
    op.execute("""
        ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_action_check
            CHECK (action IN (
                'full-cycle', 'rebuild', 'train-clusters', 'train-model',
                'gap-analysis', 'teacher-analysis', 'data-collection'
            ))
    """)
