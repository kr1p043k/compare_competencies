"""Allow 'cancelled' status in pipeline_runs.

rpd_cancel (src.api_pkg.routers.rpd) writes status='cancelled',
but the DB CHECK constraint only allowed ('started','completed','failed').
"""

from typing import Sequence, Union

from alembic import op

revision: str = "fix_pipeline_runs_status_cancelled"
down_revision: Union[str, None] = "add_rpd_import_action"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_status_check")
    op.execute("""
        ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_status_check
            CHECK (status IN ('started', 'completed', 'failed', 'cancelled'))
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_status_check")
    op.execute("""
        ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_status_check
            CHECK (status IN ('started', 'completed', 'failed'))
    """)