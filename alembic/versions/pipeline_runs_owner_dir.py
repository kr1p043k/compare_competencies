"""Add owner and direction to pipeline_runs so runs can be scoped.

Without this a teacher could read or cancel any other teacher's run by id
(rpd_status / rpd_cancel accepted any run_id).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "pipeline_runs_owner_dir"
down_revision: str | None = "merge_intake_year_main"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS owner_id UUID")
    op.execute("ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS direction_code VARCHAR(20)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_owner ON pipeline_runs (owner_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_direction "
        "ON pipeline_runs (direction_code)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pipeline_runs_direction")
    op.execute("DROP INDEX IF EXISTS idx_pipeline_runs_owner")
    op.execute("ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS direction_code")
    op.execute("ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS owner_id")
