"""Add idempotency_key to pipeline_runs for strict task dedup.

Strict idempotency: API keeps (action, idempotency_key) -> task_id map
in data/cache/pipeline_idempotency.json; DB enforces uniqueness so a
retry in the same second cannot start a duplicate heavy job.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "pipeline_idempotency_key"
down_revision: Union[str, None] = "add_rpd_import_action"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(64)")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_pipeline_runs_idempotency "
        "ON pipeline_runs (action, idempotency_key) WHERE idempotency_key IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_pipeline_runs_idempotency")
    op.execute("ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS idempotency_key")
