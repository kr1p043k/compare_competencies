"""Add 'reparse' + 'hh-import' actions to pipeline_runs CHECK constraint.

'reparse': populate-parsed-skills registers its runs so the teacher-analysis
skip-gate sees refreshed vacancy data (vacancies has no updated_at).
'hh-import': already queried by the skip-gate but missing from the CHECK
(latent gap).
"""

from typing import Sequence, Union

from alembic import op

revision: str = "add_reparse_action"
down_revision: Union[str, None] = "add_weighted_coverage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_action_check")
    op.execute("""
        ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_action_check
            CHECK (action IN (
                'full-cycle', 'rebuild', 'train-clusters', 'train-model',
                'gap-analysis', 'teacher-analysis', 'data-collection',
                'rpd-import', 'reparse', 'hh-import'
            ))
    """)


def downgrade() -> None:
    # hh-import оставляем: раннер уже опрашивал его до этой миграции.
    op.execute("ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_action_check")
    op.execute("""
        ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_action_check
            CHECK (action IN (
                'full-cycle', 'rebuild', 'train-clusters', 'train-model',
                'gap-analysis', 'teacher-analysis', 'data-collection',
                'rpd-import', 'hh-import'
            ))
    """)
