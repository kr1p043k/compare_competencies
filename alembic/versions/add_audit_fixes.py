"""Replicates sql/002_audit_fixes.sql and sql/002_search_runs.sql.

ALTER TABLE ADD COLUMN direction_id on several tables,
refined CHECK constraints, market_skill_mappings changes.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "add_audit_fixes"
down_revision: Union[str, None] = "initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── 1. analysis_results: direction_id + expanded type CHECK ─────────────
    op.execute("ALTER TABLE analysis_results ADD COLUMN IF NOT EXISTS direction_id UUID REFERENCES directions(id) ON DELETE CASCADE")
    op.execute("""
        ALTER TABLE analysis_results DROP CONSTRAINT IF EXISTS analysis_results_analysis_type_check
    """)
    op.execute("""
        UPDATE analysis_results SET analysis_type = 'teacher-analysis'
        WHERE analysis_type = 'teacher' OR analysis_type = 'teacher-analysis'
    """)
    op.execute("""
        ALTER TABLE analysis_results ADD CONSTRAINT analysis_results_analysis_type_check
            CHECK (analysis_type IN ('gap', 'coverage', 'cluster', 'trend', 'teacher-analysis'))
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ar_direction ON analysis_results(direction_id)")

    # ─── 2. pipeline_runs: expanded action CHECK ─────────────────────────────
    op.execute("""
        ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_action_check
    """)
    op.execute("""
        ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_action_check
            CHECK (action IN (
                'full-cycle', 'rebuild', 'train-clusters', 'train-model',
                'gap-analysis', 'teacher-analysis', 'data-collection'
            ))
    """)

    # ─── 3. student_skills: direction_id + competency_id ─────────────────────
    op.execute("ALTER TABLE student_skills ADD COLUMN IF NOT EXISTS direction_id UUID REFERENCES directions(id) ON DELETE SET NULL")
    op.execute("ALTER TABLE student_skills ADD COLUMN IF NOT EXISTS competency_id UUID REFERENCES competencies(id) ON DELETE SET NULL")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ss_direction ON student_skills(direction_id)")

    # ─── 4. coverage_analyses: direction_id ──────────────────────────────────
    op.execute("ALTER TABLE coverage_analyses ADD COLUMN IF NOT EXISTS direction_id UUID REFERENCES directions(id) ON DELETE CASCADE")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ca_direction ON coverage_analyses(direction_id)")

    # ─── 5. recommendations: direction_id ────────────────────────────────────
    op.execute("ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS direction_id UUID REFERENCES directions(id) ON DELETE CASCADE")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rec_direction ON recommendations(direction_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_rec_direction")
    op.execute("ALTER TABLE recommendations DROP COLUMN IF EXISTS direction_id")
    op.execute("DROP INDEX IF EXISTS idx_ca_direction")
    op.execute("ALTER TABLE coverage_analyses DROP COLUMN IF EXISTS direction_id")
    op.execute("DROP INDEX IF EXISTS idx_ss_direction")
    op.execute("ALTER TABLE student_skills DROP COLUMN IF EXISTS direction_id")
    op.execute("ALTER TABLE student_skills DROP COLUMN IF EXISTS competency_id")
    op.execute("ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_action_check")
    op.execute("DROP INDEX IF EXISTS idx_ar_direction")
    op.execute("ALTER TABLE analysis_results DROP COLUMN IF EXISTS direction_id")
    op.execute("ALTER TABLE analysis_results DROP CONSTRAINT IF EXISTS analysis_results_analysis_type_check")
