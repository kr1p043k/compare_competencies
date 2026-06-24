"""Replicates sql/004_llm_recommendations.sql.

Creates llm_recommendations, profile_evaluations tables.
ALTER TABLE recommendations: source, llm_request_id, confidence.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "add_llm_tables"
down_revision: Union[str, None] = "add_audit_fixes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── 1. llm_recommendations ──────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_recommendations (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            request_hash    TEXT NOT NULL,
            prompt_text     TEXT NOT NULL,
            model_used      VARCHAR(20) NOT NULL
                CHECK (model_used IN ('qwen3.6', 'gemma4', 'qwen_local', 'deepseek_local')),
            response_json   JSONB NOT NULL DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_req_hash ON llm_recommendations(request_hash)")

    # ─── 2. ALTER recommendations: source, llm_request_id, confidence ────────
    op.execute("""
        ALTER TABLE recommendations
            ADD COLUMN IF NOT EXISTS source VARCHAR(20)
                CHECK (source IN ('shap', 'llm', 'llm_fallback'))
    """)
    op.execute("""
        ALTER TABLE recommendations
            ADD COLUMN IF NOT EXISTS llm_request_id UUID
                REFERENCES llm_recommendations(id) ON DELETE SET NULL
    """)
    op.execute("""
        ALTER TABLE recommendations
            ADD COLUMN IF NOT EXISTS confidence FLOAT
                CHECK (confidence >= 0 AND confidence <= 1)
    """)

    # ─── 3. profile_evaluations ──────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS profile_evaluations (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id           UUID REFERENCES users(id) ON DELETE SET NULL,
            discipline_id     UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
            evaluation_type   VARCHAR(20) NOT NULL
                CHECK (evaluation_type IN ('gap', 'coverage', 'full')),
            input_summary     JSONB NOT NULL DEFAULT '{}',
            result_summary    JSONB NOT NULL DEFAULT '{}',
            llm_request_id    UUID REFERENCES llm_recommendations(id) ON DELETE SET NULL,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_prof_eval_user ON profile_evaluations(user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_prof_eval_discipline ON profile_evaluations(discipline_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS profile_evaluations CASCADE")
    op.execute("ALTER TABLE recommendations DROP COLUMN IF EXISTS confidence")
    op.execute("ALTER TABLE recommendations DROP COLUMN IF EXISTS llm_request_id")
    op.execute("ALTER TABLE recommendations DROP COLUMN IF EXISTS source")
    op.execute("DROP TABLE IF EXISTS llm_recommendations CASCADE")
