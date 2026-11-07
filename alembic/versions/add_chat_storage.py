"""Add chat_threads + chat_messages tables; relax llm_recommendations model_used CHECK.

Creates two new tables for LLM chat history and removes the overly restrictive
CHECK constraint on llm_recommendations.model_used, replacing it with plain
TEXT so any real model identifier (including provider-prefixed names like
"deepseek-ai/DeepSeek-V4.1-Pro-0813") is accepted without requiring migration
every time a new model is used.

Revision ID: add_chat_storage
Revises: merge_all_heads
"""

from collections.abc import Sequence

from alembic import op

revision: str = "add_chat_storage"
down_revision: str | None = "merge_all_heads"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── 1. chat_threads ───────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE chat_threads (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
            context_type    VARCHAR(20) NOT NULL
                CHECK (context_type IN ('teacher', 'student')),
            context_key     VARCHAR(255) NOT NULL,
            title           TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(
        "CREATE INDEX idx_ct_user ON chat_threads(user_id, created_at)"
    )

    # ─── 2. chat_messages ──────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE chat_messages (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            thread_id       UUID NOT NULL REFERENCES chat_threads(id)
                            ON DELETE CASCADE,
            role            VARCHAR(16) NOT NULL
                CHECK (role IN ('user', 'assistant')),
            seq             INTEGER NOT NULL,
            blocks_json     JSONB,
            model_used      VARCHAR(64),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (thread_id, seq)
        )
    """)
    op.execute(
        "CREATE INDEX idx_cm_thread_seq ON chat_messages(thread_id, seq)"
    )

    # ─── 3. Relax llm_recommendations.model_used CHECK ─────────────────────
    #    The original CHECK only allowed five hardcoded model names.
    #    Production uses provider-prefixed identifiers (e.g.
    #    "deepseek-ai/DeepSeek-V4.1-Pro-0813").  Rather than extending the
    #    list on every new model (which would require a migration for each
    #    one), we drop the CHECK and leave model_used as TEXT with no
    #    restriction.  The column length stays VARCHAR(64) in the model
    #    layer for practical purposes.
    #
    #    Strategy: create a CHECK that matches anything (effectively no-op),
    #    then drop the old constraint by name.
    op.execute(
        "ALTER TABLE llm_recommendations DROP CONSTRAINT IF EXISTS llm_recommendations_model_used_check"
    )
    op.execute(
        "ALTER TABLE llm_recommendations ALTER COLUMN model_used TYPE TEXT USING model_used::TEXT"
    )


def downgrade() -> None:
    # ─── 1. Restore llm_recommendations CHECK + VARCHAR ────────────────────
    op.execute(
        "ALTER TABLE llm_recommendations ALTER COLUMN model_used TYPE VARCHAR(20) USING model_used::VARCHAR(20)"
    )
    op.execute("""
        ALTER TABLE llm_recommendations
            ADD CONSTRAINT llm_recommendations_model_used_check
            CHECK (model_used IN (
                'qwen3.6', 'gemma4', 'qwen_local',
                'deepseek_local', 'gpt-oss'
            ))
    """)

    # ─── 2. Drop chat tables ──────────────────────────────────────────────
    op.execute("DROP TABLE IF EXISTS chat_messages")
    op.execute("DROP TABLE IF EXISTS chat_threads")