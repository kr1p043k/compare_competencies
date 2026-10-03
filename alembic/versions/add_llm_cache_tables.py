"""Add llm_cache + skill_embedding_cache (LLM call cache, pgvector 768).

Revision ID: add_llm_cache_tables
Revises: add_request_logs_detail
"""

from typing import Sequence, Union

from alembic import op

revision: str = "add_llm_cache_tables"
down_revision: Union[str, None] = "add_request_logs_detail"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_cache (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            task TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_hash CHAR(64) NOT NULL,
            prompt_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (task, model, prompt_hash)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_embedding_cache (
            skill_key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            embedding vector(768) NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS skill_embedding_cache")
    op.execute("DROP TABLE IF EXISTS llm_cache")
