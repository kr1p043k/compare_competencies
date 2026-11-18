"""NOT NULL on llm_interactions tokens/model/duration/created_at and user_directions.created_at

Модели объявляют эти поля обязательными (Mapped[int]/Mapped[str]), но миграции
создали их nullable -> `alembic check` показывал дрейф. Бэкфилл + SET NOT NULL
(backlog 6.3.2).

Revision ID: fix_not_null_llm_ud
Revises: fix_tz_notif_subs
"""
from __future__ import annotations

from alembic import op

revision = "fix_not_null_llm_ud"
down_revision = "fix_tz_notif_subs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE llm_interactions SET prompt_tokens = COALESCE(prompt_tokens, 0) WHERE prompt_tokens IS NULL")
    op.execute("UPDATE llm_interactions SET response_tokens = COALESCE(response_tokens, 0) WHERE response_tokens IS NULL")
    op.execute("UPDATE llm_interactions SET model = COALESCE(model, 'unknown') WHERE model IS NULL")
    op.execute("UPDATE llm_interactions SET duration_ms = COALESCE(duration_ms, 0) WHERE duration_ms IS NULL")
    op.execute("UPDATE llm_interactions SET created_at = NOW() WHERE created_at IS NULL")
    for col in ("prompt_tokens", "response_tokens", "model", "duration_ms", "created_at"):
        op.execute(f"ALTER TABLE llm_interactions ALTER COLUMN {col} SET NOT NULL")

    op.execute("UPDATE user_directions SET created_at = NOW() WHERE created_at IS NULL")
    op.execute("ALTER TABLE user_directions ALTER COLUMN created_at SET NOT NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE user_directions ALTER COLUMN created_at DROP NOT NULL")
    for col in ("prompt_tokens", "response_tokens", "model", "duration_ms", "created_at"):
        op.execute(f"ALTER TABLE llm_interactions ALTER COLUMN {col} DROP NOT NULL")
