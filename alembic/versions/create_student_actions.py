"""create student_actions table (was created by app startup, not migrations)

Раньше `student_actions` создавалась сырым SQL при старте приложения
(`src/api_pkg/student_actions.py`), поэтому в БД из миграций её не было и
`alembic check` показывал дрейф. Делаем таблицу частью схемы миграций
(idempotent CREATE IF NOT EXISTS; стартовый код остаётся совместимым).
Backlog 6.3.2.

Revision ID: create_student_actions
Revises: fix_not_null_llm_ud
"""
from __future__ import annotations

from alembic import op

revision = "create_student_actions"
down_revision = "fix_not_null_llm_ud"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS student_actions (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            action_type TEXT NOT NULL DEFAULT '',
            profession TEXT DEFAULT '',
            region TEXT DEFAULT '',
            vacancies_found INTEGER DEFAULT 0,
            result_ref TEXT DEFAULT '',
            profile TEXT DEFAULT '',
            timestamp TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS student_actions")
