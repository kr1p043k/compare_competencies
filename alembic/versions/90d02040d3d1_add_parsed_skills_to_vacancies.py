"""Добавляем parsed_skills в таблицу vacancies

Revision ID: 90d02040d3d1
Revises: 
Create Date: 2026-06-15 06:10:24.346602
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "90d02040d3d1"
down_revision: Union[str, None] = "initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("ALTER TABLE vacancies ADD COLUMN IF NOT EXISTS parsed_skills JSONB")
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_vacancies_parsed_pub
        ON vacancies (published_at) WHERE parsed_skills IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_vacancies_parsed_pub")
    op.execute("ALTER TABLE vacancies DROP COLUMN IF EXISTS parsed_skills")
