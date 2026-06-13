"""add parsed_skills to vacancies

Revision ID: 90d02040d3d1
Revises:
Create Date: 2026-06-13 03:59:24.346602

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "90d02040d3d1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "vacancies",
        sa.Column("parsed_skills", sa.JSON(), nullable=True),
    )
    op.create_index(
        "idx_vacancies_parsed_pub",
        "vacancies",
        ["published_at"],
        postgresql_where=sa.text("parsed_skills IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_vacancies_parsed_pub")
    op.drop_column("vacancies", "parsed_skills")
