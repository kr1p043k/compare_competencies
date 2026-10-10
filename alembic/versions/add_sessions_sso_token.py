"""add sso_token column to sessions

Revision ID: add_sessions_sso_token
Revises: merge_branches
Create Date: 2026-08-05 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "add_sessions_sso_token"
down_revision: str | None = "merge_branches"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sessions", sa.Column("sso_token", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("sessions", "sso_token")
