"""add sso_token column to sessions

Revision ID: add_sessions_sso_token
Revises: merge_branches
Create Date: 2026-08-05 18:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "add_sessions_sso_token"
down_revision: Union[str, None] = "merge_branches"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("sessions", sa.Column("sso_token", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("sessions", "sso_token")
