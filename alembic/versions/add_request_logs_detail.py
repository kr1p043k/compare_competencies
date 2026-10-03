"""Add detail to request_logs (audit trail: actor/action/target).

Merges the two current heads (vacancy_employer_logo, discipline_scope_table).
Mirrors sql/007_request_logs_detail.sql.

Revision ID: add_request_logs_detail
Revises: vacancy_employer_logo, discipline_scope_table
"""

from typing import Sequence, Union

from alembic import op

revision: str = "add_request_logs_detail"
down_revision: Union[str, Sequence[str], None] = (
    "vacancy_employer_logo",
    "discipline_scope_table",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE request_logs ADD COLUMN IF NOT EXISTS detail TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE request_logs DROP COLUMN IF EXISTS detail")
