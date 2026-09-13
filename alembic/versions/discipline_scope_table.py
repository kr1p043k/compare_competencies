"""Discipline analysis scope overrides (UI checkboxes, v41)."""

from typing import Sequence, Union

from alembic import op

revision: str = "discipline_scope_table"
down_revision: Union[str, None] = "ksa_updated_at"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""CREATE TABLE IF NOT EXISTS discipline_scope (
        direction_code TEXT NOT NULL,
        discipline_name TEXT NOT NULL,
        included BOOLEAN NOT NULL DEFAULT TRUE,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (direction_code, discipline_name)
    )""")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS discipline_scope")
