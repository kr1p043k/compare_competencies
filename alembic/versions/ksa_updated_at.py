"""Add ksa_entries.updated_at for teacher-edit invalidation."""

from typing import Sequence, Union

from alembic import op

revision: str = "ksa_updated_at"
down_revision: Union[str, None] = "add_rpd_import_action"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE ksa_entries ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ")
    op.execute("UPDATE ksa_entries SET updated_at = created_at WHERE updated_at IS NULL")
    op.execute("ALTER TABLE ksa_entries ALTER COLUMN updated_at SET NOT NULL")
    op.execute("ALTER TABLE ksa_entries ALTER COLUMN updated_at SET DEFAULT NOW()")


def downgrade() -> None:
    op.execute("ALTER TABLE ksa_entries DROP COLUMN IF EXISTS updated_at")
