"""Alembic merge of the two heads created by main + for_Nikita.

Resolves: fix_pipeline_cancelled (origin/main) and discipline_scope_table
(for_Nikita) both branch from add_rpd_import_action. Empty ops: both
branches' own migrations already carry their schema changes.
"""
from collections.abc import Sequence

revision: str = "merge_for_nikita_main"
down_revision: str | Sequence[str] | None = (
    "fix_pipeline_cancelled",
    "discipline_scope_table",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
