"""Alembic merge of the two heads created by main + for_Nikita.

Resolves: fix_pipeline_cancelled (origin/main) and discipline_scope_table
(for_Nikita) both branch from add_rpd_import_action. Empty ops: both
branches' own migrations already carry their schema changes.
"""
from typing import Sequence, Union

from alembic import op  # noqa: F401  (kept for alembic convention)

revision: str = "merge_for_nikita_main"
down_revision: Union[str, Sequence[str], None] = (
    "fix_pipeline_cancelled",
    "discipline_scope_table",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
