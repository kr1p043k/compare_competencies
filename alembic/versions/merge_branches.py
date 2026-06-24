"""Merge two migration branches: 90d02040d3d1 ← add_directions_and_supervisors

Branch 1: initial_schema → add_audit_fixes → add_llm_tables → add_directions_and_supervisors
Branch 2: initial_schema → 90d02040d3d1 (parsed_skills)
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "merge_branches"
down_revision: Union[str, Sequence[str], None] = ("90d02040d3d1", "add_directions_and_supervisors")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
