"""Merge the pipeline_runs owner columns with the vacancies index line.

Две ветки миграций разошлись: `pipeline_runs_owner_dir` (owner/direction для
запусков) и `vacancies_published_nulls_last` (индекс под ORDER BY published_at).
Обе независимы, поэтому сводим их merge-ревизией: без неё `alembic upgrade head`
падает с «Multiple head revisions».
"""

from collections.abc import Sequence

revision: str = "merge_owner_and_published_index"
down_revision: str | Sequence[str] | None = (
    "pipeline_runs_owner_dir",
    "vacancies_published_nulls_last",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Ничего не делает: обе ветки уже применены своими миграциями."""


def downgrade() -> None:
    """Развод веток обратно на два head (для отката достаточно pass)."""
