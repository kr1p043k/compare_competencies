"""Alembic merge: intake-year (for_Nikita/auto) + hub-publications-merge (main).

Обе ветки независимы: `add_discipline_intake_year` (колонка intake_year)
и `merge_for_nikita_main` (пустая склейка fix_pipeline_cancelled +
discipline_scope_table). Схемные изменения уже в самих миграциях, здесь
только сводим историю к одному head.
"""
from typing import Sequence, Union

revision: str = "merge_intake_year_main"
down_revision: Union[str, Sequence[str], None] = (
    "merge_for_nikita_main",
    "add_discipline_intake_year",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
