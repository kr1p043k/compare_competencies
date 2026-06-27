"""Удаляет не-IT дисциплины из таблицы disciplines.

Удаляет 13 общеобразовательных дисциплин (Безопасность жизнедеятельности,
физкультура, иностранные языки, история, и т.д.) каскадно вместе со
всеми связанными компетенциями, KSA, навыками и рекомендациями.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "remove_non_it_disciplines"
down_revision: Union[str, Sequence[str], None] = "merge_branches"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

REMOVED_DISCIPLINES = [
    "Безопасность жизнедеятельности",
    "Дисциплины по ФКиС",
    "История России",
    "Иностранный язык (англ. яз., уровень А1)",
    "Иностранный язык (англ. яз., уровень А2)",
    "Иностранный язык (англ. яз., уровень В1)",
    "Иностранный язык (англ. яз., уровень В2)",
    "Иностранный язык (англ. яз., уровень С1)",
    "Иностранный язык (русский язык)",
    "Иностранный язык для деловой коммуникации",
    "Воображение, изображение, реальность",
    "Основы проектной деятельности",
    "Практикум по подготовке инженерной документации",
]


def upgrade() -> None:
    for name in REMOVED_DISCIPLINES:
        name_safe = name.replace("'", "''")
        op.execute(f"DELETE FROM disciplines WHERE name = '{name_safe}'")


def downgrade() -> None:
    pass
