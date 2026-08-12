"""Allow ВПК category and dotted sub-codes in competencies.

Расширяем CHECK-ограничение на колонку ``competencies.category`` (добавляем
категорию ``ВПК``) и обновляем формат ``code`` в модели, чтобы принимать
под-коды вида ``УК-11.1`` / ``ПК-5.2.1``.

"""

from typing import Sequence, Union

from alembic import op


revision: str = "competency_vpk_subcodes"
down_revision: Union[str, None] = "add_rop_role_and_user_directions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE competencies DROP CONSTRAINT IF EXISTS competencies_category_check")
    op.execute(
        "ALTER TABLE competencies ADD CONSTRAINT competencies_category_check "
        "CHECK (category IN ('УК','ОПК','ПК','ППК','ИП','ВПК'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE competencies DROP CONSTRAINT IF EXISTS competencies_category_check")
    op.execute(
        "ALTER TABLE competencies ADD CONSTRAINT competencies_category_check "
        "CHECK (category IN ('УК','ОПК','ПК','ППК','ИП'))"
    )
