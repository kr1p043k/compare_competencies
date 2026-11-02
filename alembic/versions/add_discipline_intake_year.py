"""Intake year for disciplines: 2024 and 2025 curricula coexist in one DB.

До этого набор задавался единственным набором строк на направление, поэтому
дисциплины 2024 и 2025 с одинаковыми именами конфликтовали на
uq_disciplines_direction_name (direction_id, name). Добавляем колонку year
(NOT NULL DEFAULT 2025 — существующие строки становятся набором 2025 без
отдельного бэкфилла) и заменяем уникальность на (direction_id, name, year).
Третий набор (2027) схему больше не меняет.

Ревизия идёт за текущим head (login_bruteforce_guard).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "add_discipline_intake_year"
down_revision: str | None = "login_bruteforce_guard"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # DEFAULT 2025 переводит все существующие строки в набор 2025.
    op.execute(
        "ALTER TABLE disciplines "
        "ADD COLUMN IF NOT EXISTS year INTEGER NOT NULL DEFAULT 2025"
    )
    # Старая уникальность по (direction_id, name) не даст завести 2024-ю
    # дисциплину с тем же именем, что и 2025-я, поэтому её убираем и заменяем
    # на трёхколоночную.
    op.execute("DROP INDEX IF EXISTS uq_disciplines_direction_name")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_disciplines_direction_name_year "
        "ON disciplines (direction_id, name, year)"
    )


def downgrade() -> None:
    # Порядок важен: колонку year сносим ПЕРВОЙ, иначе восстановление старой
    # уникальности (direction_id, name) упадёт на одноимённых строках 2024/2025.
    op.execute("DROP INDEX IF EXISTS uq_disciplines_direction_name_year")
    op.execute("ALTER TABLE disciplines DROP COLUMN IF EXISTS year")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_disciplines_direction_name "
        "ON disciplines (direction_id, name)"
    )
