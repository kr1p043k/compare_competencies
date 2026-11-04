"""Index on vacancies(published_at DESC NULLS LAST) for list ordering.

Списки вакансий и экспорт сортируют `ORDER BY published_at DESC NULLS LAST`.
Прежний индекс `idx_vacancies_published` был на `(published_at DESC)`, что по
умолчанию означает NULLS FIRST, поэтому планировщик его не использовал:
первая страница шла через Parallel Seq Scan (~92 мс на 57k строк).

Замер после замены: 92 мс -> 0.46 мс (Index Scan, 51 строка, 42 буфера).

Revision ID: vacancies_published_nulls_last
Revises: merge_intake_year_main
"""

from collections.abc import Sequence

from alembic import op

revision: str = "vacancies_published_nulls_last"
down_revision: str | None = "merge_intake_year_main"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # CONCURRENTLY нельзя внутри транзакции alembic: создаём обычным CREATE,
    # индекс небольшой и строится быстро. IF NOT EXISTS - идемпотентно
    # (в рабочей БД индекс уже создан вручную).
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_vacancies_published_nl "
        "ON vacancies USING btree (published_at DESC NULLS LAST)"
    )
    # Старый индекс на (published_at DESC) стал избыточным: он не подходил под
    # NULLS LAST и только занимал место. Если его нет - DROP безопасен.
    op.execute("DROP INDEX IF EXISTS idx_vacancies_published")


def downgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_vacancies_published "
        "ON vacancies USING btree (published_at DESC)"
    )
    op.execute("DROP INDEX IF EXISTS idx_vacancies_published_nl")
