"""Add vacancies.source (hh | rabota) + backfill by id offset.

Пайплайн теперь может собирать вакансии с двух агрегаторов: hh.ru и
rabota.ru. Чтобы на фронте показать метку источника и дать фильтр, нужна
явная колонка. Ранее источник жил только в raw JSONB.

Backfill: id Работы.ру сдвинуты на RABOTA_ID_OFFSET = 1e9, поэтому старые
строки с hh_id >= 1000000000 помечаются как rabota, остальные - hh.

Revision ID: vacancy_source
Revises: create_student_actions
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "vacancy_source"
down_revision: str | None = "create_student_actions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ID_OFFSET = 1_000_000_000


def upgrade() -> None:
    op.execute(
        "ALTER TABLE vacancies ADD COLUMN IF NOT EXISTS source VARCHAR(20) NOT NULL DEFAULT 'hh'"
    )
    op.execute(
        f"UPDATE vacancies SET source = 'rabota' WHERE hh_id >= {ID_OFFSET} AND source = 'hh'"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_vacancies_source ON vacancies (source)")
    op.execute("""
        DO $$ BEGIN
            ALTER TABLE vacancies ADD CONSTRAINT ck_vacancies_source
                CHECK (source IN ('hh', 'rabota'));
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)


def downgrade() -> None:
    op.execute("""
        DO $$ BEGIN
            ALTER TABLE vacancies DROP CONSTRAINT IF EXISTS ck_vacancies_source;
        EXCEPTION WHEN undefined_object THEN NULL;
        END $$;
    """)
    op.execute("DROP INDEX IF EXISTS idx_vacancies_source")
    op.execute("ALTER TABLE vacancies DROP COLUMN IF EXISTS source")
