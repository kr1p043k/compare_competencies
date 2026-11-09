"""drop legacy dead tables and columns

Удаляем подтверждённо мёртвые объекты схемы (см. docs/DB-SCHEMA-LEGACY.md):
таблицы saved_files, timelog, todos и колонки competency_skills.is_flat,
disciplines.direction_codes. Проверка перед удалением: git grep по src/tests/
alembic/scripts/frontend - 0 ссылок на каждое имя.

Живые legacy-объекты НЕ трогаем: market_skill_mappings и admin_totp.last_totp_step
(используются), а также disciplines._user_direction_codes.
"""

from __future__ import annotations

from alembic import op

revision = "drop_legacy_dead_objects"
down_revision: str | None = "add_llm_cost"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS saved_files")
    op.execute("DROP TABLE IF EXISTS timelog")
    op.execute("DROP TABLE IF EXISTS todos")
    op.execute("ALTER TABLE competency_skills DROP COLUMN IF EXISTS is_flat")
    op.execute("ALTER TABLE disciplines DROP COLUMN IF EXISTS direction_codes")


def downgrade() -> None:
    # Восстанавливаем структуру (данные безвозвратно удалены - объекты признаны мёртвыми).
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_files (
            id SERIAL PRIMARY KEY,
            user_id UUID,
            file_name TEXT,
            file_path TEXT,
            caption TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS timelog (
            id SERIAL PRIMARY KEY,
            user_id UUID,
            task_name TEXT,
            start_time TIMESTAMPTZ,
            end_time TIMESTAMPTZ,
            duration INTERVAL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS todos (
            id SERIAL PRIMARY KEY,
            user_id UUID,
            title TEXT,
            description TEXT,
            done BOOLEAN DEFAULT false,
            due_date TIMESTAMPTZ,
            remind_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        "ALTER TABLE competency_skills ADD COLUMN IF NOT EXISTS is_flat BOOLEAN NOT NULL DEFAULT false"
    )
    op.execute(
        "ALTER TABLE disciplines ADD COLUMN IF NOT EXISTS direction_codes JSONB NOT NULL DEFAULT '{}'::jsonb"
    )
