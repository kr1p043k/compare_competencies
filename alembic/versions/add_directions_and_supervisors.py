"""Добавляет supervisor в directions, 6 направлений, 8 преподавателей.

Убирает unique на code (нужно для 2x 09.03.01),
добавляет unique (code, profile),
добавляет колонку supervisor.
Обновляет 09.03.02: name = 'Перспективные информационные технологии'.
Добавляет 6 новых направлений.
Создаёт/обновляет 8 пользователей-преподавателей.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "add_directions_and_supervisors"
down_revision: Union[str, None] = "add_llm_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


DIRECTIONS = [
    ("01.03.01", "Математика, механика и математическое моделирование",
     "Математика, механика и математическое моделирование", "Карапетянц А.Н."),
    ("01.03.02", "Математическое моделирование и искусственный интеллект",
     "Математическое моделирование и искусственный интеллект", "Махно В.В."),
    ("09.03.04", "Методы и средства разработки программного обеспечения",
     "Методы и средства разработки программного обеспечения", "Хусаинов Н.Ш."),
    ("09.03.02", "Перспективные информационные технологии",
     "Перспективные информационные технологии", "Хусаинов Н.Ш."),
    ("09.03.01", "Программирование и системная интеграция ИТ-решений",
     "Программирование и системная интеграция ИТ-решений", "Хусаинов Н.Ш."),
    ("09.03.01", "Технологии искусственного интеллекта",
     "Технологии искусственного интеллекта", "Хусаинов Н.Ш."),
    ("02.03.02", "Фундаментальная информатика и информационные технологии",
     "Фундаментальная информатика и информационные технологии", "Михалкович С.С."),
]

TEACHERS = [
    ("karapetyants@sfedu.ru", "Карапетянц Алексей Николаевич"),
    ("kavatulyan@sfedu.ru", "Ватульян Карина Александровна"),
    ("vvmakhno@sfedu.ru", "Махно Виктория Викторовна"),
    ("khusainov@sfedu.ru", "Хусаинов Наиль Шавяктович"),
    ("miks@sfedu.ru", "Михалкович Станислав Станиславович"),
    ("asviridov@sfedu.ru", "Свиридов Александр Славьевич"),
    ("skucherov@sfedu.ru", "Кучеров Сергей Александрович"),
]


def upgrade() -> None:
    # ─── 1. Убрать старый unique на code, добавить (code, profile) ────────────
    op.execute("""
        DO $$ BEGIN
            ALTER TABLE directions DROP CONSTRAINT IF EXISTS directions_code_key;
        EXCEPTION WHEN undefined_object THEN NULL;
        END $$;
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_directions_code_profile
        ON directions(code, profile)
    """)

    # ─── 2. Добавить колонку supervisor ───────────────────────────────────────
    op.execute("ALTER TABLE directions ADD COLUMN IF NOT EXISTS supervisor VARCHAR(255)")

    # ─── 3. Обновить 09.03.02 ─────────────────────────────────────────────────
    op.execute("""
        UPDATE directions
        SET name = 'Перспективные информационные технологии',
            profile = 'Перспективные информационные технологии'
        WHERE code = '09.03.02'
          AND (name != 'Перспективные информационные технологии'
               OR profile IS NULL
               OR profile != 'Перспективные информационные технологии')
    """)

    # ─── 4. Добавить/обновить направления ─────────────────────────────────────
    for code, name, profile, supervisor in DIRECTIONS:
        name_safe = name.replace("'", "''")
        profile_safe = profile.replace("'", "''")
        supervisor_safe = (supervisor or "").replace("'", "''")
        op.execute(f"""
            INSERT INTO directions (code, name, profile, supervisor, opop_year)
            VALUES ('{code}', '{name_safe}', '{profile_safe}', '{supervisor_safe}', 2025)
            ON CONFLICT (code, profile) DO UPDATE SET
                name = EXCLUDED.name,
                supervisor = EXCLUDED.supervisor
        """)

    # ─── 5. Удалить старые teacher-аккаунты (@edu → @sfedu.ru) ──────────────
    OLD_EMAILS = [
        "karapetyants.an@edu",
        "vatulyan.ka@edu",
        "mahno.vv@edu",
        "husainov.nsh@edu",
        "mihalkovich.ss@edu",
    ]
    for old in OLD_EMAILS:
        op.execute(f"DELETE FROM users WHERE email = '{old}'")

    # ─── 6. Добавить новых преподавателей ────────────────────────────────────
    for email, full_name in TEACHERS:
        name_safe = full_name.replace("'", "''")
        op.execute(f"""
            INSERT INTO users (email, password_hash, full_name, role)
            VALUES ('{email}', crypt('teacher123', gen_salt('bf')), '{name_safe}', 'teacher')
            ON CONFLICT (email) DO UPDATE SET
                full_name = EXCLUDED.full_name,
                role = 'teacher',
                is_active = TRUE
        """)


def downgrade() -> None:
    for email, _ in TEACHERS:
        op.execute(f"DELETE FROM users WHERE email = '{email}'")
    for code, _, profile, _ in DIRECTIONS:
        op.execute(f"DELETE FROM directions WHERE code = '{code}' AND profile = '{profile}'")
    op.execute("ALTER TABLE directions DROP COLUMN IF EXISTS supervisor")
    op.execute("DROP INDEX IF EXISTS uq_directions_code_profile")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS directions_code_key ON directions(code)")
