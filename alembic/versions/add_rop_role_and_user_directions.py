"""Добавляет роль rop и таблицу user_directions.

- Расширяет CHECK users.role до ('admin','teacher','student','rop').
- Создаёт user_directions (привязка РОП к dir_code направлений).
- Переводит существующих преподавателей-РОП в роль rop и предзаполняет привязки.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "add_rop_role_and_user_directions"
down_revision: Union[str, None] = "update_direction_supervisors"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ALL_DIR_CODES = [
    "01.03.01", "01.03.02", "02.03.02_och", "02.03.02_oz", "02.03.03",
    "09.03.01_ai_och", "09.03.01_ai_zaoch", "09.03.01_bim", "09.03.01_embedded",
    "09.03.01_prog_och", "09.03.01_prog_zaoch", "09.03.02", "09.03.04",
]

ROPS = [
    ("asviridov@sfedu.ru", ALL_DIR_CODES),
    ("skucherov@sfedu.ru", ALL_DIR_CODES),
    ("khusainov@sfedu.ru", [
        "09.03.02", "09.03.04", "09.03.01_ai_och", "09.03.01_ai_zaoch",
        "09.03.01_prog_och", "09.03.01_prog_zaoch",
    ]),
    ("karapetyants@sfedu.ru", ["01.03.01"]),
    ("kavatulyan@sfedu.ru", ["01.03.01"]),
    ("vvmakhno@sfedu.ru", ["01.03.02"]),
    ("miks@sfedu.ru", ["02.03.02_och", "02.03.02_oz"]),
]


def upgrade() -> None:
    # ─── 1. Расширить CHECK на users.role ──────────────────────────────────
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check")
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS ck_user_role")
    op.execute(
        "ALTER TABLE users ADD CONSTRAINT ck_user_role "
        "CHECK (role IN ('admin','teacher','student','rop'))"
    )

    # ─── 2. Таблица user_directions ────────────────────────────────────────
    op.create_table(
        "user_directions",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dir_code", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("idx_user_directions_user", "user_directions", ["user_id"])
    op.create_unique_constraint(
        "uq_user_directions_user_dir", "user_directions", ["user_id", "dir_code"]
    )

    # ─── 3. Перевести преподавателей-РОП в роль rop и предзаполнить привязки ──
    for email, dir_codes in ROPS:
        op.execute(
            f"UPDATE users SET role = 'rop' WHERE email = '{email}'"
        )
        for code in dir_codes:
            op.execute(f"""
                INSERT INTO user_directions (user_id, dir_code)
                SELECT id, '{code}' FROM users WHERE email = '{email}'
            """)


def downgrade() -> None:
    op.drop_table("user_directions")
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS ck_user_role")
    op.execute(
        "ALTER TABLE users ADD CONSTRAINT ck_user_role "
        "CHECK (role IN ('admin','teacher','student'))"
    )
