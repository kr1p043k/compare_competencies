"""add admin_totp.last_totp_step (6.3.2 drift)

Колонка объявлена в модели `src/models/extra_models.py` (`AdminTotp.last_totp_step`),
но миграцией не создавалась: на базе, поднятой из миграций, `alembic check`
показывал `add_column admin_totp.last_totp_step`. Хранит последний использованный
шаг TOTP, чтобы блокировать повторное использование кода (replay).

Идемпотентно: ADD COLUMN IF NOT EXISTS, поэтому на проде, где колонка могла
появиться вручную, миграция безопасна.

Revision ID: add_admin_totp_last_step
Revises: create_student_actions
"""
from __future__ import annotations

from alembic import op

revision = "add_admin_totp_last_step"
down_revision = "create_student_actions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE admin_totp ADD COLUMN IF NOT EXISTS last_totp_step BIGINT"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE admin_totp DROP COLUMN IF EXISTS last_totp_step")
