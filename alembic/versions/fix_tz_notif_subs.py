"""make notifications/subscriptions timestamps timezone-aware (6.3.2 drift)

notifications.created_at, subscriptions.created_at/updated_at были созданы
как na?ve TIMESTAMP, тогда как модели и весь остальной проект используют
timestamp WITH time zone. Приводим БД к конвенции, чтобы `alembic check` не
показывал ложный дрейф типа.

Revision ID: fix_tz_notif_subs
Revises: drop_legacy_dead_objects
"""
from __future__ import annotations

from alembic import op

revision = "fix_tz_notif_subs"
down_revision = "drop_legacy_dead_objects"
branch_labels = None
depends_on = None

_ALTERS = [
    ("notifications", "created_at"),
    ("subscriptions", "created_at"),
    ("subscriptions", "updated_at"),
]


def upgrade() -> None:
    for table, column in _ALTERS:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} "
            f"TYPE timestamptz USING {column} AT TIME ZONE 'UTC'"
        )


def downgrade() -> None:
    for table, column in _ALTERS:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} "
            f"TYPE timestamp USING {column} AT TIME ZONE 'UTC'"
        )
