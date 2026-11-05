"""Precomputed /competency-trends cache (persists across restarts).

Эндпоинт GET /api/competency-trends считает тренды по всем компетенциям в памяти
на каждый промах (0.9-1.2 c, O(навыки x месяцы)). Результат детерминирован для
пары параметров (direction, limit) и не меняется, пока не записан новый снимок
трендов. Кладём его в БД, чтобы:
  * после рестарта процесса не пересчитывать (in-memory кэш ``_CT_CACHE`` пуст);
  * несколько процессов/воркеров делили один прогретый результат.

Инвалидация (запись нового снимка) чистит таблицу через src.cache_bus, а
дополнительная страховка - TTL по computed_at в читателе кэша.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "trend_precomputed"
down_revision: str | None = "add_admin_totp"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS trend_precomputed (
            cache_key   TEXT PRIMARY KEY,
            payload     JSONB NOT NULL,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS trend_precomputed")
