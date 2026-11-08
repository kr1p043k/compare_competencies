"""Склеить все головы alembic в одну.

История миграций разветвилась: разные ветки работы добавляли независимые
ревизии от разных предков. С несколькими головами `alembic upgrade head`
падает, поэтому добавляем пустую merge-ревизию. Схему она не меняет.
"""

from collections.abc import Sequence

revision: str = "merge_all_heads"
# Родители — только НАСТОЯЩИЕ концы веток. Прочие ревизии из прежнего
# списка (add_discipline_intake_year, merge_for_nikita_main, ...) уже
# являются предками add_audit_logs и применены в БД: перечисление их
# родителями превращало применённые ревизии в лишние точки ветвления, и
# `alembic upgrade heads` падал с KeyError в head_maintainer.
down_revision: tuple[str, ...] = (
    "add_audit_logs",
    "retention_fk_on_delete",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Ничего: ревизия только склеивает ветки истории."""


def downgrade() -> None:
    """Ничего: разъединение веток не поддерживается."""
