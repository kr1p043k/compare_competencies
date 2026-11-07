"""Склеить все головы alembic в одну.

История миграций разветвилась: разные ветки работы добавляли независимые
ревизии от разных предков. С несколькими головами `alembic upgrade head`
падает, поэтому добавляем пустую merge-ревизию. Схему она не меняет.
"""

from collections.abc import Sequence

revision: str = "merge_all_heads"
# Только настоящие головы перед склейкой. Перечислять здесь ещё и их предков
# нельзя: alembic пытается снять с набора голов ревизию, которая головой уже
# не является, и падает с KeyError при `upgrade`.
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
