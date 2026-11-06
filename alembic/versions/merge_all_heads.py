"""Склеить все головы alembic в одну.

История миграций разветвилась: разные ветки работы добавляли независимые
ревизии от разных предков. С несколькими головами `alembic upgrade head`
падает, поэтому добавляем пустую merge-ревизию. Схему она не меняет.
"""

from collections.abc import Sequence

revision: str = "merge_all_heads"
down_revision: tuple[str, ...] = (
    "add_audit_logs",
    "add_discipline_intake_year",
    "add_sessions_sso_token",
    "add_subs_and_notifs",
    "discipline_scope_table",
    "fix_pipeline_cancelled",
    "merge_for_nikita_main",
    "pipeline_runs_owner_dir",
    "remove_non_it_disciplines",
    "retention_fk_on_delete",
    "vacancies_published_nulls_last",
    "vacancy_employer_logo",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Ничего: ревизия только склеивает ветки истории."""


def downgrade() -> None:
    """Ничего: разъединение веток не поддерживается."""
