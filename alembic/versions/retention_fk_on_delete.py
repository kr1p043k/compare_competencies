"""retention_fk_on_delete: llm_interactions.analysis_run_id -> ON DELETE SET NULL.

retention.py удаляет pipeline_runs в одной транзакции с request_logs и
vacancies, а llm_interactions.analysis_run_id ссылался на pipeline_runs(id)
без действия при удалении (FOREIGN KEY (analysis_run_id) REFERENCES
pipeline_runs(id)). Любая ссылающаяся строка давала ошибку внешнего ключа и
обрывала всю чистку - в итоге не удалялось ничего.

Меняем поведение на ON DELETE SET NULL, чтобы удаление прогона не падало, а
строка llm_interactions оставалась с NULL в analysis_run_id.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "retention_fk_on_delete"
down_revision: str | None = "add_admin_totp"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE llm_interactions "
        "DROP CONSTRAINT IF EXISTS llm_interactions_analysis_run_id_fkey"
    )
    op.execute(
        "ALTER TABLE llm_interactions "
        "ADD CONSTRAINT llm_interactions_analysis_run_id_fkey "
        "FOREIGN KEY (analysis_run_id) REFERENCES pipeline_runs(id) "
        "ON DELETE SET NULL"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE llm_interactions "
        "DROP CONSTRAINT IF EXISTS llm_interactions_analysis_run_id_fkey"
    )
    op.execute(
        "ALTER TABLE llm_interactions "
        "ADD CONSTRAINT llm_interactions_analysis_run_id_fkey "
        "FOREIGN KEY (analysis_run_id) REFERENCES pipeline_runs(id)"
    )
