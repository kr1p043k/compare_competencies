"""Add employer_logo to vacancies (hh employer logo_urls, 240px preferred).

Backfill from raw->employer->logo_urls is done by script (not in migration):
  logo = logo_urls.240 or logo_urls.90 or logo_urls.original
"""

from typing import Sequence, Union

from alembic import op

revision: str = "vacancy_employer_logo"
down_revision: Union[str, None] = "pipeline_idempotency_key"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE vacancies ADD COLUMN IF NOT EXISTS employer_logo TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE vacancies DROP COLUMN IF EXISTS employer_logo")
