"""ORM-модели для таблиц, которые исторически создавались сырым SQL.

Нужны, чтобы `alembic check`/autogenerate видел их в `Base.metadata` и не
показывал ложный дрейф «removed table» (backlog 6.3.2). DDL-источник —
соответствующие миграции (и `src/api_pkg/student_actions.py` для student_actions).
"""
from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


def _ts(**kw):
    """timestamp with time zone column (сокращает длинные объявления)."""
    return mapped_column(sa.DateTime(timezone=True), **kw)


class LlmCache(Base):
    __tablename__ = "llm_cache"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    task: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(sa.CHAR(64), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = _ts(nullable=False, server_default=sa.text("NOW()"))


class SkillEmbeddingCache(Base):
    __tablename__ = "skill_embedding_cache"

    skill_key: Mapped[str] = mapped_column(Text, primary_key=True)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    updated_at: Mapped[datetime] = _ts(nullable=False, server_default=sa.text("NOW()"))


class AdminTotp(Base):
    __tablename__ = "admin_totp"

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    secret: Mapped[str] = mapped_column(Text, nullable=False)
    confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sa.text("false"))
    created_at: Mapped[datetime] = _ts(nullable=False, server_default=sa.text("now()"))
    last_used_at: Mapped[datetime | None] = _ts()
    recovery_codes: Mapped[dict | None] = mapped_column(JSONB)
    failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sa.text("0"))
    locked_until: Mapped[datetime | None] = _ts()


class TotpToken(Base):
    __tablename__ = "totp_tokens"

    token_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = _ts(nullable=False)
    created_at: Mapped[datetime] = _ts(nullable=False, server_default=sa.text("now()"))

    __table_args__ = (
        sa.Index("idx_totp_tokens_expires", "expires_at"),
        sa.CheckConstraint("purpose IN ('login_challenge', 'enroll')", name="ck_totp_tokens_purpose"),
    )


class TrendPrecomputed(Base):
    __tablename__ = "trend_precomputed"

    cache_key: Mapped[str] = mapped_column(Text, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    computed_at: Mapped[datetime] = _ts(nullable=False, server_default=sa.text("now()"))


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"))
    updated_at: Mapped[datetime] = _ts(nullable=False, server_default=sa.text("now()"))
    updated_by: Mapped[str | None] = mapped_column(UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"))


class DisciplineScope(Base):
    __tablename__ = "discipline_scope"

    direction_code: Mapped[str] = mapped_column(Text, primary_key=True)
    discipline_name: Mapped[str] = mapped_column(Text, primary_key=True)
    included: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sa.text("TRUE"))
    updated_at: Mapped[datetime] = _ts(nullable=False, server_default=sa.text("NOW()"))


class StudentAction(Base):
    __tablename__ = "student_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(Text, nullable=False)
    action_type: Mapped[str] = mapped_column(Text, nullable=False, server_default=sa.text("''"))
    profession: Mapped[str | None] = mapped_column(Text, server_default=sa.text("''"))
    region: Mapped[str | None] = mapped_column(Text, server_default=sa.text("''"))
    vacancies_found: Mapped[int | None] = mapped_column(Integer, server_default=sa.text("0"))
    result_ref: Mapped[str | None] = mapped_column(Text, server_default=sa.text("''"))
    profile: Mapped[str | None] = mapped_column(Text, server_default=sa.text("''"))
    timestamp: Mapped[datetime | None] = _ts(server_default=sa.text("NOW()"))
