"""Initial KRM schema: directions, disciplines, competencies, KSA, skills, analysis.

Revision ID: 001
Revises:
Create Date: 2026-06-04
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── Directions ──────────────────────────────────────────────────────
    op.create_table(
        "directions",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("code", sa.String(20), unique=True, nullable=False, index=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("profile", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Disciplines ─────────────────────────────────────────────────────
    op.create_table(
        "disciplines",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("direction_id", sa.UUID, sa.ForeignKey("directions.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("name_en", sa.Text, nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── PDF Sources ─────────────────────────────────────────────────────
    op.create_table(
        "pdf_sources",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("discipline_id", sa.UUID, sa.ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("ocr_used", sa.Boolean, server_default=sa.false()),
        sa.Column("parse_status", sa.String(20), server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("parsed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_pdf_status", "pdf_sources", "parse_status IN ('pending', 'parsed', 'failed', 'ocr_done')")

    # ── Parse Versions ──────────────────────────────────────────────────
    op.create_table(
        "parse_versions",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("total_disciplines", sa.Integer, server_default="0"),
        sa.Column("total_competencies", sa.Integer, server_default="0"),
        sa.Column("total_skills", sa.Integer, server_default="0"),
        sa.Column("total_ksa_items", sa.Integer, server_default="0"),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Competencies ────────────────────────────────────────────────────
    op.create_table(
        "competencies",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("discipline_id", sa.UUID, sa.ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("code", sa.String(20), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("parent_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=True),
        sa.Column("sort_order", sa.Integer, server_default="0"),
        sa.Column("parse_version_id", sa.UUID, sa.ForeignKey("parse_versions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_comp_code_format", "competencies", "code ~ '^(УК|ОПК|ПК|ППК|ИП)[- ]\\d+'")

    # ── KSA Entries ─────────────────────────────────────────────────────
    op.create_table(
        "ksa_entries",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("competency_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("ksa_type", sa.String(20), nullable=False),
        sa.Column("original_text", sa.Text, nullable=False),
        sa.Column("cleaned_text", sa.Text, nullable=True),
        sa.Column("parse_version_id", sa.UUID, sa.ForeignKey("parse_versions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_ksa_type", "ksa_entries", "ksa_type IN ('knowledge', 'abilities', 'skills')")

    # ── Skills Taxonomy ─────────────────────────────────────────────────
    op.create_table(
        "skills",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("name", sa.Text, unique=True, nullable=False, index=True),
        sa.Column("name_en", sa.Text, nullable=True),
        sa.Column("source", sa.String(20), server_default="it_skills"),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("is_active", sa.Boolean, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_skill_source", "skills", "source IN ('it_skills', 'rpd_skills', 'market')")

    # ── Competency ↔ Skill ──────────────────────────────────────────────
    op.create_table(
        "competency_skills",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("competency_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("skill_id", sa.UUID, sa.ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("ksa_type", sa.String(20), nullable=False),
        sa.Column("source_text", sa.Text, nullable=True),
        sa.Column("match_type", sa.String(20), server_default="fuzzy"),
        sa.Column("parse_version_id", sa.UUID, sa.ForeignKey("parse_versions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_cs_ksa_type", "competency_skills", "ksa_type IN ('knowledge', 'abilities', 'skills', 'flat')")
    op.create_check_constraint("ck_cs_match_type", "competency_skills", "match_type IN ('exact', 'fuzzy', 'stem')")

    # ── Recommendations ─────────────────────────────────────────────────
    op.create_table(
        "recommendations",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("discipline_id", sa.UUID, sa.ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("competency_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=True),
        sa.Column("suggestion", sa.Text, nullable=False),
        sa.Column("suggestion_type", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_rec_type", "recommendations", "suggestion_type IN ('modify', 'add', 'remove')")

    # ── Market Skill Mappings ───────────────────────────────────────────
    op.create_table(
        "market_skill_mappings",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("skill_id", sa.UUID, sa.ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("market_skill_name", sa.Text, nullable=False),
        sa.Column("frequency", sa.Integer, server_default="0"),
        sa.Column("weight", sa.Float, server_default="0.0"),
        sa.Column("period", sa.Date, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Coverage Analyses ───────────────────────────────────────────────
    op.create_table(
        "coverage_analyses",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("discipline_id", sa.UUID, sa.ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("competency_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=True),
        sa.Column("total_skills", sa.Integer, server_default="0"),
        sa.Column("market_matched_skills", sa.Integer, server_default="0"),
        sa.Column("coverage_ratio", sa.Float, server_default="0.0"),
        sa.Column("analysis_date", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("coverage_analyses")
    op.drop_table("market_skill_mappings")
    op.drop_table("recommendations")
    op.drop_table("competency_skills")
    op.drop_table("skills")
    op.drop_table("ksa_entries")
    op.drop_table("competencies")
    op.drop_table("parse_versions")
    op.drop_table("pdf_sources")
    op.drop_table("disciplines")
    op.drop_table("directions")
