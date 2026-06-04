"""Initial schema: directions, disciplines, competencies, KSA, skills,
students, users, market analysis.

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
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── Helper: updated_at trigger function ─────────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
        $$ LANGUAGE plpgsql
    """)
    # ── Helper: normalize skill name ────────────────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION normalize_skill_name()
        RETURNS TRIGGER AS $$
        BEGIN NEW.name = LOWER(TRIM(NEW.name)); RETURN NEW; END;
        $$ LANGUAGE plpgsql
    """)
    # ── Helper: auto-build competency code ──────────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION auto_build_competency_code()
        RETURNS TRIGGER AS $$
        BEGIN IF NEW.code IS NULL OR NEW.code = '' THEN NEW.code := NEW.category || '-' || NEW.number; END IF; RETURN NEW; END;
        $$ LANGUAGE plpgsql
    """)
    # ── Helper: upsert student skill ────────────────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION upsert_student_skill()
        RETURNS TRIGGER AS $$
        BEGIN
            IF EXISTS (SELECT 1 FROM student_skills WHERE student_id = NEW.student_id AND skill_id = NEW.skill_id AND source = NEW.source) THEN
                UPDATE student_skills SET proficiency = NEW.proficiency, achieved_level = COALESCE(NEW.achieved_level, achieved_level), assessed_at = NOW() WHERE student_id = NEW.student_id AND skill_id = NEW.skill_id AND source = NEW.source;
                RETURN NULL;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)

    # ── Directions ──────────────────────────────────────────────────────
    op.create_table(
        "directions",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("code", sa.String(20), unique=True, nullable=False, index=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("profile", sa.Text, nullable=True),
        sa.Column("opop_year", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Disciplines ─────────────────────────────────────────────────────
    op.create_table(
        "disciplines",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("direction_id", sa.UUID, sa.ForeignKey("directions.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("name_en", sa.Text, nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("semester", sa.Integer, nullable=True),
        sa.Column("hours_total", sa.Integer, nullable=True),
        sa.Column("hours_lecture", sa.Integer, nullable=True),
        sa.Column("hours_practice", sa.Integer, nullable=True),
        sa.Column("hours_lab", sa.Integer, nullable=True),
        sa.Column("hours_self", sa.Integer, nullable=True),
        sa.Column("control_form", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── PDF Sources ─────────────────────────────────────────────────────
    op.create_table(
        "pdf_sources",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
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
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("direction_id", sa.UUID, sa.ForeignKey("directions.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("opop_year", sa.Integer, nullable=True),
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
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("discipline_id", sa.UUID, sa.ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("code", sa.String(20), nullable=False),
        sa.Column("category", sa.String(10), nullable=False),
        sa.Column("number", sa.String(10), nullable=False),
        sa.Column("name", sa.Text, nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("development_level", sa.String(10), nullable=True),
        sa.Column("parent_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=True),
        sa.Column("sort_order", sa.Integer, server_default="0"),
        sa.Column("parse_version_id", sa.UUID, sa.ForeignKey("parse_versions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_comp_code_format", "competencies", "code ~ '^(УК|ОПК|ПК|ППК|ИП)[- ]\\d+'")
    op.create_check_constraint("ck_comp_category", "competencies", "category IN ('УК', 'ОПК', 'ПК', 'ППК', 'ИП')")
    op.create_check_constraint("ck_comp_dev_level", "competencies", "development_level IN ('КС-1', 'КС-2', 'КС-3')")
    op.create_index("idx_competencies_category", "competencies", ["category"])

    # ── KSA Entries ─────────────────────────────────────────────────────
    op.create_table(
        "ksa_entries",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("competency_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("ksa_type", sa.String(20), nullable=False),
        sa.Column("original_text", sa.Text, nullable=False),
        sa.Column("cleaned_text", sa.Text, nullable=True),
        sa.Column("sort_order", sa.Integer, server_default="0"),
        sa.Column("parse_version_id", sa.UUID, sa.ForeignKey("parse_versions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_ksa_type", "ksa_entries", "ksa_type IN ('knowledge', 'abilities', 'skills')")

    # ── Skills Taxonomy ─────────────────────────────────────────────────
    op.create_table(
        "skills",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("name", sa.Text, nullable=False, index=True),
        sa.Column("name_en", sa.Text, nullable=True),
        sa.Column("description", sa.Text, nullable=True),
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
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("competency_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("skill_id", sa.UUID, sa.ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("ksa_type", sa.String(20), nullable=False),
        sa.Column("source_text", sa.Text, nullable=True),
        sa.Column("match_type", sa.String(20), server_default="fuzzy"),
        sa.Column("required_level", sa.String(10), nullable=True),
        sa.Column("parse_version_id", sa.UUID, sa.ForeignKey("parse_versions.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_cs_ksa_type", "competency_skills", "ksa_type IN ('knowledge', 'abilities', 'skills', 'flat')")
    op.create_check_constraint("ck_cs_match_type", "competency_skills", "match_type IN ('exact', 'fuzzy', 'stem')")
    op.create_unique_constraint("uq_cs_unique", "competency_skills", ["competency_id", "skill_id", "ksa_type", "parse_version_id"])

    # ── Users ────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("full_name", sa.Text, nullable=False),
        sa.Column("role", sa.String(20), server_default="teacher"),
        sa.Column("is_active", sa.Boolean, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_user_role", "users", "role IN ('admin', 'teacher')")

    # ── Recommendations ─────────────────────────────────────────────────
    op.create_table(
        "recommendations",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("discipline_id", sa.UUID, sa.ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("competency_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=True),
        sa.Column("user_id", sa.UUID, sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("suggestion", sa.Text, nullable=False),
        sa.Column("suggestion_type", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_rec_type", "recommendations", "suggestion_type IN ('modify', 'add', 'remove')")

    # ── Student Groups ──────────────────────────────────────────────────
    op.create_table(
        "student_groups",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("direction_id", sa.UUID, sa.ForeignKey("directions.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_unique_constraint("uq_student_group_name", "student_groups", ["name"])

    # ── Students ─────────────────────────────────────────────────────────
    op.create_table(
        "students",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("group_id", sa.UUID, sa.ForeignKey("student_groups.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("full_name", sa.Text, nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Student Skills ──────────────────────────────────────────────────
    op.create_table(
        "student_skills",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("student_id", sa.UUID, sa.ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("skill_id", sa.UUID, sa.ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("source", sa.String(30), server_default="self_assessment"),
        sa.Column("proficiency", sa.Float, server_default="0.0"),
        sa.Column("achieved_level", sa.String(10), nullable=True),
        sa.Column("assessed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_check_constraint("ck_ss_source", "student_skills", "source IN ('self_assessment', 'auto_extracted', 'expert', 'test')")
    op.create_check_constraint("ck_ss_proficiency", "student_skills", "proficiency >= 0.0 AND proficiency <= 1.0")
    op.create_check_constraint("ck_ss_achieved_level", "student_skills", "achieved_level IN ('КС-1', 'КС-2', 'КС-3')")
    op.create_unique_constraint("uq_student_skill_source", "student_skills", ["student_id", "skill_id", "source"])

    # ── Market Skill Mappings ───────────────────────────────────────────
    op.create_table(
        "market_skill_mappings",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("skill_id", sa.UUID, sa.ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("market_skill_name", sa.Text, nullable=False),
        sa.Column("frequency", sa.Integer, server_default="0"),
        sa.Column("weight", sa.Float, server_default="0.0"),
        sa.Column("period", sa.Date, nullable=True),
        sa.Column("source", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Coverage Analyses ───────────────────────────────────────────────
    op.create_table(
        "coverage_analyses",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("discipline_id", sa.UUID, sa.ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("competency_id", sa.UUID, sa.ForeignKey("competencies.id", ondelete="CASCADE"), nullable=True),
        sa.Column("total_skills", sa.Integer, server_default="0"),
        sa.Column("market_matched_skills", sa.Integer, server_default="0"),
        sa.Column("coverage_ratio", sa.Float, server_default="0.0"),
        sa.Column("analysis_date", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Triggers on table events ────────────────────────────────────────
    tables_updated_at = [
        "directions", "disciplines", "competencies", "skills", "users",
    ]
    for t in tables_updated_at:
        op.execute(f"CREATE TRIGGER trg_{t}_updated BEFORE UPDATE ON {t} FOR EACH ROW EXECUTE FUNCTION update_updated_at()")

    op.execute("CREATE TRIGGER trg_skills_name_lower BEFORE INSERT OR UPDATE ON skills FOR EACH ROW EXECUTE FUNCTION normalize_skill_name()")
    op.execute("CREATE TRIGGER trg_competencies_code_auto BEFORE INSERT ON competencies FOR EACH ROW EXECUTE FUNCTION auto_build_competency_code()")
    op.execute("CREATE TRIGGER trg_student_skills_upsert BEFORE INSERT ON student_skills FOR EACH ROW EXECUTE FUNCTION upsert_student_skill()")


def downgrade() -> None:
    # Drop triggers first
    for suffix in ["updated", "name_lower", "code_auto", "upsert"]:
        for t in ["directions", "disciplines", "competencies", "skills", "users"]:
            op.execute(f"DROP TRIGGER IF EXISTS trg_{t}_{suffix} ON {t} CASCADE")
    op.execute("DROP TRIGGER IF EXISTS trg_student_skills_upsert ON student_skills CASCADE")

    op.drop_table("coverage_analyses")
    op.drop_table("market_skill_mappings")
    op.drop_table("student_skills")
    op.drop_table("students")
    op.drop_table("student_groups")
    op.drop_table("recommendations")
    op.drop_table("users")
    op.drop_table("competency_skills")
    op.drop_table("skills")
    op.drop_table("ksa_entries")
    op.drop_table("competencies")
    op.drop_table("parse_versions")
    op.drop_table("pdf_sources")
    op.drop_table("disciplines")
    op.drop_table("directions")
