"""SQLAlchemy models for main project database."""

import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


# ─── Direction ─────────────────────────────────────────────────────────────

class Direction(Base):
    __tablename__ = "directions"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    profile: Mapped[Optional[str]] = mapped_column(Text)
    opop_year: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    disciplines: Mapped[list["Discipline"]] = relationship(back_populates="direction", cascade="all, delete-orphan")
    parse_versions: Mapped[list["ParseVersion"]] = relationship(back_populates="direction", cascade="all, delete-orphan")
    student_groups: Mapped[list["StudentGroup"]] = relationship(back_populates="direction", cascade="all, delete-orphan")


# ─── Discipline ────────────────────────────────────────────────────────────

class Discipline(Base):
    __tablename__ = "disciplines"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    direction_id: Mapped[str] = mapped_column(UUID, ForeignKey("directions.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_en: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    semester: Mapped[Optional[int]] = mapped_column(Integer)
    hours_total: Mapped[Optional[int]] = mapped_column(Integer)
    hours_lecture: Mapped[Optional[int]] = mapped_column(Integer)
    hours_practice: Mapped[Optional[int]] = mapped_column(Integer)
    hours_lab: Mapped[Optional[int]] = mapped_column(Integer)
    hours_self: Mapped[Optional[int]] = mapped_column(Integer)
    control_form: Mapped[Optional[str]] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    direction: Mapped["Direction"] = relationship(back_populates="disciplines")
    pdf_sources: Mapped[list["PDFSource"]] = relationship(back_populates="discipline", cascade="all, delete-orphan")
    competencies: Mapped[list["Competency"]] = relationship(back_populates="discipline", cascade="all, delete-orphan")
    recommendations: Mapped[list["Recommendation"]] = relationship(back_populates="discipline", cascade="all, delete-orphan")
    coverage_analyses: Mapped[list["CoverageAnalysis"]] = relationship(back_populates="discipline", cascade="all, delete-orphan")


# ─── PDF Source ────────────────────────────────────────────────────────────

class PDFSource(Base):
    __tablename__ = "pdf_sources"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    discipline_id: Mapped[str] = mapped_column(UUID, ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    ocr_used: Mapped[bool] = mapped_column(default=False)
    parse_status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    parsed_at: Mapped[Optional[datetime]]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    discipline: Mapped["Discipline"] = relationship(back_populates="pdf_sources")

    __table_args__ = (
        CheckConstraint(parse_status.in_(["pending", "parsed", "failed", "ocr_done"]), name="ck_pdf_status"),
    )


# ─── Parse Version ─────────────────────────────────────────────────────────

class ParseVersion(Base):
    __tablename__ = "parse_versions"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    direction_id: Mapped[str] = mapped_column(UUID, ForeignKey("directions.id", ondelete="CASCADE"), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    opop_year: Mapped[Optional[int]] = mapped_column(Integer)
    total_disciplines: Mapped[int] = mapped_column(Integer, default=0)
    total_competencies: Mapped[int] = mapped_column(Integer, default=0)
    total_skills: Mapped[int] = mapped_column(Integer, default=0)
    total_ksa_items: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    direction: Mapped["Direction"] = relationship(back_populates="parse_versions")
    competencies: Mapped[list["Competency"]] = relationship(back_populates="parse_version")
    ksa_entries: Mapped[list["KSAEntry"]] = relationship(back_populates="parse_version")
    competency_skills: Mapped[list["CompetencySkill"]] = relationship(back_populates="parse_version")


# ─── Competency ────────────────────────────────────────────────────────────

class Competency(Base):
    __tablename__ = "competencies"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    discipline_id: Mapped[str] = mapped_column(UUID, ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(20), nullable=False)
    category: Mapped[str] = mapped_column(String(10), nullable=False)
    number: Mapped[str] = mapped_column(String(10), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    development_level: Mapped[Optional[str]] = mapped_column(String(10))
    parent_id: Mapped[Optional[str]] = mapped_column(UUID, ForeignKey("competencies.id", ondelete="CASCADE"))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    parse_version_id: Mapped[Optional[str]] = mapped_column(UUID, ForeignKey("parse_versions.id"))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    discipline: Mapped["Discipline"] = relationship(back_populates="competencies")
    parent: Mapped[Optional["Competency"]] = relationship(
        back_populates="children", remote_side="Competency.id"
    )
    children: Mapped[list["Competency"]] = relationship(back_populates="parent", cascade="all, delete-orphan")
    parse_version: Mapped[Optional["ParseVersion"]] = relationship(back_populates="competencies")
    ksa_entries: Mapped[list["KSAEntry"]] = relationship(back_populates="competency", cascade="all, delete-orphan")
    competency_skills: Mapped[list["CompetencySkill"]] = relationship(back_populates="competency", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("code ~ '^(УК|ОПК|ПК|ППК|ИП)[- ]\\d+'", name="ck_comp_code_format"),
        CheckConstraint(category.in_(["УК", "ОПК", "ПК", "ППК", "ИП"]), name="ck_comp_category"),
        CheckConstraint(development_level.in_(["КС-1", "КС-2", "КС-3"]), name="ck_comp_dev_level"),
    )


# ─── KSA Entry ─────────────────────────────────────────────────────────────

class KSAEntry(Base):
    __tablename__ = "ksa_entries"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    competency_id: Mapped[str] = mapped_column(UUID, ForeignKey("competencies.id", ondelete="CASCADE"), nullable=False, index=True)
    ksa_type: Mapped[str] = mapped_column(String(20), nullable=False)
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[Optional[str]] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    parse_version_id: Mapped[Optional[str]] = mapped_column(UUID, ForeignKey("parse_versions.id"))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    competency: Mapped["Competency"] = relationship(back_populates="ksa_entries")
    parse_version: Mapped[Optional["ParseVersion"]] = relationship(back_populates="ksa_entries")

    __table_args__ = (
        CheckConstraint(ksa_type.in_(["knowledge", "abilities", "skills"]), name="ck_ksa_type"),
    )


# ─── Skill ─────────────────────────────────────────────────────────────────

class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    name_en: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(20), default="it_skills")
    category: Mapped[Optional[str]] = mapped_column(String(100))
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384))
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    competency_skills: Mapped[list["CompetencySkill"]] = relationship(back_populates="skill", cascade="all, delete-orphan")
    market_mappings: Mapped[list["MarketSkillMapping"]] = relationship(back_populates="skill", cascade="all, delete-orphan")
    student_skills: Mapped[list["StudentSkill"]] = relationship(back_populates="skill", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(source.in_(["it_skills", "rpd_skills", "market"]), name="ck_skill_source"),
    )


# ─── Competency ↔ Skill ────────────────────────────────────────────────────

class CompetencySkill(Base):
    __tablename__ = "competency_skills"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    competency_id: Mapped[str] = mapped_column(UUID, ForeignKey("competencies.id", ondelete="CASCADE"), nullable=False, index=True)
    skill_id: Mapped[str] = mapped_column(UUID, ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True)
    ksa_type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_text: Mapped[Optional[str]] = mapped_column(Text)
    match_type: Mapped[str] = mapped_column(String(20), default="fuzzy")
    parse_version_id: Mapped[Optional[str]] = mapped_column(UUID, ForeignKey("parse_versions.id"))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    competency: Mapped["Competency"] = relationship(back_populates="competency_skills")
    skill: Mapped["Skill"] = relationship(back_populates="competency_skills")
    parse_version: Mapped[Optional["ParseVersion"]] = relationship(back_populates="competency_skills")

    __table_args__ = (
        CheckConstraint(ksa_type.in_(["knowledge", "abilities", "skills", "flat"]), name="ck_cs_ksa_type"),
        CheckConstraint(match_type.in_(["exact", "fuzzy", "stem"]), name="ck_cs_match_type"),
        UniqueConstraint("competency_id", "skill_id", "ksa_type", "parse_version_id", name="uq_cs_unique"),
    )


# ─── User ──────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="teacher")
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    recommendations: Mapped[list["Recommendation"]] = relationship(back_populates="user")

    __table_args__ = (
        CheckConstraint(role.in_(["admin", "teacher"]), name="ck_user_role"),
    )


# ─── Recommendation ────────────────────────────────────────────────────────

class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    discipline_id: Mapped[str] = mapped_column(UUID, ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True)
    competency_id: Mapped[Optional[str]] = mapped_column(UUID, ForeignKey("competencies.id", ondelete="CASCADE"))
    user_id: Mapped[Optional[str]] = mapped_column(UUID, ForeignKey("users.id", ondelete="SET NULL"))
    suggestion: Mapped[str] = mapped_column(Text, nullable=False)
    suggestion_type: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    discipline: Mapped["Discipline"] = relationship(back_populates="recommendations")
    user: Mapped[Optional["User"]] = relationship(back_populates="recommendations")

    __table_args__ = (
        CheckConstraint(suggestion_type.in_(["modify", "add", "remove"]), name="ck_rec_type"),
    )


# ─── Student Group ─────────────────────────────────────────────────────────

class StudentGroup(Base):
    __tablename__ = "student_groups"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    direction_id: Mapped[str] = mapped_column(UUID, ForeignKey("directions.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    direction: Mapped["Direction"] = relationship(back_populates="student_groups")
    students: Mapped[list["Student"]] = relationship(back_populates="group", cascade="all, delete-orphan")


# ─── Student ───────────────────────────────────────────────────────────────

class Student(Base):
    __tablename__ = "students"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    group_id: Mapped[str] = mapped_column(UUID, ForeignKey("student_groups.id", ondelete="CASCADE"), nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    group: Mapped["StudentGroup"] = relationship(back_populates="students")
    skills: Mapped[list["StudentSkill"]] = relationship(back_populates="student", cascade="all, delete-orphan")


# ─── Student Skill ─────────────────────────────────────────────────────────

class StudentSkill(Base):
    __tablename__ = "student_skills"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    student_id: Mapped[str] = mapped_column(UUID, ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    skill_id: Mapped[str] = mapped_column(UUID, ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(30), default="self_assessment")
    proficiency: Mapped[float] = mapped_column(Float, default=0.0)
    assessed_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    student: Mapped["Student"] = relationship(back_populates="skills")
    skill: Mapped["Skill"] = relationship(back_populates="student_skills")

    __table_args__ = (
        CheckConstraint(source.in_(["self_assessment", "auto_extracted", "expert", "test"]), name="ck_ss_source"),
        CheckConstraint("proficiency >= 0.0 AND proficiency <= 1.0", name="ck_ss_proficiency"),
        UniqueConstraint("student_id", "skill_id", "source", name="uq_student_skill_source"),
    )


# ─── Market Skill Mapping ──────────────────────────────────────────────────

class MarketSkillMapping(Base):
    __tablename__ = "market_skill_mappings"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    skill_id: Mapped[str] = mapped_column(UUID, ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True)
    market_skill_name: Mapped[str] = mapped_column(Text, nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, default=0)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    period: Mapped[Optional[datetime]] = mapped_column()
    source: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    skill: Mapped["Skill"] = relationship(back_populates="market_mappings")


# ─── Coverage Analysis ─────────────────────────────────────────────────────

class CoverageAnalysis(Base):
    __tablename__ = "coverage_analyses"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=_uuid)
    discipline_id: Mapped[str] = mapped_column(UUID, ForeignKey("disciplines.id", ondelete="CASCADE"), nullable=False, index=True)
    competency_id: Mapped[Optional[str]] = mapped_column(UUID, ForeignKey("competencies.id", ondelete="CASCADE"))
    total_skills: Mapped[int] = mapped_column(Integer, default=0)
    market_matched_skills: Mapped[int] = mapped_column(Integer, default=0)
    coverage_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    analysis_date: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    discipline: Mapped["Discipline"] = relationship(back_populates="coverage_analyses")
