"""Pydantic models for teacher analysis output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SkillMatch:
    skill_name: str
    frequency: int = 0
    market_match: str | None = None
    match_type: str = "exact"  # exact | fuzzy | substring


@dataclass
class CrossReference:
    skill_name: str
    frequency: int = 0
    discipline: str = ""


@dataclass
class CompetencyCoverage:
    code: str
    total_skills: int = 0
    matched_skills: int = 0
    coverage: float = 0.0
    gap_skills: list[str] = field(default_factory=list)


@dataclass
class DisciplineCoverage:
    discipline_id: str
    discipline_name: str
    total_skills: int = 0
    market_matched: int = 0
    gaps: int = 0
    coverage_ratio: float = 0.0
    coverage_level: str = "low"
    top_matched: list[SkillMatch] = field(default_factory=list)
    gaps_list: list[str] = field(default_factory=list)
    emerging: list[SkillMatch] = field(default_factory=list)
    truly_missing: list[SkillMatch] = field(default_factory=list)
    cross_references: list[CrossReference] = field(default_factory=list)
    competencies: list[CompetencyCoverage] = field(default_factory=list)


@dataclass
class Recommendation:
    type: str  # update_content | add_new_content | major_revision
    priority: str  # high | medium | low
    message: str


@dataclass
class DirectionSummary:
    direction_code: str
    direction_name: str
    profile: str
    total_disciplines: int = 0
    average_coverage: float = 0.0
    total_gaps: int = 0
    top_cross_discipline_gaps: list[dict] = field(default_factory=list)
    top_emerging: list[dict] = field(default_factory=list)
    disciplines: list[dict] = field(default_factory=list)


@dataclass
class GapAnalysisResult:
    discipline: DisciplineCoverage | None = None
    recommendations: list[Recommendation] = field(default_factory=list)


@dataclass
class TeacherAnalysisReport:
    direction: DirectionSummary
    discipline_reports: list[tuple[str, GapAnalysisResult]] = field(default_factory=list)
    generated_at: str = ""
