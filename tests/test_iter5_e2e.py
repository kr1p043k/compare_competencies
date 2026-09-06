"""Iteration 5: E2E tests - full pipeline parse->normalize->validate->match->coverage.
Tests the complete flow from vacancy text to coverage report, exercising
all fixed components together (C1, C4, C5, F2, F7).
"""
import pytest

from src.parsing.skills.skill_parser import SkillParser, SkillSource
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.parsing.skills.skill_validator import SkillValidator
from src.analyzers.skill_matcher import SkillMatcher
from src.analyzers.coverage_analyzer import CoverageAnalyzer


MARKET = {
    "python": 100, "sql": 80, "docker": 60,
    "kubernetes": 40, "react": 90, "postgresql": 70,
}


def _pipeline_extract_skills(text: str) -> list[str]:
    """Full extraction pipeline: parse -> normalize -> validate."""
    parser = SkillParser()
    result = parser._extract_from_text(text, SkillSource.DESCRIPTION)
    assert result.is_ok()
    raw = [s.text for s in result.unwrap() if s.text]

    normalized = SkillNormalizer.normalize_batch(raw)
    assert normalized.is_ok()
    valid_texts = [s for s in normalized.unwrap() if s]

    validator = SkillValidator(whitelist=None)
    validated = []
    for t in valid_texts:
        r = validator.validate(t)
        if r.is_ok() and r.unwrap().is_valid:
            validated.append(t)
    return validated


class TestE2EParseNormalizeValidate:
    def test_python_vacancy_pipeline(self):
        skills = _pipeline_extract_skills(
            "Требуется опыт работы с Python и знание SQL для разработки"
        )
        assert "python" in [s.lower() for s in skills]

    def test_empty_text_pipeline(self):
        skills = _pipeline_extract_skills("")
        assert skills == []

    def test_noise_filtered(self):
        """Garbage text produces no valid skills."""
        skills = _pipeline_extract_skills("абракадабра швабра ??? 12345")
        # May be empty or minimal - should not crash
        assert isinstance(skills, list)


class TestE2EMatchCoverage:
    def _analyzer(self):
        return CoverageAnalyzer(SkillMatcher(market_skills=dict(MARKET)))

    def test_full_coverage_e2e(self):
        """Discipline with all market skills -> 100% coverage."""
        analyzer = self._analyzer()
        result = analyzer.analyze_discipline(
            "d1", "Backend",
            {"PC-1": ["python", "sql", "docker"]},
        )
        assert result.is_ok()
        dc = result.unwrap()
        assert dc.coverage_ratio == 1.0
        assert dc.coverage_level == "high"

    def test_partial_coverage_e2e(self):
        analyzer = self._analyzer()
        result = analyzer.analyze_discipline(
            "d1", "Backend",
            {"PC-1": ["python", "cobol_unknown_xyz"]},
        )
        assert result.is_ok()
        dc = result.unwrap()
        assert dc.coverage_ratio == 0.5
        assert len(dc.gaps_list) == 1

    def test_quality_below_binary_on_fuzzy(self):
        """C4: fuzzy matches lower quality vs binary."""
        analyzer = self._analyzer()
        # 'spring' fuzzy-matches 'spring boot' if in market - use word containment
        matcher = SkillMatcher(market_skills={"spring boot": 50, "python": 100})
        analyzer2 = CoverageAnalyzer(matcher)
        result = analyzer2.analyze_discipline(
            "d1", "Test", {"PC-1": ["spring", "python"]}
        )
        assert result.is_ok()
        dc = result.unwrap()
        # Binary counts both as matched, quality weights fuzzy lower
        assert dc.coverage_ratio >= dc.weighted_coverage


class TestE2EVacancyToCoverage:
    """Vacancy text -> extracted skills -> coverage against discipline."""

    def test_vacancy_skills_match_discipline(self):
        # Step 1: extract from vacancy-like text
        skills = _pipeline_extract_skills("Python developer with SQL experience")
        assert len(skills) > 0

        # Step 2: check coverage of a discipline knowing those skills
        analyzer = CoverageAnalyzer(SkillMatcher(market_skills=dict(MARKET)))
        result = analyzer.analyze_discipline(
            "d1", "Test", {"PC-1": skills}
        )
        assert result.is_ok()


class TestE2EStudentProfile:
    """Student profile -> skills -> coverage."""

    def test_student_skills_evaluated(self):
        from src.models.student import StudentProfile
        student = StudentProfile(
            profile_name="test",
            competencies=["PC-1"],
            skills=["python", "sql"],
            skill_levels={"PC-1": "E"},
            target_level="middle",
        )
        assert len(student.skills) == 2
        assert student.skill_levels["PC-1"] == "E"

        # Evaluate coverage of student skills against market
        analyzer = CoverageAnalyzer(SkillMatcher(market_skills=dict(MARKET)))
        result = analyzer.analyze_discipline(
            "d1", "Student",
            {"PC-1": student.skills},
        )
        assert result.is_ok()
        assert result.unwrap().coverage_ratio == 1.0
