"""Iteration 3: coverage_analyzer + skill_matcher tests.
Covers match types (exact/fuzzy/semantic), coverage vs weighted,
emerging skills, cross-references, parent-child aggregation.
"""
import pytest

from src.analyzers.skill_matcher import (
    SkillMatcher,
    normalize,
    coverage_level,
    SEMANTIC_THRESHOLD,
)
from src.analyzers.coverage_analyzer import CoverageAnalyzer


class TestNormalize:
    def test_lowercase(self):
        assert normalize("Python") == "python"

    def test_strip(self):
        assert normalize("  python  ") == "python"

    def test_remove_special(self):
        assert normalize("c++") == "c"


class TestCoverageLevel:
    def test_high(self):
        assert coverage_level(0.5) == "high"
        assert coverage_level(0.9) == "high"

    def test_medium(self):
        assert coverage_level(0.2) == "medium"
        assert coverage_level(0.49) == "medium"

    def test_low(self):
        assert coverage_level(0.19) == "low"
        assert coverage_level(0.0) == "low"


class TestSkillMatcherExact:
    def test_exact_match(self):
        m = SkillMatcher(market_skills={"python": 100, "sql": 50})
        result = m.match("Python")
        assert result.is_ok()
        matched, mtype, conf = result.unwrap()
        assert matched == "python"
        assert mtype == "exact"
        assert conf == 1.0

    def test_no_match_empty(self):
        m = SkillMatcher(market_skills={"python": 100})
        result = m.match("cobol_xyz_unknown")
        assert result.is_ok()
        matched, mtype, conf = result.unwrap()
        # Without embedding provider, unknown -> no match
        assert matched is None or mtype in ("no_match", "fuzzy", "semantic")

    def test_too_short(self):
        m = SkillMatcher(market_skills={"python": 100})
        result = m.match("ab")
        assert result.is_ok()
        matched, _, _ = result.unwrap()
        assert matched is None

    def test_empty_market(self):
        m = SkillMatcher(market_skills={})
        result = m.set_market({})
        assert result.is_err()


class TestSkillMatcherFuzzy:
    """F1: fuzzy = word-containment (not RapidFuzz distance)."""

    def test_word_containment(self):
        """'spring' matches 'spring boot' via word containment."""
        m = SkillMatcher(market_skills={"spring boot": 100})
        result = m.match("spring")
        assert result.is_ok()
        matched, mtype, conf = result.unwrap()
        assert matched == "spring boot"
        assert mtype == "fuzzy"
        assert conf == 0.5

    def test_no_substring_false_positive(self):
        """'java' does NOT match 'javascript' (not whole word)."""
        m = SkillMatcher(market_skills={"javascript": 100})
        result = m.match("java")
        assert result.is_ok()
        matched, mtype, _ = result.unwrap()
        # Should not fuzzy-match (java is not whole word in javascript)
        if matched is not None:
            assert mtype != "fuzzy" or matched != "javascript"

    def test_match_cache(self):
        """Second call uses cache."""
        m = SkillMatcher(market_skills={"python": 100})
        r1 = m.match("python")
        r2 = m.match("python")
        assert r1.unwrap() == r2.unwrap()


class TestSkillMatcherEmerging:
    def test_emerging_excludes_known(self):
        m = SkillMatcher(market_skills={"python": 100, "sql": 50, "docker": 30})
        result = m.get_emerging({"python"}, top_n=10)
        assert result.is_ok()
        skills = [s for s, _, _ in result.unwrap()]
        assert "python" not in skills
        assert "sql" in skills

    def test_emerging_top_n(self):
        m = SkillMatcher(market_skills={f"skill{i}": 100 - i for i in range(20)})
        result = m.get_emerging(set(), top_n=5)
        assert result.is_ok()
        assert len(result.unwrap()) == 5

    def test_emerging_no_market(self):
        m = SkillMatcher(market_skills={})
        result = m.get_emerging(set())
        assert result.is_err()


class TestCoverageAnalyzer:
    def _make_analyzer(self):
        matcher = SkillMatcher(market_skills={
            "python": 100, "sql": 80, "docker": 60,
            "kubernetes": 40, "react": 90,
        })
        return CoverageAnalyzer(matcher)

    def test_missing_discipline_id(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze_discipline("", "Test", {"PC-1": ["python"]})
        assert result.is_err()

    def test_no_competencies(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze_discipline("1", "Test", {})
        assert result.is_err()

    def test_full_coverage(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze_discipline(
            "1", "Test", {"PC-1": ["python", "sql"]}
        )
        assert result.is_ok()
        dc = result.unwrap()
        assert dc.coverage_ratio == 1.0
        assert dc.gaps == 0
        assert dc.market_matched == 2

    def test_zero_coverage(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze_discipline(
            "1", "Test", {"PC-1": ["cobol_xyz", "fortran_xyz"]}
        )
        assert result.is_ok()
        dc = result.unwrap()
        assert dc.coverage_ratio == 0.0
        assert dc.gaps == 2

    def test_weighted_vs_binary(self):
        """C4: weighted_coverage <= coverage_ratio (fuzzy matches lower quality)."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze_discipline(
            "1", "Test", {"PC-1": ["python", "cobol_xyz"]}
        )
        assert result.is_ok()
        dc = result.unwrap()
        assert dc.coverage_ratio == 0.5  # 1/2 binary
        assert dc.weighted_coverage <= dc.coverage_ratio

    def test_parent_child_aggregation(self):
        """Parent competency with no skills aggregates from children."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze_discipline(
            "1", "Test",
            {"PC-1": [], "PC-1.1": ["python"], "PC-1.2": ["sql"]},
        )
        assert result.is_ok()
        dc = result.unwrap()
        # Parent PC-1 should aggregate children
        parent = next((c for c in dc.competencies if c.code == "PC-1"), None)
        assert parent is not None
        assert parent.total_skills == 2

    def test_short_skills_skipped(self):
        """Skills < 3 chars are skipped in discipline-level matching."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze_discipline(
            "1", "Test", {"PC-1": ["ab", "python"]}
        )
        assert result.is_ok()
        dc = result.unwrap()
        assert dc.total_skills == 1  # only python counted
