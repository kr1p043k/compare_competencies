"""Extended tests for CoverageAnalyzer — weighted coverage, cross-refs."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src import Ok
from src.analyzers.coverage_analyzer import CoverageAnalyzer
from src.analyzers.skill_matcher import SkillMatcher, normalize


class TestCoverageAnalyzerExtended:
    def test_analyze_all_matched_weighted(self):
        matcher = SkillMatcher({"python": 100, "sql": 50})
        matcher._market_embeddings = None
        analyzer = CoverageAnalyzer(matcher)
        comps = {"ПК-1": ["python", "sql"]}
        result = analyzer.analyze_discipline("id", "test", comps)
        assert result.is_ok()
        dc = result.ok()
        assert dc.coverage_ratio == 1.0
        assert dc.weighted_coverage == 1.0
        assert len(dc.competencies) == 1
        assert dc.competencies[0].weighted_coverage == 1.0

    def test_analyze_partial_match_weighted(self):
        matcher = SkillMatcher({"python": 100, "sql": 50})
        matcher._market_embeddings = None  # disable semantic
        analyzer = CoverageAnalyzer(matcher)
        comps = {"ПК-1": ["python", "sql", "zzz_nonexistent_skill_999"]}
        result = analyzer.analyze_discipline("id", "test", comps)
        assert result.is_ok()
        dc = result.ok()
        assert dc.coverage_ratio == pytest.approx(2 / 3, 0.01)
        assert dc.weighted_coverage == dc.coverage_ratio  # both exact matches, same value

    def test_analyze_with_cross_refs(self):
        matcher = SkillMatcher({"python": 100, "sql": 50, "docker": 200})
        analyzer = CoverageAnalyzer(matcher)
        comps = {"ПК-1": ["python", "unknown_skill_x99"]}
        rpd_norm = {"python", "sql", "docker", "unknown_skill_x99"}
        disc_skill_map = {"БД": {"sql"}, "DevOps": {"docker"}, "test": {"python"}}
        result = analyzer.analyze_discipline("id", "test", comps, rpd_norm, disc_skill_map)
        assert result.is_ok()
        dc = result.ok()
        assert len(dc.cross_references) > 0

    def test_analyze_empty_competencies(self):
        matcher = SkillMatcher()
        analyzer = CoverageAnalyzer(matcher)
        result = analyzer.analyze_discipline("id", "test", {})
        assert result.is_err()

    def test_analyze_missing_id(self):
        matcher = SkillMatcher()
        analyzer = CoverageAnalyzer(matcher)
        result = analyzer.analyze_discipline("", "test", {"ПК-1": ["python"]})
        assert result.is_err()

    def test_analyze_with_discipline_scorer(self):
        matcher = SkillMatcher({"python": 100, "sql": 50})
        scorer = MagicMock()
        scorer._ensure_embeddings = MagicMock()
        scorer.get_discipline_embedding = MagicMock(return_value=None)
        analyzer = CoverageAnalyzer(matcher, discipline_scorer=scorer)
        comps = {"ПК-1": ["python", "sql"]}
        rpd_norm = {"python", "sql"}
        disc_map = {"БД": {"sql"}}
        result = analyzer.analyze_discipline("id", "test", comps, rpd_norm, disc_map)
        assert result.is_ok()

    def test_weighted_coverage_per_competency(self):
        matcher = SkillMatcher({"python": 100, "sql": 50})
        matcher._market_embeddings = None
        analyzer = CoverageAnalyzer(matcher)
        comps = {"ПК-1": ["python", "sql", "zzz_nonexistent_1"], "ПК-2": ["python", "zzz_nonexistent_2"]}
        result = analyzer.analyze_discipline("id", "test", comps)
        assert result.is_ok()
        cc = result.ok().competencies
        assert len(cc) == 2
        assert cc[0].weighted_coverage == cc[0].coverage  # both exact matches when matched
