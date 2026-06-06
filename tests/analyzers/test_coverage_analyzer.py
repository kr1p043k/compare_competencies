from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.analyzers.coverage_analyzer import CoverageAnalyzer
from src.analyzers.skill_matcher import SkillMatcher, normalize as norm_fn
from src.errors import CoverageError
from src.models.teacher_analysis import DisciplineCoverage, CompetencyCoverage


class TestCoverageAnalyzer:
    @pytest.fixture
    def matcher(self):
        m = SkillMatcher({"python": 100, "sql": 80, "docker": 60})
        return m

    @pytest.fixture
    def analyzer(self, matcher):
        return CoverageAnalyzer(matcher)

    def test_init(self, matcher):
        ca = CoverageAnalyzer(matcher)
        assert ca.matcher is matcher

    def test_analyze_discipline_empty_id(self, analyzer):
        result = analyzer.analyze_discipline("", "test", {"C1": ["python"]})
        assert result.is_err()
        err = result.err()
        assert "discipline_id is required" in err.message

    def test_analyze_discipline_empty_competencies(self, analyzer):
        result = analyzer.analyze_discipline("d1", "test", {})
        assert result.is_err()
        assert "No competencies" in result.err().message

    def test_analyze_discipline_all_matched(self, analyzer):
        result = analyzer.analyze_discipline("d1", "CS", {"C1": ["python", "sql"]})
        assert result.is_ok()
        dc: DisciplineCoverage = result.unwrap()
        assert dc.discipline_id == "d1"
        assert dc.total_skills == 2
        assert dc.market_matched == 2
        assert dc.gaps == 0
        assert dc.coverage_ratio == 1.0
        assert dc.coverage_level == "high"

    def test_analyze_discipline_with_gaps(self, analyzer):
        result = analyzer.analyze_discipline("d1", "CS", {"C1": ["python", "unknown_xyz_skill"]})
        assert result.is_ok()
        dc = result.unwrap()
        assert dc.market_matched == 1
        assert dc.gaps == 1
        assert len(dc.gaps_list) == 1
        assert dc.gaps_list[0] == "unknown_xyz_skill"

    def test_analyze_discipline_with_cross_refs(self, analyzer):
        rpd_norm = {"java", "sql"}
        skill_map = {"CS": {"python", "sql"}, "Math": {"java"}}
        result = analyzer.analyze_discipline(
            "d1", "CS", {"C1": ["python", "sql"]},
            direction_rpd_norm=rpd_norm,
            discipline_skill_map=skill_map,
        )
        assert result.is_ok()
        dc = result.unwrap()
        assert dc.discipline_id == "d1"
