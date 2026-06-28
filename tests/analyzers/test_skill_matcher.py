from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.analyzers.skill_matcher import SkillMatcher, normalize, coverage_level
from src.errors import MatchingError


class TestNormalize:
    def test_normalize_lowercase_strip(self):
        assert normalize("  Python  ") == "python"

    def test_normalize_removes_punctuation(self):
        assert normalize("C++") == "c"
        assert normalize("C#") == "c"
        assert normalize("F#") == "f"

    def test_normalize_preserves_spaces(self):
        assert normalize("Machine Learning") == "machine learning"
        assert normalize("Deep Learning") == "deep learning"

    def test_normalize_empty(self):
        assert normalize("") == ""
        assert normalize("   ") == ""


class TestCoverageLevel:
    def test_high(self):
        assert coverage_level(0.5) == "high"
        assert coverage_level(0.75) == "high"
        assert coverage_level(1.0) == "high"

    def test_medium(self):
        assert coverage_level(0.2) == "medium"
        assert coverage_level(0.3) == "medium"
        assert coverage_level(0.49) == "medium"

    def test_low(self):
        assert coverage_level(0.0) == "low"
        assert coverage_level(0.1) == "low"
        assert coverage_level(0.19) == "low"


class TestSkillMatcher:
    def test_init_empty(self):
        m = SkillMatcher()
        assert m.market_skills == {}

    def test_init_with_data(self):
        m = SkillMatcher({"python": 100})
        assert m.market_skills == {"python": 100}

    def test_set_market_ok(self):
        m = SkillMatcher()
        result = m.set_market({"python": 50, "sql": 30})
        assert result.is_ok()
        assert m.market_skills == {"python": 50, "sql": 30}

    def test_set_market_empty(self):
        m = SkillMatcher()
        result = m.set_market({})
        assert result.is_err()

    def test_match_exact(self):
        m = SkillMatcher({"python": 100})
        result = m.match("Python")
        assert result.is_ok()
        skill, mtype, conf = result.unwrap()
        assert skill == "python"
        assert mtype == "exact"
        assert conf == 1.0

    def test_match_fuzzy(self):
        m = SkillMatcher({"sql": 100})
        result = m.match("ms sql server")
        assert result.is_ok()
        skill, mtype, conf = result.unwrap()
        assert skill == "sql"
        assert mtype == "fuzzy"
        assert conf == 0.5

    def test_match_no_match(self):
        m = SkillMatcher({"python": 100})
        result = m.match("quantum computing")
        assert result.is_ok()
        skill, mtype, conf = result.unwrap()
        assert skill is None
        assert mtype == "no_match"

    def test_match_too_short(self):
        m = SkillMatcher({"go": 50})
        result = m.match("go")
        assert result.is_ok()
        skill, mtype, conf = result.unwrap()
        assert skill is None
        assert mtype == "no_match"

    def test_get_emerging_ok(self):
        m = SkillMatcher({"python": 100, "docker": 80, "sql": 60})
        result = m.get_emerging({"python"}, top_n=5)
        assert result.is_ok()
        emerging = result.unwrap()
        names = {s for s, _, _ in emerging}
        assert "python" not in names
        assert "docker" in names
        assert "sql" in names

    def test_get_emerging_no_market(self):
        m = SkillMatcher()
        result = m.get_emerging({"python"})
        assert result.is_err()
        assert "No market skills" in result.err().message

    def test_get_emerging_with_exclude(self):
        m = SkillMatcher({"python": 100, "docker": 80, "sql": 60})
        result = m.get_emerging({"python"}, also_exclude={"docker"}, top_n=5)
        assert result.is_ok()
        names = {s for s, _, _ in result.unwrap()}
        assert "docker" not in names
        assert "sql" in names

    def test_word_match_static(self):
        assert SkillMatcher._word_match("go", "golang") is False
        assert SkillMatcher._word_match("sql", "sql server") is True
        assert SkillMatcher._word_match("ml", "machine learning") is False

    def test_fuzzy_via_word_match(self):
        m = SkillMatcher({"sql": 100})
        result = m.match("MS SQL Server")
        assert result.is_ok()
        skill, mtype, conf = result.unwrap()
        assert skill == "sql"
        assert mtype == "fuzzy"
        assert conf == 0.5
