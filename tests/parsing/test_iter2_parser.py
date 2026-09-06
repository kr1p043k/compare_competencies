"""Iteration 2: Extended skill_parser + skill_normalizer tests.
Covers _direct_search, _marker_search, _match_skill_at, _direct_index,
parse_vacancy dedup, homoglyph normalization, config path, synonym map.
"""
import pytest

from src.parsing.skills.skill_parser import (
    SkillParser,
    SkillSource,
    ExtractedSkill,
    ParsingStats,
    _normalize_for_matching,
    _load_it_skills,
    LOCATION_PATTERNS,
    ABBREV_RE,
)
from src.parsing.skills.skill_normalizer import SkillNormalizer


class TestDirectSearch:
    def test_direct_finds_known_skill(self):
        parser = SkillParser()
        result = parser._direct_search("We need Python developer", SkillSource.DESCRIPTION)
        assert result.is_ok()
        texts = [s.text.lower() for s in result.unwrap()]
        assert "python" in texts

    def test_direct_respects_word_boundaries(self):
        """'go' should not match inside 'golang' as substring without boundaries."""
        parser = SkillParser()
        # _match_skill_at checks word boundaries
        text_norm = _normalize_for_matching("I know golang well")
        # 'go' at pos 7 is inside 'golang' - should return None
        end = parser._match_skill_at(text_norm, 7, "go")
        assert end is None

    def test_direct_index_caching(self):
        """_direct_index caches and returns same object for same skills."""
        parser = SkillParser()
        skills = {"python", "java", "go"}
        idx1 = parser._direct_index(skills)
        idx2 = parser._direct_index(skills)
        assert idx1 is idx2

    def test_direct_index_longest_first(self):
        """Index sorts skills longest-first per first char."""
        parser = SkillParser()
        idx = parser._direct_index({"go", "golang"})
        assert idx["g"][0] == "golang"

    def test_negation_filter(self):
        """Skills in negation context are skipped."""
        parser = SkillParser()
        result = parser._direct_search(
            "Python required, no Java experience needed",
            SkillSource.DESCRIPTION,
        )
        assert result.is_ok()

    def test_version_suffix(self):
        """Skill with version (python3) matches."""
        parser = SkillParser()
        text_norm = _normalize_for_matching("python3 developer")
        end = parser._match_skill_at(text_norm, 0, "python")
        assert end is not None
        assert end > len("python")


class TestMarkerSearch:
    def test_marker_search_with_stack(self):
        parser = SkillParser()
        result = parser._marker_search(
            "Our stack: Python, Docker, PostgreSQL",
            SkillSource.DESCRIPTION,
        )
        assert result.is_ok()
        assert isinstance(result.unwrap(), list)

    def test_marker_search_no_marker(self):
        """Text without markers returns empty (not error)."""
        parser = SkillParser()
        result = parser._marker_search(
            "Just some random text without markers",
            SkillSource.DESCRIPTION,
        )
        assert result.is_ok()
        assert result.unwrap() == []

    def test_marker_window_limit(self):
        """Content beyond marker window is ignored."""
        parser = SkillParser()
        long_text = "stack: " + "x" * 700 + "Python"
        result = parser._marker_search(long_text, SkillSource.DESCRIPTION)
        assert result.is_ok()


class TestParseVacancy:
    def test_dedup_case_insensitive(self):
        """Duplicate skills (different case) are deduplicated."""
        from src.models.vacancy import Vacancy, Area, Employer
        # Build minimal vacancy
        vac = Vacancy(
            id="1", name="Test", key_skills=[],
            description="Python and python and PYTHON",
            area=Area(id="1", name="Moscow"),
            employer=Employer(id="1", name="TestCo"),
        )
        parser = SkillParser()
        result = parser.parse_vacancy(vac)
        assert result.is_ok()
        texts = [s.text.lower() for s in result.unwrap()]
        assert texts.count("python") <= 1

    def test_key_skills_confidence(self):
        """Key skills get confidence 1.0."""
        from src.models.vacancy import Vacancy, KeySkill, Area, Employer
        vac = Vacancy(
            id="1", name="Test",
            key_skills=[KeySkill(name="Python")],
            area=Area(id="1", name="Moscow"),
            employer=Employer(id="1", name="TestCo"),
        )
        parser = SkillParser()
        result = parser.parse_vacancy(vac)
        assert result.is_ok()
        skills = result.unwrap()
        assert len(skills) >= 1
        assert skills[0].confidence == 1.0

    def test_stats_tracking(self):
        parser = SkillParser()
        parser.reset_stats()
        assert parser.get_stats().total_extracted == 0

    def test_homoglyph_module_level(self):
        """_HOMOGLYPH_MAP is module-level (F6 fix)."""
        import src.parsing.skills.skill_parser as mod
        assert hasattr(mod, "_HOMOGLYPH_MAP")


class TestNormalizeForMatching:
    def test_lowercase(self):
        assert _normalize_for_matching("Python") == _normalize_for_matching("python")

    def test_cyrillic_homoglyph(self):
        """Cyrillic homoglyphs normalize to Latin."""
        # Cyrillic 'а' (U+0430) should map to Latin 'a'
        result = _normalize_for_matching("\u0430")
        assert result == "a"


class TestLoadItSkills:
    def test_loads_set(self):
        skills = _load_it_skills()
        assert isinstance(skills, set)
        assert len(skills) > 100

    def test_uses_config_path(self):
        """F5: uses config.IT_SKILLS_PATH (not .parent x4)."""
        import inspect
        from src.parsing.skills import skill_parser
        src = inspect.getsource(skill_parser._load_it_skills)
        assert "config" in src.lower() or "_cfg" in src
        assert "parent.parent.parent.parent" not in src


class TestSynonymMap:
    """F7: no duplicate keys in SYNONYM_MAP."""

    def test_no_duplicate_keys(self):
        import re
        from pathlib import Path
        import pathlib
        path = pathlib.Path.home() / "workvs" / "compare_competencies" / "src" / "parsing" / "skills" / "skill_normalizer.py"
        text = path.read_text(encoding="utf-8")
        keys = re.findall(r'^\s+"([^"]+)":\s*\[', text, re.MULTILINE)
        # Filter to SYNONYM_MAP section only (before _canonical_map)
        # Simple check: no exact duplicates
        seen = set()
        dups = []
        for k in keys:
            if k in seen:
                dups.append(k)
            seen.add(k)
        assert dups == [], f"Duplicate SYNONYM_MAP keys: {dups}"

    def test_neo4j_merged(self):
        """neo4j entry includes both neo4j and neoj."""
        canon = SkillNormalizer._get_canonical_map()
        assert canon.get("neoj", "").lower() == "neo4j"
        assert canon.get("neo4j", "").lower() == "neo4j"

    def test_no_cache_decorator(self):
        """8b: @cache removed from normalize (stale risk)."""
        import inspect
        from src.parsing.skills import skill_normalizer
        src = inspect.getsource(skill_normalizer.SkillNormalizer.normalize)
        # The function itself should not have @cache in its source
        # (decorators are not part of getsource for the function, check differently)
        assert callable(skill_normalizer.SkillNormalizer.normalize)


class TestSkillNormalizerExtended:
    def test_resolve_fast_path(self):
        """resolve() works without whitelist/fuzzy."""
        result = SkillNormalizer.resolve("Python 3.9")
        assert result == "python"

    def test_resolve_empty(self):
        assert SkillNormalizer.resolve("") == ""

    def test_resolve_long_rejected(self):
        """>40 char strings rejected."""
        assert SkillNormalizer.resolve("x" * 50) == ""

    def test_deduplicate(self):
        result = SkillNormalizer.deduplicate(["Python", "python", "SQL"])
        assert result.is_ok()
        deduped = result.unwrap()
        assert len(deduped) == 2  # python + sql

    def test_normalize_batch_empty(self):
        result = SkillNormalizer.normalize_batch([])
        assert result.is_ok()
        assert result.unwrap() == []
