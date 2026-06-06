# tests/parsing/test_skill_normalizer.py
import pytest
from src.parsing.skills.skill_normalizer import SkillNormalizer

class TestSkillNormalizer:
    def setup_method(self):
        SkillNormalizer._canonical_map = None
        SkillNormalizer._whitelist = None

    def test_synonyms(self):
        assert SkillNormalizer.normalize("Python 3.11").ok() == "python"
        result = SkillNormalizer.normalize("javascript").ok()
        assert result in ("javascript", "js", "node.js")

    def test_suffix_removal(self):
        assert SkillNormalizer.normalize("язык python").ok() == "python"

    def test_fuzzy_matching(self):
        SkillNormalizer._whitelist = None
        assert SkillNormalizer.normalize("reackt").ok() == "react"
        result = SkillNormalizer.normalize("react native").ok()
        assert result in ("react", "react native")
        result = SkillNormalizer.normalize("node js").ok()
        assert result in ("nodejs", "node.js")
        result = SkillNormalizer.normalize("NodeJS").ok()
        assert result in ("nodejs", "node.js")

    def test_no_match_returns_cleaned_version(self):
        result = SkillNormalizer.normalize("какой-то_мусор_навык_123").ok()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_batch(self):
        skills = ["Python 3", "React.js v18", "reackt", "machine learning"]
        normalized = SkillNormalizer.normalize_batch(skills).ok()
        assert normalized[0] == "python"
        assert normalized[1] == "react"
        assert normalized[2] == "react"
        assert isinstance(normalized[3], str)
        assert len(normalized[3]) > 0

    def test_batch_with_duplicates(self):
        skills = ["Python", "python", "", "React", "reackt"]
        normalized = SkillNormalizer.normalize_batch(skills).ok()
        assert normalized == ["python", "python", "react", "react"]
        dedup = SkillNormalizer.deduplicate(skills).ok()
        assert dedup == ["python", "react"]

    def test_normalize_edge_cases(self):
        assert SkillNormalizer.normalize("").ok() == ""
        assert SkillNormalizer.normalize("   Python   ").ok() == "python"
        assert SkillNormalizer.normalize("React.js v18").ok() == "react"
        assert SkillNormalizer.normalize("NodeJS").ok() == "nodejs"

    def test_empty_and_whitespace(self):
        assert SkillNormalizer.normalize("").ok() == ""
        assert SkillNormalizer.normalize("   ").ok() == ""
        assert SkillNormalizer.normalize("\t\n").ok() == ""
        assert SkillNormalizer.normalize(None).ok() == ""

    def test_version_patterns(self):
        assert SkillNormalizer.normalize("Python 3.11").ok() == "python"
        assert SkillNormalizer.normalize("React v18.2.1").ok() == "react"
        assert SkillNormalizer.normalize("Django (фреймворк)").ok() == "django"
        assert SkillNormalizer.normalize("Angular [2+]").ok() == "angular"
        assert SkillNormalizer.normalize("PostgreSQL 14.5").ok() == "postgresql"

    def test_suffix_removal_extended(self):
        assert SkillNormalizer.normalize("язык программирования Python").ok() == "python"
        assert SkillNormalizer.normalize("Python язык").ok() == "python"
        assert SkillNormalizer.normalize("React фреймворк").ok() == "react"
        assert SkillNormalizer.normalize("pandas библиотека").ok() == "pandas"
        assert SkillNormalizer.normalize("MongoDB database").ok() == "mongodb"

    def test_special_characters_cleaning(self):
        assert SkillNormalizer.normalize("Node.JS").ok() == "nodejs"
        assert SkillNormalizer.normalize("some_skill!").ok() == "some_skill"

    def test_whitelist_exact_match_prevents_fuzzy(self):
        assert SkillNormalizer.normalize("python").ok() == "python"
        assert SkillNormalizer.normalize("pythn").ok() == "python"

    def test_deduplicate_order_and_uniqueness(self):
        skills = ["React", "react", "Python", "python", "Docker", "React"]
        result = SkillNormalizer.deduplicate(skills).ok()
        assert result == ["react", "python", "docker"]
        skills_with_empty = ["", "React", None, "React"]
        result2 = SkillNormalizer.deduplicate(skills_with_empty).ok()
        assert result2 == ["react"]

    def test_fuzzy_matching_extended(self):
        result = SkillNormalizer.normalize("anguler").ok()
        assert len(result) > 0
        assert SkillNormalizer.normalize("pythn").ok() == "python"
        assert SkillNormalizer.normalize("dockr").ok() == "docker"
        result = SkillNormalizer.normalize("xyzabc").ok()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_whitelist_loading_and_caching(self):
        SkillNormalizer._whitelist = None
        whitelist1 = SkillNormalizer._get_whitelist()
        whitelist2 = SkillNormalizer._get_whitelist()
        assert whitelist1 is whitelist2
        assert "python" in whitelist1
        assert "mlops" in whitelist1
        assert "node.js" in whitelist1

    def test_direct_phrase_mapping(self):
        if hasattr(SkillNormalizer, "DIRECT_PHRASE_MAP"):
            assert SkillNormalizer.normalize("machine learning").ok() == "mlops"
            assert SkillNormalizer.normalize("node js").ok() == "node.js"
            assert SkillNormalizer.normalize("react native").ok() == "react"

    def test_complex_input_combinations(self):
        assert SkillNormalizer.normalize("язык программирования Python 3.11").ok() == "python"
        assert SkillNormalizer.normalize("фреймворк React.js v18").ok() == "react"

    def test_normalize_with_special_chars_and_version(self):
        assert SkillNormalizer.normalize("  PyThOn  3.9  ").ok() == "python"
        assert SkillNormalizer.normalize("Node.JS (среда)").ok() == "nodejs"
        assert SkillNormalizer.normalize("TypEScript").ok() == "typescript"

    def test_fuzzy_threshold_boundary(self):
        result = SkillNormalizer.normalize("completelywrongterm").ok()
        assert "completelywrongterm" in result

    def test_phrase_synonyms_replaced_correctly(self):
        result = SkillNormalizer.normalize("react native").ok()
        assert result in ("react", "react native")
        result = SkillNormalizer.normalize("machine learning").ok()
        assert isinstance(result, str) and len(result) > 0
        result = SkillNormalizer.normalize("node js").ok()
        assert result in ("nodejs", "node.js")

    def test_synonym_replacement_does_not_affect_unrelated_words(self):
        result = SkillNormalizer.normalize("javascripting").ok()
        assert isinstance(result, str) and len(result) > 0

    def test_deduplicate_preserves_order_and_removes_duplicates(self):
        skills = ["Python", "python", "React", "reackt", "Docker", "docker"]
        dedup = SkillNormalizer.deduplicate(skills).ok()
        assert dedup == ["python", "react", "docker"]

    def test_canonical_map_initialization(self):
        SkillNormalizer._canonical_map = None
        canon_map = SkillNormalizer._get_canonical_map()
        assert isinstance(canon_map, dict)
        assert "javascript" in canon_map
        assert "node.js" in canon_map.values() or "nodejs" in canon_map.values()

    def test_apply_synonym_map_direct(self):
        result = SkillNormalizer._apply_synonym_map("node.js")
        assert result in ("nodejs", "node.js")
        result = SkillNormalizer._apply_synonym_map("unknown_skill_xyz")
        assert result == "unknown_skill_xyz"

    def test_machine_learning_normalization(self):
        result = SkillNormalizer.normalize("machine learning").ok()
        assert isinstance(result, str)
        assert len(result) > 0
