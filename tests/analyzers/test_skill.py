# tests/analyzers/test_skill.py
import pytest

from src import Ok, Err
from src.analyzers.skills.skill_filter import SkillFilter
from src.analyzers.skills.skill_level_analyzer import SkillLevelAnalyzer


class TestSkillFilterExtended:
    def test_filter_weights_removes_generic(self):
        sf = SkillFilter()
        weights = {"frontend": 10.0, "python": 8.0, "web": 5.0}
        filtered = sf.filter_weights(weights, min_weight=0).unwrap()
        assert "frontend" not in filtered
        assert "web" not in filtered
        assert "python" in filtered

    def test_filter_weights_min_weight(self):
        sf = SkillFilter()
        weights = {"python": 0.1, "sql": 0.005}
        filtered = sf.filter_weights(weights, min_weight=0.01).unwrap()
        assert "python" in filtered
        assert "sql" not in filtered

    def test_normalize_weights_minmax(self):
        sf = SkillFilter()
        weights = {"a": 1, "b": 5, "c": 10}
        norm = sf.normalize_weights(weights, method="minmax").unwrap()
        assert norm["c"] == 1.0
        assert norm["a"] == 0.1
        assert 0.1 < norm["b"] < 1.0

    def test_normalize_weights_log(self):
        sf = SkillFilter()
        weights = {"a": 1, "b": 100}
        norm = sf.normalize_weights(weights, method="log").unwrap()
        assert norm["b"] == 1.0
        assert norm["a"] < 1.0

    def test_merge_with_reference(self):
        sf = SkillFilter()
        sf.GENERIC_WORDS = sf.GENERIC_WORDS - {"sql"}
        skill_weights = {"python": 0.9, "django": 0.7}
        comp_freq = {"python": 50, "sql": 30, "frontend": 20}
        merged = sf.merge_with_reference(skill_weights, comp_freq).unwrap()
        assert "python" in merged
        assert "sql" in merged

    def test_get_clean_weights_full_pipeline(self):
        sf = SkillFilter()
        sf.GENERIC_WORDS = sf.GENERIC_WORDS - {"sql"}
        raw = {"python": 0.9, "frontend": 0.8, "sql": 0.5}
        freq = {"python": 100, "sql": 50}
        clean = sf.get_clean_weights(raw, freq, use_reference=True, normalize_method="minmax").unwrap()
        assert "frontend" not in clean
        assert "python" in clean
        assert clean["python"] == 1.0
        assert "sql" in clean
        assert 0 < clean["sql"] < 1.0

    def test_validate_skills(self):
        sf = SkillFilter()
        skills = ["python", "frontend", "very long skill name with many words"]
        valid = sf.validate_skills(skills).unwrap()
        assert "python" in valid
        assert "frontend" not in valid
        assert "very long skill name with many words" not in valid

    def test_get_skill_categories(self):
        sf = SkillFilter()
        skills = ["python", "react", "postgresql", "docker", "aws", "machine learning"]
        cats = sf.get_skill_categories(skills).unwrap()
        assert "programming_languages" in cats
        assert "frameworks" in cats
        assert "databases" in cats
        assert "devops" in cats
        assert "cloud" in cats
        assert "data_science" in cats

    def test_filter_weights_empty_input(self):
        sf = SkillFilter()
        result = sf.filter_weights({}).unwrap()
        assert result == {}

    def test_normalize_weights_all_equal_minmax(self):
        sf = SkillFilter()
        weights = {"a": 5.0, "b": 5.0}
        norm = sf.normalize_weights(weights, method="minmax").unwrap()
        assert norm == {"a": 1.0, "b": 1.0}

    def test_merge_with_reference_empty_comp_freq(self):
        sf = SkillFilter()
        skill_weights = {"python": 0.9, "django": 0.7}
        merged = sf.merge_with_reference(skill_weights, {}).unwrap()
        assert "python" in merged
        assert merged["python"] == 0.9

    def test_merge_with_reference_equal_counts(self):
        sf = SkillFilter()
        skill_weights = {}
        comp_freq = {"python": 10, "django": 10}
        merged = sf.merge_with_reference(skill_weights, comp_freq).unwrap()
        assert merged["python"] == 0.8
        assert merged["django"] == 0.8

    def test_merge_with_reference_adds_skills_only_in_tfidf(self):
        sf = SkillFilter()
        skill_weights = {"fastapi": 0.9}
        comp_freq = {"python": 10}
        merged = sf.merge_with_reference(skill_weights, comp_freq).unwrap()
        assert "fastapi" in merged
        assert merged["fastapi"] == 0.9

    def test_get_clean_weights_empty_raw(self):
        sf = SkillFilter()
        clean = sf.get_clean_weights({}).unwrap()
        assert clean == {}

    def test_get_clean_weights_no_competency_freq_uses_raw(self):
        sf = SkillFilter()
        raw = {"python": 0.9}
        clean = sf.get_clean_weights(raw, use_reference=False).unwrap()
        assert "python" in clean

    def test_get_clean_weights_removes_long_phrases_in_clean(self):
        sf = SkillFilter()
        raw = {"python": 0.9, "this is a very long skill phrase with six words": 0.5}
        clean = sf.get_clean_weights(raw, use_reference=False).unwrap()
        assert "python" in clean
        assert "this is a very long skill phrase with six words" not in clean

    def test_get_clean_weights_empty_after_generic_and_long(self):
        sf = SkillFilter()
        raw = {"frontend": 0.9, "this is a very long skill phrase with six words": 0.5}
        clean = sf.get_clean_weights(raw, use_reference=False).unwrap()
        assert clean == {}

    def test_get_clean_weights_unknown_reduced_not_removed(self):
        sf = SkillFilter()
        raw = {"unknownskill1": 0.9, "unknownskill2": 0.8}
        clean = sf.get_clean_weights(raw, use_reference=True).unwrap()
        assert len(clean) == 2
        assert "unknownskill1" in clean
        assert "unknownskill2" in clean
        assert clean["unknownskill1"] == 1.0
        assert 0.0 < clean["unknownskill2"] < 1.0

    def test_get_skill_categories_full(self):
        sf = SkillFilter()
        skills = [
            "python",
            "react",
            "postgresql",
            "docker",
            "aws",
            "machine learning",
            "html",
            "pytest",
            "git",
            "unknown",
        ]
        cats = sf.get_skill_categories(skills).unwrap()
        assert "programming_languages" in cats
        assert "frameworks" in cats
        assert "databases" in cats
        assert "devops" in cats
        assert "cloud" in cats
        assert "data_science" in cats
        assert "frontend" in cats
        assert "testing" in cats
        assert "other" in cats
        assert "git" in cats["devops"]

    def test_get_skill_categories_empty(self):
        sf = SkillFilter()
        cats = sf.get_skill_categories([]).unwrap()
        assert cats == {}

    def test_filter_weights_min_weight_edge(self):
        sf = SkillFilter()
        weights = {"python": 0.009, "django": 0.01}
        filtered = sf.filter_weights(weights, min_weight=0.01).unwrap()
        assert "python" not in filtered
        assert "django" in filtered

    def test_validate_skills_partial_match(self):
        sf = SkillFilter()
        skills = ["python programming", "advanced sql", "unknown skill"]
        valid = sf.validate_skills(skills).unwrap()
        assert "python programming" in valid
        assert isinstance(valid, list)


class TestSkillFilterFull:
    @pytest.fixture
    def sf(self):
        return SkillFilter()

    def test_filter_weights_removes_unknown(self, sf):
        weights = {"python": 0.5, "unknownskillxyz": 0.5}
        filtered = sf.filter_weights(weights, min_weight=0.0).unwrap()
        assert "python" in filtered
        assert "unknownskillxyz" not in filtered

    def test_filter_weights_partial_match_kept(self, sf):
        weights = {"machine learning expert": 0.5}
        filtered = sf.filter_weights(weights, min_weight=0.0).unwrap()
        assert "machine learning expert" in filtered

    def test_normalize_weights_softmax(self, sf):
        weights = {"a": 1, "b": 2}
        norm = sf.normalize_weights(weights, method="softmax").unwrap()
        total = sum(norm.values())
        assert round(total, 4) == 1.0
        assert norm["b"] > norm["a"]

    def test_normalize_weights_invalid_method(self, sf):
        weights = {"a": 1, "b": 2}
        norm = sf.normalize_weights(weights, method="invalid").unwrap()
        assert norm == weights

    def test_normalize_weights_log_max_zero(self, sf):
        weights = {"a": 0.0, "b": 0.0}
        norm = sf.normalize_weights(weights, method="log").unwrap()
        assert norm == weights

    def test_merge_with_reference_count_based(self, sf):
        skill_weights = {"python": 0.9}
        comp_freq = {"python": 50, "django": 30, "fastapi": 20}
        merged = sf.merge_with_reference(skill_weights, comp_freq).unwrap()
        assert "django" in merged
        assert 0.0 <= merged["django"] <= 1.0

    def test_get_clean_weights_after_empty(self, sf):
        raw = {"frontend": 0.9, "backend": 0.8, "web": 0.7}
        clean = sf.get_clean_weights(raw, use_reference=False).unwrap()
        assert isinstance(clean, dict)

    def test_get_clean_weights_log_normalization(self, sf):
        sf.GENERIC_WORDS = sf.GENERIC_WORDS - {"sql"}
        raw = {"python": 0.9, "sql": 0.5}
        freq = {"python": 100, "sql": 10}
        clean = sf.get_clean_weights(raw, freq, normalize_method="log").unwrap()
        assert "python" in clean
        assert "sql" in clean
        max_val = max(clean.values())
        assert max_val == 1.0


class TestSkillFilterCoverage:
    def test_filter_weights_empty_input(self):
        sf = SkillFilter()
        result = sf.filter_weights({}).unwrap()
        assert result == {}

    def test_filter_weights_keeps_valid_skill(self):
        sf = SkillFilter()
        weights = {"python": 0.5}
        filtered = sf.filter_weights(weights, min_weight=0.01).unwrap()
        assert "python" in filtered

    def test_normalize_weights_minmax_all_equal(self):
        sf = SkillFilter()
        weights = {"a": 5.0, "b": 5.0}
        norm = sf.normalize_weights(weights, method="minmax").unwrap()
        assert norm == {"a": 1.0, "b": 1.0}

    def test_merge_with_reference_empty_comp_freq(self):
        sf = SkillFilter()
        skill_weights = {"python": 0.9, "django": 0.7}
        merged = sf.merge_with_reference(skill_weights, {}).unwrap()
        assert "python" in merged
        assert merged["python"] == 0.9

    def test_merge_with_reference_all_counts_equal(self):
        sf = SkillFilter()
        skill_weights = {}
        comp_freq = {"python": 10, "django": 10}
        merged = sf.merge_with_reference(skill_weights, comp_freq).unwrap()
        assert merged["python"] == 0.8
        assert merged["django"] == 0.8

    def test_merge_with_reference_adds_skills_only_in_tfidf(self):
        sf = SkillFilter()
        skill_weights = {"fastapi": 0.9}
        comp_freq = {"python": 10}
        merged = sf.merge_with_reference(skill_weights, comp_freq).unwrap()
        assert "fastapi" in merged
        assert merged["fastapi"] == 0.9

    def test_get_clean_weights_empty_raw(self):
        sf = SkillFilter()
        clean = sf.get_clean_weights({}).unwrap()
        assert clean == {}

    def test_get_clean_weights_no_competency_freq_uses_raw(self):
        sf = SkillFilter()
        raw = {"python": 0.9}
        clean = sf.get_clean_weights(raw, use_reference=False).unwrap()
        assert "python" in clean

    def test_get_clean_weights_removes_long_phrases(self):
        sf = SkillFilter()
        raw = {"python": 0.9, "this is a very long skill phrase with six words": 0.5}
        clean = sf.get_clean_weights(raw, use_reference=False).unwrap()
        assert "python" in clean
        assert "this is a very long skill phrase with six words" not in clean

    def test_get_clean_weights_empty_after_generic_and_long(self):
        sf = SkillFilter()
        raw = {"frontend": 0.9, "this is a very long skill phrase with six words": 0.5}
        clean = sf.get_clean_weights(raw, use_reference=False).unwrap()
        assert clean == {}

    def test_get_clean_weights_unknown_reduced_not_removed(self):
        sf = SkillFilter()
        raw = {"unknownskill1": 0.9, "unknownskill2": 0.8}
        clean = sf.get_clean_weights(raw, use_reference=True).unwrap()
        assert len(clean) == 2
        assert "unknownskill1" in clean
        assert "unknownskill2" in clean

    def test_get_skill_categories_full(self):
        sf = SkillFilter()
        skills = [
            "python",
            "react",
            "postgresql",
            "docker",
            "aws",
            "machine learning",
            "html",
            "pytest",
            "git",
            "unknown",
        ]
        cats = sf.get_skill_categories(skills).unwrap()
        assert "programming_languages" in cats
        assert "frameworks" in cats
        assert "databases" in cats
        assert "devops" in cats
        assert "cloud" in cats
        assert "data_science" in cats
        assert "frontend" in cats
        assert "testing" in cats
        assert "other" in cats
        assert "python" in cats["programming_languages"]
        assert "react" in cats["frameworks"]
        assert "postgresql" in cats["databases"]
        assert "docker" in cats["devops"]
        assert "git" in cats["devops"]
        assert "aws" in cats["cloud"]
        assert "machine learning" in cats["data_science"]
        assert "html" in cats["frontend"]
        assert "pytest" in cats["testing"]
        assert "unknown" in cats["other"]

    def test_get_skill_categories_empty(self):
        sf = SkillFilter()
        cats = sf.get_skill_categories([]).unwrap()
        assert cats == {}

    def test_validate_skills_partial_match(self):
        sf = SkillFilter()
        skills = ["python programming", "advanced react", "django rest"]
        valid = sf.validate_skills(skills).unwrap()
        assert "python programming" in valid
        assert "advanced react" in valid
        assert "django rest" in valid

    def test_normalize_weights_softmax_sums_to_one(self):
        sf = SkillFilter()
        weights = {"a": 1, "b": 2, "c": 3}
        norm = sf.normalize_weights(weights, method="softmax").unwrap()
        assert abs(sum(norm.values()) - 1.0) < 0.01

    def test_normalize_weights_softmax_empty(self):
        sf = SkillFilter()
        weights = {"a": 0.0, "b": 0.0}
        norm = sf.normalize_weights(weights, method="softmax").unwrap()
        assert norm["a"] == pytest.approx(0.5)
        assert norm["b"] == pytest.approx(0.5)

    def test_merge_with_reference_count_based_fallback(self):
        sf = SkillFilter()
        skill_weights = {}
        comp_freq = {"python": 50, "django": 30, "fastapi": 10}
        merged = sf.merge_with_reference(skill_weights, comp_freq).unwrap()
        assert "django" in merged
        assert 0.0 <= merged["django"] <= 1.0

    def test_get_clean_weights_log_normalization_zero(self):
        sf = SkillFilter()
        raw = {"python": 5.0, "sql": 0.0}
        clean = sf.get_clean_weights(raw, use_reference=False, normalize_method="log").unwrap()
        assert clean["python"] == 1.0

    def test_get_clean_weights_empty_after_filters(self):
        sf = SkillFilter()
        raw = {}
        clean = sf.get_clean_weights(raw).unwrap()
        assert clean == {}

    def test_get_skill_categories_misc(self):
        sf = SkillFilter()
        skills = ["some_random_tool_xyz"]
        cats = sf.get_skill_categories(skills).unwrap()
        assert "other" in cats
        assert "some_random_tool_xyz" in cats["other"]


class TestSkillLevelAnalyzerExtended:
    def test_analyze_vacancies(self):
        analyzer = SkillLevelAnalyzer()
        vacancies = [
            {"skills": ["python", "sql"], "experience": "junior"},
            {"skills": ["python", "docker"], "experience": "middle"},
            {"skills": ["docker", "k8s"], "experience": "senior"},
            {"skills": ["python"], "experience": "не указано"},
        ]
        analyzer.analyze_vacancies(vacancies)
        assert "python" in analyzer.skill_by_level
        assert analyzer.skill_by_level["python"]["junior"] == 1
        assert analyzer.skill_by_level["python"]["middle"] == 2
        assert analyzer.skill_by_level["docker"]["senior"] == 1

    def test_get_skill_level(self):
        analyzer = SkillLevelAnalyzer()
        analyzer.skill_by_level = {
            "python": {"junior": 5, "middle": 10, "senior": 3},
            "docker": {"junior": 0, "middle": 3, "senior": 7},
            "git": {"junior": 4, "middle": 5, "senior": 4},
        }
        assert analyzer.get_skill_level("python") == "middle"
        assert analyzer.get_skill_level("docker") == "senior"
        assert analyzer.get_skill_level("git") == "all_levels"
        assert analyzer.get_skill_level("unknown") == "middle"

    def test_get_weights_for_level(self):
        analyzer = SkillLevelAnalyzer()
        analyzer.skill_by_level = {
            "python": {"junior": 10, "middle": 20, "senior": 5},
            "docker": {"junior": 1, "middle": 5, "senior": 15},
        }
        original_weights = {"python": 1.0, "docker": 1.0}
        weights_junior = analyzer.get_weights_for_level(original_weights, "junior").unwrap()
        assert weights_junior["python"] < 1.5
        assert weights_junior["docker"] < 0.8

    def test_get_skill_roadmap(self):
        analyzer = SkillLevelAnalyzer()
        analyzer.skill_by_level = {
            "python": {"junior": 5, "middle": 10, "senior": 3},
            "docker": {"middle": 2, "senior": 8},
        }
        roadmap = analyzer.get_skill_roadmap("python").unwrap()
        assert roadmap == {"junior": True, "middle": True, "senior": True}
        roadmap2 = analyzer.get_skill_roadmap("docker").unwrap()
        assert roadmap2 == {"junior": False, "middle": True, "senior": True}


class TestSkillLevelAnalyzerFull:
    @pytest.fixture
    def analyzer(self):
        return SkillLevelAnalyzer()

    def test_analyze_vacancies_with_leading_role(self, analyzer):
        vacancies = [
            {"skills": ["python"], "experience": {"name": "Ведущий разработчик"}},
        ]
        analyzer.analyze_vacancies(vacancies)
        assert analyzer.skill_by_level["python"]["senior"] == 1

    def test_get_weights_for_level_unknown_skill(self, analyzer):
        weights = analyzer.get_weights_for_level({"unknown": 1.0}, "middle").unwrap()
        assert weights["unknown"] == 1.0

    def test_get_weights_for_level_is_lower(self, analyzer):
        analyzer.skill_by_level = {
            "html": {"junior": 10, "middle": 2, "senior": 0},
            "k8s": {"junior": 0, "middle": 3, "senior": 10},
            "sql": {"junior": 5, "middle": 5, "senior": 5},
        }

        weights = analyzer.get_weights_for_level({"html": 1.0}, "middle").unwrap()
        assert weights["html"] < 1.0

        weights = analyzer.get_weights_for_level({"k8s": 1.0}, "junior").unwrap()
        assert weights["k8s"] < 1.0

        weights = analyzer.get_weights_for_level({"sql": 1.0}, "middle").unwrap()
        assert weights["sql"] > 0.6

    def test_is_lower_level_edge_cases(self, analyzer):
        assert analyzer._is_lower_level("junior", "junior") is False
        assert analyzer._is_lower_level("senior", "middle") is False
        assert analyzer._is_lower_level("unknown", "middle") is False

    def test_analyze_vacancies_with_leading_dict(self, analyzer):
        vacancies = [
            {"skills": ["architecture"], "experience": {"name": "Ведущий архитектор"}},
        ]
        analyzer.analyze_vacancies(vacancies)
        assert analyzer.skill_by_level["architecture"]["senior"] == 1
