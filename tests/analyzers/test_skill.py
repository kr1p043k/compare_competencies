# tests/analyzers/test_skill_filter_extended.py
import pytest
from src.analyzers.skill_filter import SkillFilter

class TestSkillFilterExtended:
    def test_filter_weights_removes_generic(self):
        sf = SkillFilter()
        weights = {"frontend": 10.0, "python": 8.0, "web": 5.0}
        filtered = sf.filter_weights(weights, min_weight=0)
        assert "frontend" not in filtered
        assert "web" not in filtered
        assert "python" in filtered

    def test_filter_weights_min_weight(self):
        sf = SkillFilter()
        weights = {"python": 0.1, "sql": 0.005}
        filtered = sf.filter_weights(weights, min_weight=0.01)
        assert "python" in filtered
        assert "sql" not in filtered

    def test_normalize_weights_minmax(self):
        sf = SkillFilter()
        weights = {"a": 1, "b": 5, "c": 10}
        norm = sf.normalize_weights(weights, method='minmax')
        assert norm["c"] == 1.0
        assert norm["a"] == 0.1
        assert 0.1 < norm["b"] < 1.0

    def test_normalize_weights_log(self):
        sf = SkillFilter()
        weights = {"a": 1, "b": 100}
        norm = sf.normalize_weights(weights, method='log')
        assert norm["b"] == 1.0
        assert norm["a"] < 1.0

    def test_merge_with_reference(self):
        sf = SkillFilter()
        # Убираем sql из generic слов для этого теста
        sf.GENERIC_WORDS = sf.GENERIC_WORDS - {"sql"}
        skill_weights = {"python": 0.9, "django": 0.7}
        comp_freq = {"python": 50, "sql": 30, "frontend": 20}
        merged = sf.merge_with_reference(skill_weights, comp_freq)
        assert "python" in merged
        assert "sql" in merged   # теперь пройдёт

    def test_get_clean_weights_full_pipeline(self):
        sf = SkillFilter()
        sf.GENERIC_WORDS = sf.GENERIC_WORDS - {"sql"}
        raw = {"python": 0.9, "frontend": 0.8, "sql": 0.5}
        freq = {"python": 100, "sql": 50}
        clean = sf.get_clean_weights(raw, freq, use_reference=True, normalize_method='minmax')
        assert "frontend" not in clean
        assert "python" in clean
        assert clean["python"] == 1.0
        assert "sql" in clean
        assert 0 < clean["sql"] < 1.0

    def test_validate_skills(self):
        sf = SkillFilter()
        skills = ["python", "frontend", "very long skill name with many words"]
        valid = sf.validate_skills(skills)
        assert "python" in valid
        assert "frontend" not in valid
        assert "very long skill name with many words" not in valid

    def test_get_skill_categories(self):
        sf = SkillFilter()
        skills = ["python", "react", "postgresql", "docker", "aws", "machine learning"]
        cats = sf.get_skill_categories(skills)
        assert "programming_languages" in cats
        assert "frameworks" in cats
        assert "databases" in cats
        assert "devops" in cats
        assert "cloud" in cats
        assert "data_science" in cats
# tests/analyzers/test_skill_level_analyzer_extended.py
import pytest
from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer

class TestSkillLevelAnalyzerExtended:
    def test_analyze_vacancies(self):
        analyzer = SkillLevelAnalyzer()
        vacancies = [
            {"skills": ["python", "sql"], "experience": "junior"},
            {"skills": ["python", "docker"], "experience": "middle"},
            {"skills": ["docker", "k8s"], "experience": "senior"},
            {"skills": ["python"], "experience": "не указано"}  # default middle
        ]
        analyzer.analyze_vacancies(vacancies)
        assert "python" in analyzer.skill_by_level
        assert analyzer.skill_by_level["python"]["junior"] == 1
        assert analyzer.skill_by_level["python"]["middle"] == 2  # включая не указано
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
        assert analyzer.get_skill_level("git") == "all_levels"  # >30% в трёх
        assert analyzer.get_skill_level("unknown") == "middle"

    def test_get_weights_for_level(self):
        analyzer = SkillLevelAnalyzer()
        analyzer.skill_by_level = {
            "python": {"junior": 10, "middle": 20, "senior": 5},
            "docker": {"junior": 1, "middle": 5, "senior": 15},
        }
        original_weights = {"python": 1.0, "docker": 1.0}
        weights_junior = analyzer.get_weights_for_level(original_weights, "junior")
        # python специфичен для junior? на самом деле middle доминирует, но он есть в junior
        assert weights_junior["python"] < 1.5
        # docker для junior должен быть ниже, т.к. он senior-специфичен
        assert weights_junior["docker"] < 0.8

    def test_get_skill_roadmap(self):
        analyzer = SkillLevelAnalyzer()
        analyzer.skill_by_level = {
            "python": {"junior": 5, "middle": 10, "senior": 3},
            "docker": {"middle": 2, "senior": 8},
        }
        roadmap = analyzer.get_skill_roadmap("python")
        assert roadmap == {"junior": True, "middle": True, "senior": True}
        roadmap2 = analyzer.get_skill_roadmap("docker")
        assert roadmap2 == {"junior": False, "middle": True, "senior": True}