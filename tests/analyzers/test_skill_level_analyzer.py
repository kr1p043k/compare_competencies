# tests/analyzers/test_skill_level_analyzer.py
import pytest
from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer


class TestSkillLevelAnalyzerFull:
    @pytest.fixture
    def analyzer(self):
        return SkillLevelAnalyzer()

    def test_init(self, analyzer):
        assert analyzer.skill_by_level is not None
        assert analyzer.level_thresholds == {
            'junior': 0.3,
            'middle': 0.5,
            'senior': 0.7
        }

    def test_analyze_vacancies_with_dict_experience(self, analyzer):
        """experience как dict с name (реальный формат hh.ru)"""
        vacancies = [
            {"skills": ["python", "sql"], "experience": {"name": "Junior"}},
            {"skills": ["python", "docker"], "experience": {"name": "Middle"}},
            {"skills": ["docker", "k8s"], "experience": {"name": "Senior"}},
        ]
        analyzer.analyze_vacancies(vacancies)
        assert analyzer.skill_by_level["python"]["junior"] == 1
        assert analyzer.skill_by_level["python"]["middle"] == 1
        assert analyzer.skill_by_level["docker"]["senior"] == 1

    def test_analyze_vacancies_with_russian_labels(self, analyzer):
        """Русские названия уровней"""
        vacancies = [
            {"skills": ["python"], "experience": {"name": "Младший разработчик"}},
            {"skills": ["docker"], "experience": {"name": "Старший инженер"}},
            {"skills": ["git"], "experience": {"name": "Начинающий"}},
        ]
        analyzer.analyze_vacancies(vacancies)
        assert analyzer.skill_by_level["python"]["junior"] == 1
        assert analyzer.skill_by_level["docker"]["senior"] == 1
        assert analyzer.skill_by_level["git"]["junior"] == 1

    def test_analyze_vacancies_unknown_level_defaults_middle(self, analyzer):
        """Неизвестный уровень → middle"""
        vacancies = [
            {"skills": ["python"], "experience": "неизвестный_уровень"},
        ]
        analyzer.analyze_vacancies(vacancies)
        assert analyzer.skill_by_level["python"]["middle"] == 1

    def test_get_weights_for_level_all_levels_skill(self, analyzer):
        """Навык all_levels получает повышающий коэффициент"""
        analyzer.skill_by_level = {
            "git": {"junior": 5, "middle": 5, "senior": 5},
        }
        weights = analyzer.get_weights_for_level({"git": 1.0}, "middle")
        # git → all_levels → коэффициент 1.1
        # level_percentage = 5/15 ≈ 0.33
        # adjusted = 1.0 * (0.3 + 0.33) * 1.1 = 0.693
        assert weights["git"] == pytest.approx(0.693, rel=0.1)

    def test_get_weights_for_level_specific_skill(self, analyzer):
        """Навык специфичный для целевого уровня"""
        analyzer.skill_by_level = {
            "k8s": {"junior": 0, "middle": 2, "senior": 10},
        }
        weights = analyzer.get_weights_for_level({"k8s": 1.0}, "senior")
        # k8s → senior-specific → коэффициент 1.5
        # level_percentage = 10/12 ≈ 0.833
        # adjusted = 1.0 * (0.3 + 0.833) * 1.5 = 1.6995
        assert weights["k8s"] > 1.0

    def test_get_weights_for_level_lower_skill(self, analyzer):
        """Навык ниже целевого уровня"""
        analyzer.skill_by_level = {
            "html": {"junior": 10, "middle": 2, "senior": 0},
        }
        weights = analyzer.get_weights_for_level({"html": 1.0}, "senior")
        # html → junior → коэффициент 0.8
        assert weights["html"] < 1.0

    def test_get_weights_for_level_higher_skill(self, analyzer):
        """Навык выше целевого уровня"""
        analyzer.skill_by_level = {
            "k8s": {"junior": 0, "middle": 1, "senior": 10},
        }
        weights = analyzer.get_weights_for_level({"k8s": 1.0}, "junior")
        # k8s → senior → is_lower_level("senior", "junior") = False
        # попадает в else → коэффициент 0.6
        assert weights["k8s"] < 1.0

    def test_is_lower_level(self, analyzer):
        assert analyzer._is_lower_level("junior", "middle") is True
        assert analyzer._is_lower_level("middle", "senior") is True
        assert analyzer._is_lower_level("senior", "junior") is False
        assert analyzer._is_lower_level("middle", "middle") is False

    def test_get_skill_roadmap_unknown_skill(self, analyzer):
        roadmap = analyzer.get_skill_roadmap("nonexistent")
        assert roadmap == {"junior": False, "middle": False, "senior": False}

    def test_analyze_vacancies_with_leading_variants(self, analyzer):
        """Строка 48: все варианты 'ведущий'/'старший' → senior"""
        vacancies = [
            {"skills": ["arch"], "experience": {"name": "Ведущий инженер"}},
            {"skills": ["lead"], "experience": {"name": "Старший разработчик"}},
        ]
        analyzer.analyze_vacancies(vacancies)
        assert analyzer.skill_by_level["arch"]["senior"] == 1
        assert analyzer.skill_by_level["lead"]["senior"] == 1