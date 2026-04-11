# tests/analyzers/test_profile_evaluator.py
import pytest
from datetime import datetime
from src.models.student import StudentProfile, ExperienceLevel
from src.analyzers.profile_evaluator import ProfileEvaluator
from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer
from src.analyzers.comparator import CompetencyComparator
from unittest.mock import MagicMock, patch
from src.models.student import ProfileEvaluation
class TestProfileEvaluatorExtended:
    @pytest.fixture
    def sample_student(self):
        return StudentProfile(
            profile_name="test",
            competencies=["comp1"],
            skills=["python", "sql"],
            target_level="middle",
            created_at=datetime.now()
        )

    @pytest.fixture
    def skill_weights(self):
        return {"python": 100, "sql": 80, "docker": 60, "mlops": 40}

    @pytest.fixture
    def vacancies_skills(self):
        return [["python", "sql"], ["python", "docker"]]

    def test_evaluate_profile(self, sample_student, skill_weights, vacancies_skills):
        level_analyzer = SkillLevelAnalyzer()
        # Добавим фиктивные данные о вакансиях
        level_analyzer.skill_by_level["python"]["middle"] = 10
        level_analyzer.skill_by_level["sql"]["middle"] = 8

        evaluator = ProfileEvaluator(skill_weights, vacancies_skills)
        # Мокаем comparator
        with patch.object(evaluator, '_get_or_create_comparator') as mock_comp:
            mock_comp.return_value = MagicMock()
            mock_comp.return_value.compare.return_value = (0.75, 0.8)
            eval_result = evaluator.evaluate_profile(
                sample_student, "test", level_analyzer, skill_weights
            )
            assert eval_result.profile_name == "test"
            assert eval_result.level == "middle"
            assert 0 <= eval_result.readiness_score <= 100
            assert "raw" in eval_result.coverage
            assert eval_result.gaps["total"] >= 0

    def test_evaluate_multiple_profiles(self, skill_weights, vacancies_skills):
        students = {
            "base": StudentProfile(
                profile_name="base",
                competencies=[],
                skills=["python"],
                target_level="junior",
                created_at=datetime.now()
            ),
            "advanced": StudentProfile(
                profile_name="advanced",
                competencies=[],
                skills=["python", "docker", "mlops"],
                target_level="senior",
                created_at=datetime.now()
            )
        }
        level_analyzer = SkillLevelAnalyzer()
        evaluator = ProfileEvaluator(skill_weights, vacancies_skills)
        with patch.object(evaluator, 'evaluate_profile') as mock_eval:
            # Создаём полноценные объекты ProfileEvaluation
            eval1 = ProfileEvaluation(
                profile_name="base",
                student=students["base"],
                level="junior",
                raw_score=0.65,
                confidence=0.8,
                coverage={"raw": 65.0, "adjusted": 65.0, "difficulty_multiplier": 1.0},
                readiness_score=65,
                recommendation="test",
                gaps={
                    "high_priority": 0, "medium_priority": 1, "low_priority": 2,
                    "total": 3, "high_priority_skills": [], "medium_priority_skills": ["sql"]
                }
            )
            eval2 = ProfileEvaluation(
                profile_name="advanced",
                student=students["advanced"],
                level="senior",
                raw_score=0.85,
                confidence=0.9,
                coverage={"raw": 85.0, "adjusted": 68.0, "difficulty_multiplier": 1.25},
                readiness_score=85,
                recommendation="test",
                gaps={
                    "high_priority": 0, "medium_priority": 0, "low_priority": 1,
                    "total": 1, "high_priority_skills": [], "medium_priority_skills": []
                }
            )
            mock_eval.side_effect = [eval1, eval2]
            weights_by_level = {"junior": skill_weights, "senior": skill_weights}
            comparison = evaluator.evaluate_multiple_profiles(students, level_analyzer, weights_by_level)
            assert comparison.average_readiness == 75.0
            assert comparison.best_evaluation.profile_name == "advanced"

    def test_calculate_readiness(self):
        evaluator = ProfileEvaluator({}, [])
        readiness = evaluator._calculate_readiness(
            score=0.8, adjusted_coverage=70.0,
            gaps={"high_priority": ["a", "b"]},
            difficulty_multiplier=1.0
        )
        assert 0 <= readiness <= 100

    def test_level_difficulty_multipliers(self):
        evaluator = ProfileEvaluator({}, [])
        assert evaluator.level_difficulty["junior"] == 1.0
        assert evaluator.level_difficulty["middle"] == 1.1
        assert evaluator.level_difficulty["senior"] == 1.25
        # --------------------------------------------------------------
    # Дополнительные тесты для покрытия пропущенных строк
    # --------------------------------------------------------------

    def test_get_or_create_comparator_fit_fails(self, skill_weights, vacancies_skills):
        """Покрытие строки 134 (else при неудаче fit_market)"""
        evaluator = ProfileEvaluator(skill_weights, vacancies_skills)
        level_analyzer = SkillLevelAnalyzer()
        with patch.object(CompetencyComparator, 'fit_market', return_value=False):
            comparator = evaluator._get_or_create_comparator('middle', level_analyzer)
            assert comparator is not None
            # Проверяем, что comparator создан, но fit_market не удался (логгируется warning)
            # Достаточно того, что метод не упал и вернул объект

    def test_evaluate_multiple_profiles_empty(self, skill_weights, vacancies_skills):
        """Покрытие строки 227 (пустой словарь профилей)"""
        evaluator = ProfileEvaluator(skill_weights, vacancies_skills)
        level_analyzer = SkillLevelAnalyzer()
        with pytest.raises(ValueError, match="Нужен хотя бы один профиль для оценки"):
            evaluator.evaluate_multiple_profiles({}, level_analyzer, {})

    def test_get_recommendation_branches(self):
        """Покрытие всех ветвей _get_recommendation (строки 182-201)"""
        evaluator = ProfileEvaluator({}, [])
        assert "✅ Готов к senior уровню" in evaluator._get_recommendation(85, "senior")
        assert "📈 Неплохо для middle" in evaluator._get_recommendation(70, "middle")
        assert "⚠️ Нужно подготовиться к junior" in evaluator._get_recommendation(50, "junior")
        assert "❌ Недостаточно готов к lead" in evaluator._get_recommendation(30, "lead")

    def test_generate_summary(self):
        """Покрытие метода _generate_summary (строки 230-233)"""
        evaluator = ProfileEvaluator({}, [])
        # Создаём фейковые оценки
        student = StudentProfile(
            profile_name="test",
            competencies=[],
            skills=[],
            target_level="middle",
            created_at=datetime.now()
        )
        eval1 = ProfileEvaluation(
            profile_name="base",
            student=student,
            level="junior",
            raw_score=0.5,
            confidence=0.8,
            coverage={"raw": 50, "adjusted": 50, "difficulty_multiplier": 1.0},
            readiness_score=50,
            recommendation="test",
            gaps={"total": 0}
        )
        eval2 = ProfileEvaluation(
            profile_name="advanced",
            student=student,
            level="senior",
            raw_score=0.9,
            confidence=0.9,
            coverage={"raw": 90, "adjusted": 72, "difficulty_multiplier": 1.25},
            readiness_score=90,
            recommendation="test",
            gaps={"total": 0}
        )
        summary = evaluator._generate_summary([eval1, eval2], eval2, 70.0)
        assert "Оценено 2 профилей" in summary
        assert "Средняя готовность: 70.00%" in summary
        assert "Лучший профиль: advanced (90.00%)" in summary

    def test_evaluate_multiple_profiles_fallback_weights(self, skill_weights, vacancies_skills):
        """Покрытие ветки, когда skill_weights_by_level не содержит нужного уровня"""
        student = StudentProfile(
            profile_name="fallback",
            competencies=[],
            skills=["python"],
            target_level="senior",
            created_at=datetime.now()
        )
        level_analyzer = SkillLevelAnalyzer()
        evaluator = ProfileEvaluator(skill_weights, vacancies_skills)

        # Мокаем evaluate_profile, чтобы не вызывать реальную логику
        with patch.object(evaluator, 'evaluate_profile') as mock_eval:
            mock_eval.return_value = ProfileEvaluation(
                profile_name="fallback",
                student=student,
                level="senior",
                raw_score=0.8,
                confidence=0.9,
                coverage={"raw": 80},
                readiness_score=80,
                recommendation="",
                gaps={"total": 0}
            )
            # Не передаём 'senior' в weights_by_level
            weights_by_level = {"junior": skill_weights, "middle": skill_weights}
            evaluator.evaluate_multiple_profiles(
                {"fallback": student}, level_analyzer, weights_by_level
            )
            # Убеждаемся, что evaluate_profile вызван с правильными весами (должен fallback к self.skill_weights)
            call_args = mock_eval.call_args[0]
            # call_args: (student, profile_name, level_analyzer, profile_skill_weights)
            assert call_args[3] == evaluator.skill_weights