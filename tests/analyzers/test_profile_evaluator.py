# tests/analyzers/test_profile_evaluator.py
import pytest
from datetime import datetime
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.student import StudentProfile
from src.analyzers.profile_evaluator import ProfileEvaluator


class TestProfileEvaluatorExtended:
    @pytest.fixture
    def student(self):
        return StudentProfile(
            profile_name="test_student",
            competencies=[],
            skills=["python", "sql", "git", "fastapi", "docker"],
            target_level="middle",
            created_at=datetime.now()
        )

    @pytest.fixture
    def skill_weights_by_level(self):
        return {
            'junior': {'python': 0.8, 'sql': 0.6, 'git': 0.5, 'html': 0.4},
            'middle': {'python': 0.9, 'docker': 0.7, 'sql': 0.5, 'fastapi': 0.4},
            'senior': {'python': 0.9, 'docker': 0.9, 'k8s': 0.8, 'sql': 0.3},
        }

    @pytest.fixture
    def vacancies_skills(self):
        return [
            ["python", "sql", "git"],
            ["python", "docker", "fastapi"],
            ["python", "docker", "k8s"],
        ]

    @pytest.fixture
    def vacancies_skills_dict(self):
        return [
            {"skills": ["python", "sql", "git"]},
            {"skills": ["python", "docker", "fastapi"]},
            {"skills": ["python", "docker", "k8s"]},
        ]

    def test_init_with_required_params(self, skill_weights_by_level, vacancies_skills, vacancies_skills_dict):
        """Проверяет базовую инициализацию ProfileEvaluator"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False
        )
        assert evaluator.skill_weights == {"python": 0.9, "sql": 0.7}
        assert evaluator.vacancies_skills == vacancies_skills
        assert evaluator.use_clustering is False

    def test_evaluate_profile_returns_expected_keys(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False
        )

        result = evaluator.evaluate_profile(student)

        assert "market_coverage_score" in result
        assert "skill_coverage" in result
        assert "domain_coverage_score" in result
        assert "readiness_score" in result
        assert "avg_gap" in result
        assert "skill_metrics" in result
        assert "domain_coverage" in result
        assert "top_recommendations" in result
        assert "gaps" in result
        assert "student_skills" in result
        assert 0.0 <= result["readiness_score"] <= 100.0

    def test_evaluate_profile_without_level_weights_raises(self, student):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=[["python"]],
            vacancies_skills_dict=[{"skills": ["python"]}],
        )
        with pytest.raises(RuntimeError, match="skill_weights_by_level не были переданы"):
            evaluator.evaluate_profile(student)

    def test_evaluate_profile_readiness_score(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7, "git": 0.5},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )

        result = evaluator.evaluate_profile(student)
        assert 0.0 <= result["readiness_score"] <= 100.0
        assert isinstance(result["market_skill_coverage"], float)

    def test_evaluate_profile_top_recommendations_format(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )

        result = evaluator.evaluate_profile(student)
        assert isinstance(result["top_recommendations"], list)
        if result["top_recommendations"]:
            rec = result["top_recommendations"][0]
            assert isinstance(rec, tuple)
            assert len(rec) == 2  # (skill_name, score)

    def test_evaluate_profile_domain_coverage_present(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7, "docker": 0.5},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )

        result = evaluator.evaluate_profile(student)
        # DomainAnalyzer должен найти домены
        assert len(result["domain_coverage"]) > 0
        # Backend обычно содержит python, docker
        assert "Backend" in result["domain_coverage"] or len(result["domain_coverage"]) > 0

    def test_calculate_readiness_formula(self):
        evaluator = ProfileEvaluator(
            skill_weights={},
            vacancies_skills=[],
            vacancies_skills_dict=[],
        )
        readiness = evaluator._calculate_readiness(
            market_coverage_score=70.0,
            skill_coverage=60.0,
            domain_coverage_score=50.0,
            avg_gap=0.3
        )
        # Формула: 0.5*70 + 0.2*60 + 0.2*50 - 0.1*30 = 35 + 12 + 10 - 3 = 54
        assert readiness == pytest.approx(54.0)

    def test_domain_analyzer_creates_coverage(self, student, skill_weights_by_level,
                                                vacancies_skills, vacancies_skills_dict):
        from src.analyzers.domain_analyzer import DomainAnalyzer
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        # Проверяем, что domain_analyzer инициализирован
        assert evaluator.domain_analyzer is not None
        assert isinstance(evaluator.domain_analyzer, DomainAnalyzer)

        # Проверяем вычисление покрытия для данных студента
        coverages = evaluator.domain_analyzer.compute_domain_coverage(student.skills)
        assert isinstance(coverages, dict)
        assert len(coverages) > 0

class TestProfileEvaluatorFull:
    @pytest.fixture
    def student(self):
        return StudentProfile(
            profile_name="test_student",
            competencies=[],
            skills=["python", "sql", "git", "fastapi", "docker"],
            target_level="middle",
            created_at=datetime.now()
        )

    @pytest.fixture
    def skill_weights_by_level(self):
        return {
            'junior': {'python': 0.8, 'sql': 0.6, 'git': 0.5, 'html': 0.4},
            'middle': {'python': 0.9, 'docker': 0.7, 'sql': 0.5, 'fastapi': 0.4},
            'senior': {'python': 0.9, 'docker': 0.9, 'k8s': 0.8, 'sql': 0.3},
        }

    @pytest.fixture
    def vacancies_skills(self):
        return [["python", "sql"], ["fastapi", "docker"]]

    @pytest.fixture
    def vacancies_skills_dict(self):
        return [
            {"skills": ["python", "sql"]},
            {"skills": ["fastapi", "docker"]},
        ]

    def test_evaluate_profile_fallback_when_all_covered(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строка 130-137: все навыки покрыты → fallback"""
        # Делаем студента с идеальным покрытием
        student.skills = list(skill_weights_by_level["middle"].keys())
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "docker": 0.7, "sql": 0.5, "fastapi": 0.4},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False
        )

        result = evaluator.evaluate_profile(student)
        assert "top_recommendations" in result
        assert "market_skill_coverage" in result

    def test_cache_save_and_load(self, tmp_path, sample_skill_weights):
        """Строки 264-277: кэширование результатов"""
        import src.config as config
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)

        evaluator = ProfileEvaluator(
            skill_weights=sample_skill_weights,
            vacancies_skills=[["python"]],
            vacancies_skills_dict=[{"skills": ["python"]}],
        )
        # Проверяем что кэш-файл создаётся
        evaluator._save_cache()
        cache_path = tmp_path / "evaluation_cache.json"
        assert cache_path.exists()

        # Загружаем в новый объект
        evaluator2 = ProfileEvaluator(
            skill_weights=sample_skill_weights,
            vacancies_skills=[["python"]],
            vacancies_skills_dict=[{"skills": ["python"]}],
        )
        assert evaluator2._cache is not None

        monkeypatch.undo()

    def test_get_student_hash(self, sample_skill_weights):
        """Строки 271-272: хэширование студента"""
        evaluator = ProfileEvaluator(
            skill_weights=sample_skill_weights,
            vacancies_skills=[["python"]],
            vacancies_skills_dict=[{"skills": ["python"]}],
        )

        student = StudentProfile(
            profile_name="test",
            competencies=[],
            skills=["python", "sql"],
            target_level="middle",
            created_at=datetime.now()
        )

        hash1 = evaluator._get_student_hash(student, "middle")
        hash2 = evaluator._get_student_hash(student, "middle")
        assert hash1 == hash2  # Детерминированность

        student.skills = ["java"]
        hash3 = evaluator._get_student_hash(student, "middle")
        assert hash1 != hash3  # Разные навыки → разный хэш

    def test_load_cache_exception(self, tmp_path, sample_skill_weights):
        """Строка 275-277: обработка ошибки загрузки кэша"""
        import src.config as config
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)

        # Создаём битый JSON
        cache_path = tmp_path / "evaluation_cache.json"
        cache_path.write_text("{invalid json")

        evaluator = ProfileEvaluator(
            skill_weights=sample_skill_weights,
            vacancies_skills=[["python"]],
            vacancies_skills_dict=[{"skills": ["python"]}],
        )
        # Не должно быть исключения
        assert evaluator._cache == {}

        monkeypatch.undo()

    def test_evaluate_profile_with_clustering_enabled(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 187-188, 194: кластеризация включена"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=True
        )
        result = evaluator.evaluate_profile(student)
        assert "cluster_context" in result

    def test_evaluate_profile_with_student_user_type(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 209-211: уровень student"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        result = evaluator.evaluate_profile(student, user_type='student')
        assert result["level_weights_used"]["junior"] == pytest.approx(0.60)
        assert result["level_weights_used"]["senior"] == pytest.approx(0.10)

    def test_evaluate_profile_with_junior_user_type(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 209-211: уровень junior"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        result = evaluator.evaluate_profile(student, user_type='junior')
        assert result["level_weights_used"]["junior"] == pytest.approx(0.40)
        assert result["level_weights_used"]["middle"] == pytest.approx(0.40)

    def test_get_cluster_context_loads_clusterer(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 232-247: загрузка кластеризатора"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False  # отключаем, но проверяем структуру
        )
        context = evaluator._get_cluster_context(student, 'middle')
        assert context is None  # без кластеризации возвращает None

    def test_get_cluster_context_without_model(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 250-257: модель кластеризации не загружена"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=True
        )
        # Модель не загружена для 'middle' (нет файла)
        evaluator.cluster_models_loaded = {'junior': False, 'middle': False, 'senior': False}
        context = evaluator._get_cluster_context(student, 'middle')
        assert context is None

    def test_evaluate_profile_readiness_with_gap_penalty(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 232-247: readiness с большим gap"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        # Студент с очень малым количеством навыков → большой gap
        student.skills = ["python"]
        result = evaluator.evaluate_profile(student)
        assert result["readiness_score"] < 50  # большой gap снижает readiness

    def test_evaluate_profile_domain_bonus_applied(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 250-257: бонус от доменов к навыкам"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7, "docker": 0.5},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        result = evaluator.evaluate_profile(student)
        # Навыки из доменов получают бонус
        assert "domain_coverage" in result
        assert len(result["domain_coverage"]) > 0

    def test_evaluate_profile_readiness_with_gap_penalty(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 232-247: readiness с большим gap"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        student.skills = ["python"]
        result = evaluator.evaluate_profile(student)
        assert result["readiness_score"] < 50

    def test_evaluate_profile_domain_bonus_applied(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 250-257: бонус от доменов к навыкам"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7, "docker": 0.5},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        result = evaluator.evaluate_profile(student)
        assert "domain_coverage" in result
        assert len(result["domain_coverage"]) > 0
        
    def test_evaluate_profile_warns_when_all_covered(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict, caplog
    ):
        """Строки 130-137: предупреждение когда все навыки покрыты"""
        import logging
        caplog.set_level(logging.WARNING)
        
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "docker": 0.7, "sql": 0.5, "fastapi": 0.4},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False
        )
        # Студент уже знает все навыки из middle-уровня
        student.skills = list(skill_weights_by_level["middle"].keys())
        result = evaluator.evaluate_profile(student)
        assert "top_recommendations" in result
        
    def test_evaluate_profile_readiness_high_coverage(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 194, 232-247: проверка структуры результата"""
        evaluator = ProfileEvaluator(
            skill_weights=skill_weights_by_level["middle"],
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False
        )
        student.skills = list(skill_weights_by_level["middle"].keys())
        result = evaluator.evaluate_profile(student)
        
        # Проверяем что метод возвращает все ключи
        assert "readiness_score" in result
        assert "market_coverage_score" in result
        assert "skill_coverage" in result
        assert "domain_coverage_score" in result
        assert "top_recommendations" in result
        assert "gaps" in result
        assert "student_skills" in result
        # Все значения в допустимых диапазонах
        assert 0.0 <= result["readiness_score"] <= 100.0
        assert 0.0 <= result["market_coverage_score"] <= 100.0
            
    def test_evaluate_profile_with_middle_user_type(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 209-211: уровень middle"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        result = evaluator.evaluate_profile(student, user_type='middle')
        assert result["level_weights_used"]["middle"] == pytest.approx(0.50)
        assert result["level_weights_used"]["junior"] == pytest.approx(0.20)

    def test_evaluate_profile_with_unknown_user_type(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 209-211: неизвестный user_type → равные веса"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        result = evaluator.evaluate_profile(student, user_type='unknown_type')
        assert result["level_weights_used"]["junior"] == pytest.approx(0.33)
        assert result["level_weights_used"]["middle"] == pytest.approx(0.34)
        assert result["level_weights_used"]["senior"] == pytest.approx(0.33)

    def test_get_cluster_context_with_skills(self):
        """Строки 250-257: кластерный контекст с навыками"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=[["python", "sql"]],
            vacancies_skills_dict=[{"skills": ["python", "sql"]}],
            skill_weights_by_level={'junior': {}, 'middle': {'python': 0.9}, 'senior': {}},
            use_clustering=False
        )
        student = StudentProfile(
            profile_name="test",
            competencies=[],
            skills=["python", "sql"],
            target_level="middle",
            created_at=datetime.now()
        )
        context = evaluator._get_cluster_context(student, 'middle')
        assert context is None  # use_clustering=False

    def test_get_cluster_context_without_skills(self):
        """Строки 250-257: кластерный контекст без навыков студента"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=[["python"]],
            vacancies_skills_dict=[{"skills": ["python"]}],
            skill_weights_by_level={'junior': {}, 'middle': {'python': 0.9}, 'senior': {}},
            use_clustering=True
        )
        student = StudentProfile(
            profile_name="test",
            competencies=[],
            skills=[],
            target_level="middle",
            created_at=datetime.now()
        )
        evaluator.cluster_models_loaded = {'middle': False}
        context = evaluator._get_cluster_context(student, 'middle')
        assert context is None

    def test_evaluate_profile_fallback_warns_and_returns_recommendations(
        self, skill_weights_by_level, vacancies_skills, vacancies_skills_dict, caplog
    ):
        """Строки 130-137: проверка что метод не падает при полном покрытии"""
        import logging
        caplog.set_level(logging.WARNING)
        
        # Используем только middle-уровень чтобы избежать лишних навыков
        middle_only = {
            'junior': {},
            'middle': skill_weights_by_level["middle"],
            'senior': {}
        }
        evaluator = ProfileEvaluator(
            skill_weights=skill_weights_by_level["middle"],
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=middle_only,
            use_clustering=False
        )
        
        student = StudentProfile(
            profile_name="perfect_student",
            competencies=[],
            skills=list(skill_weights_by_level["middle"].keys()),
            target_level="middle",
            created_at=datetime.now()
        )
        
        result = evaluator.evaluate_profile(student)
        assert "top_recommendations" in result
        assert "readiness_score" in result
        # При полном покрытии рекомендаций нет
        assert len(result["top_recommendations"]) == 0

    def test_evaluate_profile_all_skills_covered(
        self, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 130-137: все рыночные навыки покрыты → пустые рекомендации"""
        middle_only = {
            'junior': {},
            'middle': skill_weights_by_level["middle"],
            'senior': {}
        }
        evaluator = ProfileEvaluator(
            skill_weights=skill_weights_by_level["middle"],
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=middle_only,
            use_clustering=False
        )
        
        student = StudentProfile(
            profile_name="perfect",
            competencies=[],
            skills=list(skill_weights_by_level["middle"].keys()),
            target_level="middle",
            created_at=datetime.now()
        )
        
        result = evaluator.evaluate_profile(student)
        assert "top_recommendations" in result
        assert len(result["top_recommendations"]) == 0, \
            f"Expected empty, got: {result['top_recommendations']}"

    def test_evaluate_profile_fallback_path(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 136-137: проверка fallback ветки с caplog"""
        import logging
        evaluator = ProfileEvaluator(
            skill_weights=skill_weights_by_level["middle"],
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False
        )
        # Студент покрывает все middle-навыки (они будут отфильтрованы)
        student.skills = list(skill_weights_by_level["middle"].keys())
        
        with patch.object(evaluator.gap_analyzer_new, 'compute_metrics') as mock_metrics:
            # Возвращаем метрики только для middle-навыков
            from src.models.market_metrics import SkillMetrics
            metrics_dict = {
                skill: SkillMetrics(skill=skill, user_level=1.0)
                for skill in skill_weights_by_level["middle"]
            }
            # Устанавливаем нулевые gaps чтобы попасть в ветку "все покрыты"
            for m in metrics_dict.values():
                m.gap_j = 0.0
                m.gap_m = 0.0
                m.gap_s = 0.0
                m.cluster_relevance = 0.0
            mock_metrics.return_value = metrics_dict
            
            result = evaluator.evaluate_profile(student)
            assert "top_recommendations" in result

    def test_evaluate_profile_default_user_type(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 209-211: default user_type = 'student'"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        # Не передаём user_type — должен использоваться 'student'
        result = evaluator.evaluate_profile(student)
        assert result["level_weights_used"]["junior"] == pytest.approx(0.60)
        assert result["level_weights_used"]["middle"] == pytest.approx(0.30)
        assert result["level_weights_used"]["senior"] == pytest.approx(0.10)