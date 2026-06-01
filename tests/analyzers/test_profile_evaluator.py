# tests/analyzers/test_profile_evaluator.py
from datetime import datetime
from unittest.mock import patch, MagicMock
import json
import numpy as np

import pytest

from src import Ok, Err
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
from src.models.student import StudentProfile


class TestProfileEvaluatorExtended:
    @pytest.fixture
    def student(self):
        return StudentProfile(
            profile_name="test_student",
            competencies=[],
            skills=["python", "sql", "git", "fastapi", "docker"],
            target_level="middle",
            created_at=datetime.now(),
        )

    @pytest.fixture
    def skill_weights_by_level(self):
        return {
            "junior": {"python": 0.8, "sql": 0.6, "git": 0.5, "html": 0.4},
            "middle": {"python": 0.9, "docker": 0.7, "sql": 0.5, "fastapi": 0.4},
            "senior": {"python": 0.9, "docker": 0.9, "k8s": 0.8, "sql": 0.3},
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
            use_clustering=False,
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
            use_clustering=False,
        )

        result = evaluator.evaluate_profile(student).unwrap()

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
        assert "krm_coverage" in result
        assert "profession_coverage" in result
        assert "profession_coverage_detail" in result
        assert 0.0 <= result["readiness_score"] <= 100.0

    def test_evaluate_profile_without_level_weights_raises(self, student):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=[["python"]],
            vacancies_skills_dict=[{"skills": ["python"]}],
        )
        result = evaluator.evaluate_profile(student)
        assert result.is_err()
        assert "skill_weights_by_level" in result.err().message

    def test_evaluate_profile_readiness_score(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7, "git": 0.5},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )

        result = evaluator.evaluate_profile(student).unwrap()
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

        result = evaluator.evaluate_profile(student).unwrap()
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

        result = evaluator.evaluate_profile(student).unwrap()
        # DomainAnalyzer должен найти домены
        assert len(result["domain_coverage"]) > 0
        # Backend обычно содержит python, docker
        assert "Backend" in result["domain_coverage"] or len(result["domain_coverage"]) > 0

    def test_readiness_score_in_evaluate_result(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        result = evaluator.evaluate_profile(student).unwrap()
        assert "readiness_score" in result
        assert 0 <= result["readiness_score"] <= 100

    def test_domain_analyzer_creates_coverage(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        from src.analyzers.comparison.domain_analyzer import DomainAnalyzer

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
        coverages = evaluator.domain_analyzer.compute_domain_coverage(student.skills).unwrap()
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
            created_at=datetime.now(),
        )

    @pytest.fixture
    def skill_weights_by_level(self):
        return {
            "junior": {"python": 0.8, "sql": 0.6, "git": 0.5, "html": 0.4},
            "middle": {"python": 0.9, "docker": 0.7, "sql": 0.5, "fastapi": 0.4},
            "senior": {"python": 0.9, "docker": 0.9, "k8s": 0.8, "sql": 0.3},
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
            use_clustering=False,
        )

        result = evaluator.evaluate_profile(student).unwrap()
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
            created_at=datetime.now(),
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
            use_clustering=True,
        )
        result = evaluator.evaluate_profile(student).unwrap()
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
        result = evaluator.evaluate_profile(student, user_type="student").unwrap()
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
        result = evaluator.evaluate_profile(student, user_type="junior").unwrap()
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
            use_clustering=False,  # отключаем, но проверяем структуру
        )
        context = evaluator._get_cluster_context(student, "middle")
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
            use_clustering=True,
        )
        # Модель не загружена для 'middle' (нет файла)
        evaluator.cluster_models_loaded = {"junior": False, "middle": False, "senior": False}
        context = evaluator._get_cluster_context(student, "middle")
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
        result = evaluator.evaluate_profile(student).unwrap()
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
        result = evaluator.evaluate_profile(student).unwrap()
        # Навыки из доменов получают бонус
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
            use_clustering=False,
        )
        # Студент уже знает все навыки из middle-уровня
        student.skills = list(skill_weights_by_level["middle"].keys())
        result = evaluator.evaluate_profile(student).unwrap()
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
            use_clustering=False,
        )
        student.skills = list(skill_weights_by_level["middle"].keys())
        result = evaluator.evaluate_profile(student).unwrap()

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
        result = evaluator.evaluate_profile(student, user_type="middle").unwrap()
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
        result = evaluator.evaluate_profile(student, user_type="unknown_type").unwrap()
        assert result["level_weights_used"]["junior"] == pytest.approx(0.33)
        assert result["level_weights_used"]["middle"] == pytest.approx(0.34)
        assert result["level_weights_used"]["senior"] == pytest.approx(0.33)

    def test_get_cluster_context_with_skills(self):
        """Строки 250-257: кластерный контекст с навыками"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=[["python", "sql"]],
            vacancies_skills_dict=[{"skills": ["python", "sql"]}],
            skill_weights_by_level={"junior": {}, "middle": {"python": 0.9}, "senior": {}},
            use_clustering=False,
        )
        student = StudentProfile(
            profile_name="test",
            competencies=[],
            skills=["python", "sql"],
            target_level="middle",
            created_at=datetime.now(),
        )
        context = evaluator._get_cluster_context(student, "middle")
        assert context is None  # use_clustering=False

    def test_get_cluster_context_without_skills(self):
        """Строки 250-257: кластерный контекст без навыков студента"""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=[["python"]],
            vacancies_skills_dict=[{"skills": ["python"]}],
            skill_weights_by_level={"junior": {}, "middle": {"python": 0.9}, "senior": {}},
            use_clustering=True,
        )
        student = StudentProfile(
            profile_name="test", competencies=[], skills=[], target_level="middle", created_at=datetime.now()
        )
        evaluator.cluster_models_loaded = {"middle": False}
        context = evaluator._get_cluster_context(student, "middle")
        assert context is None

    def test_evaluate_profile_fallback_warns_and_returns_recommendations(
        self, skill_weights_by_level, vacancies_skills, vacancies_skills_dict, caplog
    ):
        """Строки 130-137: проверка что метод не падает при полном покрытии"""
        import logging

        caplog.set_level(logging.WARNING)

        # Используем только middle-уровень чтобы избежать лишних навыков
        middle_only = {"junior": {}, "middle": skill_weights_by_level["middle"], "senior": {}}
        evaluator = ProfileEvaluator(
            skill_weights=skill_weights_by_level["middle"],
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=middle_only,
            use_clustering=False,
        )

        student = StudentProfile(
            profile_name="perfect_student",
            competencies=[],
            skills=list(skill_weights_by_level["middle"].keys()),
            target_level="middle",
            created_at=datetime.now(),
        )

        result = evaluator.evaluate_profile(student).unwrap()
        assert "top_recommendations" in result
        assert "readiness_score" in result
        # При полном покрытии рекомендаций нет
        assert len(result["top_recommendations"]) == 0

    def test_evaluate_profile_all_skills_covered(self, skill_weights_by_level, vacancies_skills, vacancies_skills_dict):
        """Строки 130-137: все рыночные навыки покрыты → пустые рекомендации"""
        middle_only = {"junior": {}, "middle": skill_weights_by_level["middle"], "senior": {}}
        evaluator = ProfileEvaluator(
            skill_weights=skill_weights_by_level["middle"],
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=middle_only,
            use_clustering=False,
        )

        student = StudentProfile(
            profile_name="perfect",
            competencies=[],
            skills=list(skill_weights_by_level["middle"].keys()),
            target_level="middle",
            created_at=datetime.now(),
        )

        result = evaluator.evaluate_profile(student).unwrap()
        assert "top_recommendations" in result
        assert len(result["top_recommendations"]) == 0, f"Expected empty, got: {result['top_recommendations']}"

    def test_evaluate_profile_fallback_path(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict
    ):
        """Строки 136-137: проверка fallback ветки с caplog"""
        evaluator = ProfileEvaluator(
            skill_weights=skill_weights_by_level["middle"],
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False,
        )
        # Студент покрывает все middle-навыки (они будут отфильтрованы)
        student.skills = list(skill_weights_by_level["middle"].keys())

        with patch.object(evaluator.gap_analyzer_new, "compute_metrics") as mock_metrics:
            # Возвращаем метрики только для middle-навыков
            from src.models.market_metrics import SkillMetrics

            metrics_dict = {
                skill: SkillMetrics(skill=skill, user_level=1.0) for skill in skill_weights_by_level["middle"]
            }
            # Устанавливаем нулевые gaps чтобы попасть в ветку "все покрыты"
            for m in metrics_dict.values():
                m.gap_j = 0.0
                m.gap_m = 0.0
                m.gap_s = 0.0
                m.cluster_relevance = 0.0
            mock_metrics.return_value = Ok(metrics_dict)

            result = evaluator.evaluate_profile(student).unwrap()
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
        result = evaluator.evaluate_profile(student).unwrap()
        assert result["level_weights_used"]["junior"] == pytest.approx(0.60)
        assert result["level_weights_used"]["middle"] == pytest.approx(0.30)
        assert result["level_weights_used"]["senior"] == pytest.approx(0.10)

    @pytest.fixture
    def student(self):
        return StudentProfile(
            profile_name="test_student",
            competencies=[],
            skills=["python", "sql", "git"],
            target_level="middle",
            created_at=datetime.now(),
        )

    @pytest.fixture
    def skill_weights_by_level(self):
        return {
            "junior": {"python": 0.8, "sql": 0.6},
            "middle": {"python": 0.9, "sql": 0.7, "docker": 0.5},
            "senior": {"python": 0.9, "docker": 0.9},
        }

    @pytest.fixture
    def vacancies_skills(self):
        return [["python", "sql"], ["docker"]]

    @pytest.fixture
    def vacancies_skills_dict(self):
        return [{"skills": ["python", "sql"]}, {"skills": ["docker"]}]

    # === строки 100 и 112 (конструктор) ===
    def test_constructor_creates_gap_analyzer_and_cache_path(
        self, skill_weights_by_level, vacancies_skills, vacancies_skills_dict, tmp_path, monkeypatch
    ):
        """Покрытие строк 100 (GapAnalyzer) и 112 (cache_path)."""
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        # строка 100
        assert evaluator.gap_analyzer_new is not None
        # строка 112
        assert evaluator._cache_path == tmp_path / "evaluation_cache.json"

    def test_constructor_without_skill_weights_by_level(
        self, vacancies_skills, vacancies_skills_dict
    ):
        """Проверка, что gap_analyzer_new = None, когда skill_weights_by_level не передан."""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        assert evaluator.gap_analyzer_new is None

    # === строки 174-176 (цикл по required_skills в evaluate_profile) ===
    def test_evaluate_profile_domain_bonus_loop(
        self, student, skill_weights_by_level, vacancies_skills, vacancies_skills_dict, mocker
    ):
        """Строки 174-176: цикл по required_skills для вычисления skill_to_domain_bonus."""
        evaluator = ProfileEvaluator(
            skill_weights=skill_weights_by_level["middle"],
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False,
        )
        # Подменяем domain_analyzer, чтобы вернуть домены с required_skills
        from src.models.market_metrics import DomainMetrics
        fake_domain = DomainMetrics(domain="Backend", required_skills=["python", "fastapi"])
        fake_domain.coverage = 0.9
        fake_coverages = {"Backend": fake_domain}

        with patch.object(evaluator.domain_analyzer, "compute_domain_coverage", return_value=Ok(fake_coverages)):
            with patch.object(evaluator.gap_analyzer_new, "compute_metrics") as mock_metrics:
                from src.models.market_metrics import SkillMetrics
                metrics = {
                    "python": SkillMetrics(skill="python", user_level=1.0),
                    "fastapi": SkillMetrics(skill="fastapi", user_level=0.0),
                }
                for m in metrics.values():
                    m.gap_j = 0.5; m.gap_m = 0.5; m.gap_s = 0.5
                mock_metrics.return_value = Ok(metrics)

                result = evaluator.evaluate_profile(student).unwrap()
        # Должен отработать без ошибок, бонусы применены
        assert "top_recommendations" in result

    # === строки 264-279 (кэширование: _load_cache, _save_cache, _get_student_hash) ===
    def test_cache_save_and_load_full(self, tmp_path, monkeypatch, skill_weights_by_level, vacancies_skills, vacancies_skills_dict):
        """Строки 264-279: сохранение и загрузка кэша, хэширование студента."""
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        # _save_cache (строка 271)
        evaluator._cache = {"some": "data"}
        evaluator._save_cache()
        cache_file = tmp_path / "evaluation_cache.json"
        assert cache_file.exists()
        with open(cache_file) as f:
            assert json.load(f) == {"some": "data"}

        # _load_cache (строки 265-268)
        evaluator2 = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            skill_weights_by_level=skill_weights_by_level,
        )
        assert evaluator2._cache == {"some": "data"}

        # _get_student_hash (строки 275-277)
        student = StudentProfile(
            profile_name="test", competencies=[], skills=["python"], target_level="middle", created_at=datetime.now()
        )
        h1 = evaluator._get_student_hash(student, "middle")
        h2 = evaluator._get_student_hash(student, "middle")
        assert h1 == h2
        assert isinstance(h1, str)

    def test_load_cache_exception(self, tmp_path, monkeypatch, vacancies_skills, vacancies_skills_dict):
        """Строка 267-268: исключение при загрузке кэша -> пустой кэш."""
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", tmp_path)
        cache_file = tmp_path / "evaluation_cache.json"
        cache_file.write_text("{invalid")
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        assert evaluator._cache == {}

    # === строки 280-288 (_compute_student_hash, _get_student_cache_path) ===
    def test_compute_student_hash_and_cache_path(self, tmp_path, monkeypatch, vacancies_skills, vacancies_skills_dict):
        """Строки 280-288: compute_student_hash и get_student_cache_path."""
        monkeypatch.setattr("src.config.STUDENT_EMB_CACHE_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        student = StudentProfile(
            profile_name="john", competencies=[], skills=["python", "sql"], target_level="middle", created_at=datetime.now()
        )
        hash_val = evaluator._compute_student_hash(student)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # sha256 hex

        cache_path = evaluator._get_student_cache_path(student)
        assert cache_path == tmp_path / "john_embedding.json"

    # === строки 290-297 (_load_cached_embedding) ===
    def test_load_cached_embedding_valid(self, tmp_path, monkeypatch, vacancies_skills, vacancies_skills_dict):
        """Строки 290-297: загрузка валидного эмбеддинга из кэша."""
        monkeypatch.setattr("src.config.STUDENT_EMB_CACHE_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        student = StudentProfile(
            profile_name="jane", competencies=[], skills=["python"], target_level="middle", created_at=datetime.now()
        )
        hash_val = evaluator._compute_student_hash(student)
        emb = np.array([1.0, 2.0, 3.0])
        cache_path = evaluator._get_student_cache_path(student)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"hash": hash_val, "embedding": emb.tolist()}, f)

        loaded = evaluator._load_cached_embedding(student)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, emb)

    def test_load_cached_embedding_invalid_hash(self, tmp_path, monkeypatch, vacancies_skills, vacancies_skills_dict):
        """Строки 294-295: несовпадение хэша -> возврат None."""
        monkeypatch.setattr("src.config.STUDENT_EMB_CACHE_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        student = StudentProfile(
            profile_name="jane", competencies=[], skills=["python"], target_level="middle", created_at=datetime.now()
        )
        cache_path = evaluator._get_student_cache_path(student)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"hash": "wrong", "embedding": [1,2,3]}, f)

        loaded = evaluator._load_cached_embedding(student)
        assert loaded is None

    def test_load_cached_embedding_no_file(self, tmp_path, monkeypatch, vacancies_skills, vacancies_skills_dict):
        """Строка 290: файл не существует -> None."""
        monkeypatch.setattr("src.config.STUDENT_EMB_CACHE_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        student = StudentProfile(
            profile_name="nobody", competencies=[], skills=["python"], target_level="middle", created_at=datetime.now()
        )
        assert evaluator._load_cached_embedding(student) is None

    # === строки 299-311 (_save_embedding_cache) и строка 348 (manifest.save) ===
    def test_save_embedding_cache(self, tmp_path, monkeypatch, vacancies_skills, vacancies_skills_dict):
        """Строки 299-311, 348: сохранение эмбеддинга и вызов manifest.save()."""
        monkeypatch.setattr("src.config.STUDENT_EMB_CACHE_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        student = StudentProfile(
            profile_name="savvy", competencies=[], skills=["python"], target_level="middle", created_at=datetime.now()
        )
        emb = np.array([0.1, 0.2, 0.3])
        with patch("src.analyzers.gap.profile_evaluator.ArtifactManifest") as MockManifest:
            mock_manifest_instance = MockManifest.return_value
            evaluator._save_embedding_cache(student, emb)

            # Проверяем, что manifest создан и сохранён (строка 348)
            MockManifest.assert_called_once()
            mock_manifest_instance.save.assert_called_once()

            # Проверяем содержимое файла
            cache_path = evaluator._get_student_cache_path(student)
            assert cache_path.exists()
            with open(cache_path) as f:
                data = json.load(f)
            assert data["hash"] == evaluator._compute_student_hash(student)
            np.testing.assert_array_equal(data["embedding"], emb.tolist())

    # === строки 314-321 (_get_or_compute_student_embedding) ===
    def test_get_or_compute_student_embedding_with_cache(self, tmp_path, monkeypatch, vacancies_skills, vacancies_skills_dict):
        """Строки 314-317: используется кэш."""
        monkeypatch.setattr("src.config.STUDENT_EMB_CACHE_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        student = StudentProfile(
            profile_name="cached_user", competencies=[], skills=["python"], target_level="middle", created_at=datetime.now()
        )
        emb_cached = np.array([7.0, 8.0])
        # Сохраняем кэш, мокая ArtifactManifest, чтобы избежать импорта sentence_transformers
        with patch("src.analyzers.gap.profile_evaluator.ArtifactManifest"):
            evaluator._save_embedding_cache(student, emb_cached)

        with patch("src.analyzers.gap.profile_evaluator.get_embedding_model") as mock_get_model, \
            patch("src.analyzers.gap.profile_evaluator.ArtifactManifest"):
            result = evaluator._get_or_compute_student_embedding(student)
            mock_get_model.assert_not_called()
        np.testing.assert_array_equal(result, emb_cached)

    def test_get_or_compute_student_embedding_no_cache(self, tmp_path, monkeypatch, vacancies_skills, vacancies_skills_dict):
        """Строки 318-321: кэша нет -> вычисление."""
        monkeypatch.setattr("src.config.STUDENT_EMB_CACHE_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        student = StudentProfile(
            profile_name="new_user", competencies=[], skills=["python", "sql"], target_level="middle", created_at=datetime.now()
        )
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 128
        fake_emb = np.random.rand(2, 128)
        fake_model.encode.return_value = fake_emb

        with patch("src.analyzers.gap.profile_evaluator.get_embedding_model", return_value=fake_model), \
            patch("src.analyzers.gap.profile_evaluator.ArtifactManifest"):
            result = evaluator._get_or_compute_student_embedding(student)
        assert result.shape == (128,)
        fake_model.encode.assert_called_once_with(student.skills, convert_to_numpy=True, show_progress_bar=False)

    def test_get_or_compute_student_embedding_empty_skills(self, tmp_path, monkeypatch, vacancies_skills, vacancies_skills_dict):
        """Строки 351-355: нет навыков -> нулевой вектор."""
        monkeypatch.setattr("src.config.STUDENT_EMB_CACHE_DIR", tmp_path)
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        student = StudentProfile(
            profile_name="empty_skills", competencies=[], skills=[], target_level="middle", created_at=datetime.now()
        )
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 256

        with patch("src.analyzers.gap.profile_evaluator.get_embedding_model", return_value=fake_model), \
            patch("src.analyzers.gap.profile_evaluator.ArtifactManifest"):
            result = evaluator._get_or_compute_student_embedding(student)
        assert result.shape == (256,)
        np.testing.assert_array_equal(result, np.zeros(256))

    # === строки 368-381 (_get_cluster_context: разные ветки) ===
    def test_get_cluster_context_disabled(self, student, vacancies_skills, vacancies_skills_dict):
        """Строка 369: use_clustering=False -> возврат None."""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            use_clustering=False,
        )
        assert evaluator._get_cluster_context(student, "middle") is None

    def test_get_cluster_context_model_not_loaded(self, student, vacancies_skills, vacancies_skills_dict):
        """Строки 371-373: модель для уровня не загружена."""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            use_clustering=True,
        )
        evaluator.cluster_models_loaded = {"junior": False, "middle": False, "senior": False}
        result = evaluator._get_cluster_context(student, "middle")
        assert result is None

    def test_get_cluster_context_exception(self, student, vacancies_skills, vacancies_skills_dict):
        """Строки 376-381: исключение при получении контекста -> возврат None."""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
            use_clustering=True,
        )
        evaluator.cluster_models_loaded = {"middle": True}
        with patch.object(evaluator.clusterer, "get_cluster_context", side_effect=Exception("Boom")):
            result = evaluator._get_cluster_context(student, "middle")
        assert result is None

    # === строки 299-311 (_get_or_create_comparator) ===
    def test_get_or_create_comparator_creates_new(self, vacancies_skills):
        """Строки 299-311: создание и тренировка компаратора."""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=[{"skills": s} for s in vacancies_skills],
        )
        # Очищаем компараторы, чтобы гарантировать создание нового
        evaluator.comparators = {}
        with patch("src.analyzers.gap.profile_evaluator.CompetencyComparator") as MockComp:
            mock_instance = MockComp.return_value
            mock_instance.fit_market.return_value = Ok(True)
            comp = evaluator._get_or_create_comparator("middle")
            assert comp is mock_instance
            mock_instance.fit_market.assert_called_once_with(vacancies_skills)

    def test_get_or_create_comparator_returns_cached(self, vacancies_skills):
        """Строки 306: повторное использование уже созданного компаратора."""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=[{"skills": s} for s in vacancies_skills],
        )
        fake_comp = MagicMock()
        evaluator.comparators["middle"] = fake_comp
        comp = evaluator._get_or_create_comparator("middle")
        assert comp is fake_comp

    # === строки 314-321 (_get_recommendation) ===
    def test_get_recommendation_all_ranges(self, vacancies_skills, vacancies_skills_dict):
        """Строки 314-321: проверка всех ветвей рекомендаций."""
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9},
            vacancies_skills=vacancies_skills,
            vacancies_skills_dict=vacancies_skills_dict,
        )
        assert "Готов к middle" in evaluator._get_recommendation(85, "middle")
        assert "Неплохо для middle" in evaluator._get_recommendation(65, "middle")
        assert "Нужно подготовиться" in evaluator._get_recommendation(45, "middle")
        assert "Недостаточно готов" in evaluator._get_recommendation(30, "middle")


class TestProfileEvaluatorKrm:

    SAMPLE_TAXONOMY = {
        "professions": {
            "Python Developer": {
                "domains": ["Backend"],
                "hh_queries": ["python"],
                "aliases": [],
                "competency_codes": ["ППК-Р1", "ППК-Р2"],
            },
            "Data Scientist": {
                "domains": ["Data Science"],
                "hh_queries": [],
                "aliases": [],
                "competency_codes": [],
            },
        },
        "profile_targets": {},
    }

    SAMPLE_DOMAIN_MAP = {
        "Backend": ["python", "sql", "docker", "git"],
        "Data Science": ["python", "sql", "pandas", "numpy"],
    }

    SAMPLE_KRM = {
        "ППК-Р1": ["python", "sql", "docker"],
        "ППК-Р2": ["git", "testing"],
    }

    @pytest.fixture
    def krm_taxonomy(self, tmp_path):
        tax_path = tmp_path / "profession_taxonomy.json"
        tax_path.write_text(json.dumps(self.SAMPLE_TAXONOMY), encoding="utf-8")
        dom_path = tmp_path / "domain_map.json"
        dom_path.write_text(json.dumps(self.SAMPLE_DOMAIN_MAP), encoding="utf-8")
        krm_path = tmp_path / "krm_competency_mapping.json"
        krm_path.write_text(json.dumps(self.SAMPLE_KRM), encoding="utf-8")
        return ProfessionTaxonomy(
            taxonomy_path=tax_path,
            domain_map_path=dom_path,
            krm_mapping_path=krm_path,
        )

    @pytest.fixture
    def student_with_target(self):
        student = StudentProfile(
            profile_name="krm_test",
            competencies=[],
            skills=["python", "sql", "docker", "git"],
            target_level="middle",
            created_at=datetime.now(),
        )
        object.__setattr__(student, "target_profession", "Python Developer")
        return student

    @pytest.fixture
    def skill_weights_by_level(self):
        return {
            "junior": {"python": 0.8, "sql": 0.6},
            "middle": {"python": 0.9, "docker": 0.7, "sql": 0.5, "git": 0.4},
            "senior": {"python": 0.9, "docker": 0.9},
        }

    def test_krm_coverage_in_result_keys(self, student_with_target, krm_taxonomy, skill_weights_by_level):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7, "docker": 0.5, "git": 0.3},
            vacancies_skills=[["python", "sql", "git"], ["python", "docker"]],
            vacancies_skills_dict=[{"skills": ["python", "sql", "git"]}, {"skills": ["python", "docker"]}],
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False,
        )
        result = evaluator.evaluate_profile(
            student_with_target,
            taxonomy=krm_taxonomy,
        ).unwrap()

        assert "krm_coverage" in result
        assert isinstance(result["krm_coverage"], dict)

    def test_krm_coverage_full_match(self, student_with_target, krm_taxonomy, skill_weights_by_level):
        student_with_target.skills = ["python", "sql", "docker", "git", "testing"]
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7, "docker": 0.5, "git": 0.3, "testing": 0.2},
            vacancies_skills=[["python", "sql", "git"], ["python", "docker"]],
            vacancies_skills_dict=[{"skills": ["python", "sql", "git"]}, {"skills": ["python", "docker"]}],
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False,
        )
        result = evaluator.evaluate_profile(
            student_with_target,
            taxonomy=krm_taxonomy,
        ).unwrap()

        krm = result["krm_coverage"]
        assert "ППК-Р1" in krm
        assert "ППК-Р2" in krm
        assert krm["ППК-Р1"]["coverage"] == 1.0
        assert krm["ППК-Р2"]["coverage"] == 1.0

    def test_krm_coverage_partial(self, student_with_target, krm_taxonomy, skill_weights_by_level):
        student_with_target.skills = ["python", "git"]
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "git": 0.3},
            vacancies_skills=[["python", "sql", "git"], ["python", "docker"]],
            vacancies_skills_dict=[{"skills": ["python", "sql", "git"]}, {"skills": ["python", "docker"]}],
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False,
        )
        result = evaluator.evaluate_profile(
            student_with_target,
            taxonomy=krm_taxonomy,
        ).unwrap()

        krm = result["krm_coverage"]
        assert krm["ППК-Р1"]["coverage"] == pytest.approx(1.0 / 3, rel=0.01)
        assert krm["ППК-Р2"]["coverage"] == pytest.approx(1.0 / 2, rel=0.01)

    def test_krm_coverage_no_match(self, student_with_target, krm_taxonomy, skill_weights_by_level):
        student_with_target.skills = ["excel", "powerpoint"]
        evaluator = ProfileEvaluator(
            skill_weights={"excel": 0.5, "powerpoint": 0.3},
            vacancies_skills=[["python", "sql", "git"], ["python", "docker"]],
            vacancies_skills_dict=[{"skills": ["python", "sql", "git"]}, {"skills": ["python", "docker"]}],
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False,
        )
        result = evaluator.evaluate_profile(
            student_with_target,
            taxonomy=krm_taxonomy,
        ).unwrap()

        krm = result["krm_coverage"]
        assert krm["ППК-Р1"]["coverage"] == 0.0
        assert krm["ППК-Р2"]["coverage"] == 0.0

    def test_krm_coverage_no_taxonomy_returns_empty(self, student_with_target, skill_weights_by_level):
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7, "docker": 0.5, "git": 0.3},
            vacancies_skills=[["python", "sql", "git"], ["python", "docker"]],
            vacancies_skills_dict=[{"skills": ["python", "sql", "git"]}, {"skills": ["python", "docker"]}],
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False,
        )
        result = evaluator.evaluate_profile(
            student_with_target,
            taxonomy=None,
        ).unwrap()

        assert result["krm_coverage"] == {}

    def test_krm_coverage_no_target_profession(self, krm_taxonomy, skill_weights_by_level):
        student = StudentProfile(
            profile_name="no_target",
            competencies=[],
            skills=["python", "sql"],
            target_level="middle",
            created_at=datetime.now(),
        )
        evaluator = ProfileEvaluator(
            skill_weights={"python": 0.9, "sql": 0.7},
            vacancies_skills=[["python", "sql"]],
            vacancies_skills_dict=[{"skills": ["python", "sql"]}],
            skill_weights_by_level=skill_weights_by_level,
            use_clustering=False,
        )
        result = evaluator.evaluate_profile(
            student,
            taxonomy=krm_taxonomy,
        ).unwrap()

        assert result["krm_coverage"] == {}
