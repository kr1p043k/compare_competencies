# tests/conftest.py
import pytest
import json
from pathlib import Path
from pydantic import ValidationError

# Точные импорты под реальную структуру проекта
from src.models.student import StudentProfile
from src.models.vacancy import Vacancy
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.embedding_comparator import EmbeddingComparator
from src.analyzers.skill_filter import SkillFilter
from src.loaders_student.student_loader import StudentLoader
#from src.predictors.skill_forecast import SkillForecast
from src.predictors.ml_recommendation_engine import MLRecommendationEngine

from src.visualization.charts import (
    plot_radar_chart,
    plot_heatmap,
    plot_coverage_comparison,
    plot_top_deficits,
    plot_skill_trends,
)

# ====================== ОБЩИЕ ФИКСТУРЫ ======================

@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def sample_skill_weights() -> dict:
    """Пример весов навыков (используется GapAnalyzer и SkillFilter)"""
    return {
        "python": 2450,
        "sql": 1890,
        "docker": 1420,
        "fastapi": 980,
        "machine learning": 870,
        "pandas": 650,
        "git": 1200,
        "postgresql": 1100,
        "frontend": 50,          # будет отфильтровано
        "разработка": 10         # generic — будет удалено
    }


@pytest.fixture(scope="session")
def sample_student() -> StudentProfile:
    return StudentProfile(
        student_id="test",
        name="Анна Иванова",
        competencies=["Python", "SQL", "Git", "Pandas", "Docker"],
        target_role="Data Scientist"
    )


@pytest.fixture(scope="session")
def sample_vacancies() -> list[Vacancy]:
    return [
        Vacancy(title="Junior DS", skills=["Python", "SQL", "Pandas", "Machine Learning"]),
        Vacancy(title="Middle Python", skills=["Python", "FastAPI", "Docker", "PostgreSQL"])
    ]


# ====================== АНАЛИЗАТОРЫ ======================

@pytest.fixture
def gap_analyzer(sample_skill_weights):
    return GapAnalyzer(skill_weights=sample_skill_weights)


@pytest.fixture
def skill_filter():
    return SkillFilter()


@pytest.fixture
def student_loader():
    return StudentLoader()


@pytest.fixture
def embedding_comparator():
    return EmbeddingComparator()


#@pytest.fixture
#def skill_forecast():
#    return SkillForecast()


@pytest.fixture
def ml_recommender():
    return MLRecommendationEngine()