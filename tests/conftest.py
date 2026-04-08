# tests/conftest.py
import pytest
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import logging
import pytest
import sys
from unittest.mock import MagicMock
from src.models.student import StudentProfile
from src.models.vacancy import Vacancy, Area, Employer, KeySkill
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.embedding_comparator import EmbeddingComparator
from src.analyzers.skill_filter import SkillFilter
from src.loaders_student.student_loader import StudentLoader
from src.parsing.skill_validator import SkillValidator
from src.analyzers.profile_evaluator import ProfileEvaluator
from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer
from src.visualization.charts import (
    plot_radar_chart,
    plot_heatmap,
    plot_coverage_comparison,
    plot_top_deficits,
)
def pytest_configure(config):
    """Глобальный мок SentenceTransformer для всех тестов."""
    # Создаём мок-модель, которая возвращает случайные эмбеддинги
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(10, 384)
    mock_model.eval.return_value = mock_model

    # Подменяем модуль sentence_transformers в sys.modules
    mock_st = MagicMock()
    mock_st.SentenceTransformer = MagicMock(return_value=mock_model)
    sys.modules['sentence_transformers'] = mock_st

    # Также патчим конкретные модули проекта, если они уже загружены
    try:
        import src.parsing.vacancy_parser as vp
        vp.SentenceTransformer = lambda *a, **k: mock_model
    except ImportError:
        pass
    try:
        import src.analyzers.embedding_comparator as ec
        ec.SentenceTransformer = lambda *a, **k: mock_model
    except ImportError:
        pass

    # Подавляем логи от httpx/httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
@pytest.fixture(autouse=True)
def suppress_http_logging():
    """Отключает файловые обработчики и повышает уровень httpx/httpcore логов."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    yield
@pytest.fixture(autouse=True)
def clean_logging_handlers():
    """Удаляет все FileHandler из корневого логгера, чтобы избежать ошибок закрытого файла."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    # Добавляем NullHandler, чтобы подавить вывод, если нужно
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(logging.NullHandler())
    yield
    # Можно восстановить, но для тестов не обязательно
@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent.parent / "data"

@pytest.fixture(scope="session")
def sample_skill_weights() -> dict:
    return {
        "python": 2450, "sql": 1890, "docker": 1420,
        "fastapi": 980, "pandas": 650, "git": 1200,
        "postgresql": 1100, "machine learning": 870
    }

@pytest.fixture(scope="session")
def sample_student() -> StudentProfile:
    """Теперь используем актуальные поля StudentProfile"""
    return StudentProfile(
        profile_name="base",                    # ← обязательно!
        competencies=["SS1.1", "DL-1.3"],
        skills=["Python", "SQL", "Git", "Pandas", "Docker"],
        target_level="middle",
        created_at=datetime.now()
    )

@pytest.fixture(scope="session")
def sample_vacancies() -> list[Vacancy]:
    area = Area(id=1, name="Москва")
    employer = Employer(id="123", name="Test Corp")
    return [
        Vacancy(
            id="1",
            name="Junior Data Scientist",
            area=area,
            employer=employer,
            key_skills=[KeySkill(name="Python"), KeySkill(name="SQL")]
        ),
        Vacancy(
            id="2",
            name="Middle Python Developer",
            area=area,
            employer=employer,
            key_skills=[KeySkill(name="FastAPI"), KeySkill(name="Docker")]
        )
    ]

# Анализаторы
@pytest.fixture
def gap_analyzer(sample_skill_weights):
    return GapAnalyzer(skill_weights=sample_skill_weights)

@pytest.fixture
def embedding_comparator():
    return EmbeddingComparator()   # модель грузится внутри

@pytest.fixture
def skill_filter():
    return SkillFilter()

@pytest.fixture
def student_loader():
    return StudentLoader()

# Для визуализации
@pytest.fixture
def sample_student_skills():
    return ["Python", "SQL", "Git", "Pandas", "Docker"]

@pytest.fixture
def sample_market_top_skills():
    return ["Python", "SQL", "Machine Learning", "FastAPI", "Docker"]

@pytest.fixture
def sample_results_dict():
    return {
        "base": {
            "coverage_percent": 65,
            "weighted_coverage_percent": 72,
            "covered_skills": ["Python", "SQL"],
            "high_demand_gaps": [{"skill": "FastAPI", "frequency": 87}]
        }
    }
    
@pytest.fixture
def validator():
    return SkillValidator()

@pytest.fixture
def mock_embedder(monkeypatch):
    mock = MagicMock()
    mock.encode.return_value = np.random.rand(1, 384)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda *a, **k: mock)

@pytest.fixture
def profile_evaluator():
    return ProfileEvaluator()

@pytest.fixture
def skill_level_analyzer():
    return SkillLevelAnalyzer()

@pytest.fixture
def charts_module():
    """Временная fixture до реализации класса Charts (если он нужен).
    Пока используем существующие plot-функции."""
    import src.visualization.charts as charts
    return charts