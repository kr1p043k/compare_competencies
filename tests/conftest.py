# tests/conftest.py
import logging
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pytest
import time

from src import config
from src.analyzers.comparison.domain_analyzer import DomainAnalyzer
from src.analyzers.comparison.embedding_comparator import EmbeddingComparator
from src.analyzers.gap.gap_analyzer import GapAnalyzer
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.analyzers.skills.skill_filter import SkillFilter
from src.analyzers.skills.skill_level_analyzer import SkillLevelAnalyzer
from src.loaders_student.student_loader import StudentLoader
from src.models.student import StudentProfile
from src.models.vacancy import Area, Employer, KeySkill, Vacancy
from src.parsing.skills.skill_validator import SkillValidator

@pytest.fixture(autouse=True)
def mock_sleep():
    """Автоматически подменяет time.sleep на заглушку во всех тестах, кроме помеченных @pytest.mark.no_sleep_mock"""
    with patch("time.sleep", return_value=None) as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_sleeps():
    with patch("time.sleep", return_value=None), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        yield
def pytest_configure(config):
    """Глобальный мок SentenceTransformer для всех тестов."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(10, 384)
    mock_model.eval.return_value = mock_model
    mock_model.get_sentence_embedding_dimension.return_value = 384

    mock_st = MagicMock()
    mock_st.SentenceTransformer = MagicMock(return_value=mock_model)
    mock_st.__version__ = "2.7.0"
    sys.modules["sentence_transformers"] = mock_st

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# =====================================================================
# ГЛОБАЛЬНЫЙ МОК ДИРЕКТОРИЙ — предотвращает создание файлов в data/
# =====================================================================
@pytest.fixture(autouse=True)
def mock_all_data_dirs(tmp_path, monkeypatch):
    """
    Подменяет все директории data/ на временные,
    чтобы тесты НЕ создавали файлы в реальных директориях.
    """
    dirs = {
        "DATA_DIR": tmp_path / "data",
        "DATA_RAW_DIR": tmp_path / "data" / "raw",
        "DATA_PROCESSED_DIR": tmp_path / "data" / "processed",
        "DATA_RESULT_DIR": tmp_path / "data" / "result",
        "HISTORY_DIR": tmp_path / "data" / "history",
        "MODELS_DIR": tmp_path / "data" / "models",
        "STUDENTS_DIR": tmp_path / "data" / "students",
        "EMBEDDINGS_CACHE_DIR": tmp_path / "data" / "embeddings" / "cache",
        "DATA_EMBEDDINGS_DIR": tmp_path / "data" / "embeddings",
    }

    for attr, path in dirs.items():
        if hasattr(config, attr):
            path.mkdir(parents=True, exist_ok=True)
            monkeypatch.setattr(config, attr, path)

    yield


@pytest.fixture(autouse=True)
def suppress_http_logging():
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    yield


@pytest.fixture(autouse=True)
def clean_logging_handlers():
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(logging.NullHandler())
    yield


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def sample_skill_weights() -> dict:
    return {
        "python": 2450,
        "sql": 1890,
        "docker": 1420,
        "fastapi": 980,
        "pandas": 650,
        "git": 1200,
        "postgresql": 1100,
        "machine learning": 870,
    }


@pytest.fixture(scope="session")
def sample_student() -> StudentProfile:
    return StudentProfile(
        profile_name="base",
        competencies=["SS1.1", "DL-1.3"],
        skills=["Python", "SQL", "Git", "Pandas", "Docker"],
        target_level="middle",
        created_at=datetime.now(),
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
            key_skills=[KeySkill(name="Python"), KeySkill(name="SQL")],
        ),
        Vacancy(
            id="2",
            name="Middle Python Developer",
            area=area,
            employer=employer,
            key_skills=[KeySkill(name="FastAPI"), KeySkill(name="Docker")],
        ),
    ]


@pytest.fixture
def gap_analyzer(sample_skill_weights):
    return GapAnalyzer(sample_skill_weights)


@pytest.fixture
def embedding_comparator():
    return EmbeddingComparator()


@pytest.fixture
def skill_filter():
    return SkillFilter()


@pytest.fixture
def student_loader():
    return StudentLoader()


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
            "high_demand_gaps": [{"skill": "FastAPI", "frequency": 87}],
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
def profile_evaluator(sample_skill_weights):
    vacancies_skills = [["python", "sql"], ["fastapi", "docker"]]
    return ProfileEvaluator(
        skill_weights=sample_skill_weights,
        vacancies_skills=vacancies_skills,
        vacancies_skills_dict=[{"skills": s} for s in vacancies_skills],
    )


@pytest.fixture
def skill_level_analyzer():
    return SkillLevelAnalyzer()


@pytest.fixture
def domain_analyzer():
    return DomainAnalyzer()
