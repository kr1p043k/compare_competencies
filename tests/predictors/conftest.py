"""Общие фикстуры тестов LTR-движка.

`engine_with_mocks` нужен нескольким модулям (`test_ltr_recommendation_engine`,
`test_shap_topk_alignment`), поэтому живёт в conftest: фикстуры из одного
тест-модуля pytest в другой не отдаёт, и SHAP-тесты падали на setup с
«fixture 'engine_with_mocks' not found», то есть не запускались вообще.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import Ok
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine


@pytest.fixture
def mock_vacancy_parser():
    parser = MagicMock()
    parser.extract_skills_from_vacancies.return_value = Ok({
        "frequencies": {"python": 50, "sql": 40, "docker": 30, "java": 20, "c++": 15},
        "hybrid_weights": {"python": 0.9, "sql": 0.8, "docker": 0.7, "java": 0.6, "c++": 0.5},
    })
    parser.extract_skills_from_description.return_value = Ok([])
    return parser


@pytest.fixture
def mock_skill_filter():
    sf = MagicMock()
    sf.get_skill_categories.return_value = Ok({})
    return sf


@pytest.fixture
def mock_level_analyzer():
    la = MagicMock()
    la.get_skill_level.return_value = "middle"
    la.analyze_vacancies.return_value = None
    return la


@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.encode.return_value = np.random.rand(5, 128)
    model.get_sentence_embedding_dimension.return_value = 128
    return model


@pytest.fixture
def engine_with_mocks(mock_vacancy_parser, mock_skill_filter, mock_level_analyzer, mock_embedding_model, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with patch("src.predictors.ltr_recommendation_engine.VacancyParser", return_value=mock_vacancy_parser), \
         patch("src.predictors.ltr_recommendation_engine.SkillFilter", return_value=mock_skill_filter), \
         patch("src.predictors.ltr_recommendation_engine.SkillLevelAnalyzer", return_value=mock_level_analyzer), \
         patch("src.predictors.ltr_recommendation_engine.get_embedding_model", return_value=mock_embedding_model), \
         patch("src.predictors.ltr_recommendation_engine.config.MODELS_DIR", models_dir):
        engine = LTRRecommendationEngine()
        engine.skill_embeddings = {
            skill: np.random.rand(128) for skill in ["python", "sql", "docker", "java", "c++"]
        }
        engine.skill_metadata = {
            skill: {
                "frequency": f, "hybrid_weight": w, "level": "middle", "category": "devops",
                "hybrid_weight_normalized": w, "freq_normalized": f/50
            }
            for skill, f, w in [("python", 50, 0.9), ("sql", 40, 0.8), ("docker", 30, 0.7),
                                ("java", 20, 0.6), ("c++", 15, 0.5)]
        }
        engine.feature_names = [
            "level_encoded", "category_encoded", "cosine_sim", "in_student_profile",
            "skill_freq_normalized", "co_occurrence", "category_avg_weight", "student_skills_count"
        ]
        engine.category_avg_weight = {"devops": 0.8}
        engine.total_vacancies = 100
        engine.vacancy_skills_corpus = [{"python", "sql"}, {"docker", "sql"}, {"java", "c++"}]
        yield engine
