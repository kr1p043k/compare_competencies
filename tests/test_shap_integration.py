"""Tests for SHAP integration in production flow."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine


class TestSHAPIntegration:
    def test_shap_computed_when_enabled(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with patch("src.predictors.ltr_recommendation_engine.get_embedding_model") as mock_emb:
            mock_emb.return_value = MagicMock()
            with patch("src.predictors.ltr_recommendation_engine.config.MODELS_DIR", models_dir):
                engine = LTRRecommendationEngine()
                engine.is_fitted = True
                engine.model = MagicMock()
                engine.model.predict.return_value = np.array([0.5])
                engine.skill_embeddings = {"python": np.random.rand(128), "docker": np.random.rand(128), "sql": np.random.rand(128)}
                engine.skill_metadata = {
                    "docker": {"frequency": 30, "hybrid_weight": 0.7, "level": "middle", "category": "devops",
                               "hybrid_weight_normalized": 0.7, "freq_normalized": 0.6},
                    "python": {"frequency": 50, "hybrid_weight": 0.9, "level": "middle", "category": "programming_languages",
                               "hybrid_weight_normalized": 0.9, "freq_normalized": 1.0},
                }
                engine.feature_names = ["level_encoded", "category_encoded", "cosine_sim", "in_student_profile",
                                        "skill_freq_normalized", "co_occurrence", "category_avg_weight", "student_skills_count"]
                engine.category_avg_weight = {"devops": 0.8}
                engine.total_vacancies = 100
                engine.vacancy_skills_corpus = [{"python", "sql"}, {"docker", "sql"}]
                engine.skill_filter = MagicMock()
                engine.skill_filter.get_skill_categories.return_value = {}
                engine.level_analyzer = MagicMock()
                engine.level_analyzer.get_skill_level.return_value = "middle"
                engine._taxonomy = None

                with patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer") as mock_ex:
                    mock_instance = MagicMock()
                    mock_instance.shap_values.return_value = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0]])
                    mock_ex.return_value = mock_instance

                    result = engine.predict_skill_impact(["python"], ["docker"])

        assert result.is_ok()
        recs = result.unwrap()
        assert len(recs) == 1
        skill, score, explanation = recs[0]
        assert skill == "docker"
        assert score > 0
        assert len(explanation) > 0

    def test_shap_explanation_fallback(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with patch("src.predictors.ltr_recommendation_engine.get_embedding_model") as mock_emb:
            mock_emb.return_value = MagicMock()
            with patch("src.predictors.ltr_recommendation_engine.config.MODELS_DIR", models_dir):
                engine = LTRRecommendationEngine()
                engine.is_fitted = True
                engine.model = MagicMock()
                engine.model.predict.return_value = np.array([0.5])
                engine.skill_embeddings = {"docker": np.random.rand(128)}
                engine.skill_metadata = {
                    "docker": {"frequency": 30, "hybrid_weight": 0.7, "level": "middle", "category": "devops",
                               "hybrid_weight_normalized": 0.7, "freq_normalized": 0.6},
                }
                engine.feature_names = ["level_encoded"]
                engine.category_avg_weight = {}
                engine.total_vacancies = 100
                engine.vacancy_skills_corpus = []
                engine.skill_filter = MagicMock()
                engine.skill_filter.get_skill_categories.return_value = {}
                engine.level_analyzer = MagicMock()
                engine.level_analyzer.get_skill_level.return_value = "middle"
                engine._taxonomy = None

                with patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer",
                           side_effect=Exception("shap_error")):
                    result = engine.predict_skill_impact(["python"], ["docker"])

        assert result.is_ok()
        recs = result.unwrap()
        assert len(recs) == 1
        skill, score, explanation = recs[0]
        assert "docker" in explanation

    def test_cross_domain_explanation_included(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with patch("src.predictors.ltr_recommendation_engine.get_embedding_model") as mock_emb:
            mock_emb.return_value = MagicMock()
            with patch("src.predictors.ltr_recommendation_engine.config.MODELS_DIR", models_dir):
                engine = LTRRecommendationEngine()
                engine.is_fitted = True
                engine.model = MagicMock()
                engine.model.predict.return_value = np.array([0.5])
                engine.skill_embeddings = {"python": np.random.rand(128), "docker": np.random.rand(128)}
                engine.skill_metadata = {
                    "docker": {"frequency": 30, "hybrid_weight": 0.7, "level": "middle", "category": "devops",
                               "hybrid_weight_normalized": 0.7, "freq_normalized": 0.6},
                }
                engine.feature_names = ["level_encoded"]
                engine.category_avg_weight = {}
                engine.total_vacancies = 100
                engine.vacancy_skills_corpus = []
                engine.skill_filter = MagicMock()
                engine.skill_filter.get_skill_categories.return_value = {}
                engine.level_analyzer = MagicMock()
                engine.level_analyzer.get_skill_level.return_value = "middle"
                engine._taxonomy = MagicMock()
                engine._taxonomy.get_category.side_effect = lambda s: {"docker": "devops", "python": "programming_languages"}.get(s, "other")
                engine._taxonomy.get_category_label_by_id.side_effect = lambda c: {"devops": "DevOps", "programming_languages": "Языки программирования"}.get(c, c)
                engine._get_skill_category = MagicMock(side_effect=lambda s: {"docker": "devops", "python": "programming_languages"}.get(s, "other"))

                result = engine.predict_skill_impact_with_shap(["python"], ["docker"], compute_shap=True)

        assert result.is_ok()
        recs, shap_vals, X = result.unwrap()
        assert len(recs) == 1
        skill, score, explanation = recs[0]
        assert "docker" in explanation

    def test_predict_impact_wrapper(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with patch("src.predictors.ltr_recommendation_engine.get_embedding_model") as mock_emb:
            mock_emb.return_value = MagicMock()
            with patch("src.predictors.ltr_recommendation_engine.config.MODELS_DIR", models_dir):
                engine = LTRRecommendationEngine()
                engine.is_fitted = True
                engine.model = MagicMock()
                engine.model.predict.return_value = np.array([0.5, 0.3])
                engine.skill_embeddings = {"docker": np.random.rand(128), "python": np.random.rand(128)}
                engine.skill_metadata = {
                    "docker": {"frequency": 30, "hybrid_weight": 0.7, "level": "middle", "category": "devops",
                               "hybrid_weight_normalized": 0.7, "freq_normalized": 0.6},
                    "python": {"frequency": 50, "hybrid_weight": 0.9, "level": "middle", "category": "programming_languages",
                               "hybrid_weight_normalized": 0.9, "freq_normalized": 1.0},
                }
                engine.feature_names = ["level_encoded"]
                engine.category_avg_weight = {}
                engine.total_vacancies = 100
                engine.vacancy_skills_corpus = []
                engine.skill_filter = MagicMock()
                engine.skill_filter.get_skill_categories.return_value = {}
                engine.level_analyzer = MagicMock()
                engine.level_analyzer.get_skill_level.return_value = "middle"
                engine._taxonomy = None

                result = engine.predict_impact(["python"], ["docker"])
                assert result.is_ok()
                impacts = result.unwrap()
                assert len(impacts) > 0
                assert impacts[0].skill == "docker"
