# tests/predictors/test_ltr_recommendation_engine.py
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import sys
sys.modules['shap'] = MagicMock()
sys.modules['cv2'] = MagicMock()

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src import Ok
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine, _SYNTHETIC_DOMAIN_PROFILES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_vacancy_parser():
    parser = MagicMock()
    parser.extract_skills_from_vacancies.return_value = {
        "frequencies": {"python": 50, "sql": 40, "docker": 30, "java": 20, "c++": 15},
        "hybrid_weights": {"python": 0.9, "sql": 0.8, "docker": 0.7, "java": 0.6, "c++": 0.5},
    }
    parser.extract_skills_from_description.return_value = []
    return parser


@pytest.fixture
def mock_skill_filter():
    sf = MagicMock()
    sf.get_skill_categories.return_value = {}
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


# ---------------------------------------------------------------------------
# Tests for __init__
# ---------------------------------------------------------------------------
class TestLTRInit:
    def test_default_init(self, tmp_path):
        with patch("src.predictors.ltr_recommendation_engine.get_embedding_model") as mock_get_emb:
            mock_get_emb.return_value = MagicMock()
            with patch("src.predictors.ltr_recommendation_engine.config.MODELS_DIR", tmp_path / "models"):
                engine = LTRRecommendationEngine()
                assert engine.is_fitted is False
                assert engine.model is None

    def test_custom_model_path(self, tmp_path):
        custom = tmp_path / "custom.pkl"
        with patch("src.predictors.ltr_recommendation_engine.get_embedding_model") as mock_get_emb:
            mock_get_emb.return_value = MagicMock()
            engine = LTRRecommendationEngine(model_path=custom)
            assert engine.model_path == custom


# ---------------------------------------------------------------------------
# Tests for fit()
# ---------------------------------------------------------------------------
class TestLTRFit:
    def _make_mock_train_test_split(self):
        """возвращает мок, корректно обрабатывающий два вызова train_test_split"""
        def side_effect(*args, **kwargs):
            X = args[0]
            y = args[1]
            # первый вызов: test_size=0.3 -> возвращает X_train, X_tmp, y_train, y_tmp
            if kwargs.get("test_size") == 0.3:
                return X, X, y, y
            # второй вызов: test_size=0.5 -> возвращает X_val, X_test, y_val, y_test
            return X, X, y, y
        return side_effect

    def test_fit_success(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.vacancy_parser.extract_skills_from_vacancies.return_value = {
            "frequencies": {"python": 50, "sql": 40, "docker": 30, "java": 20, "c++": 15},
            "hybrid_weights": {"python": 0.9, "sql": 0.8, "docker": 0.7, "java": 0.6, "c++": 0.5},
        }
        vacancies = [{"key_skills": [{"name": "python"}, {"name": "sql"}], "experience": "middle"}] * 100

        with patch("src.predictors.ltr_recommendation_engine.train_test_split",
                side_effect=self._make_mock_train_test_split()), \
            patch("src.predictors.ltr_recommendation_engine.xgb.XGBRegressor") as MockXGB, \
            patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer"), \
            patch("src.predictors.ltr_recommendation_engine.plt.figure"), \
            patch("src.predictors.ltr_recommendation_engine.plt.savefig"), \
            patch("src.predictors.ltr_recommendation_engine.plt.close"), \
            patch("src.predictors.ltr_recommendation_engine.joblib.dump") as mock_dump, \
            patch("src.predictors.ltr_recommendation_engine.ArtifactManifest"), \
            patch("src.predictors.ltr_recommendation_engine.r2_score", return_value=0.9), \
            patch("src.predictors.ltr_recommendation_engine.mean_absolute_error", return_value=0.1), \
            patch("src.predictors.ltr_recommendation_engine.ndcg_score", return_value=0.8):
            mock_instance = MockXGB.return_value
            mock_instance.predict.return_value = np.array([0.5] * 100)
            engine.fit(vacancies)

        assert engine.is_fitted is True
        mock_dump.assert_called_once()

    def test_fit_importance_plot_exception(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.vacancy_parser.extract_skills_from_vacancies.return_value = {
            "frequencies": {"python": 50, "sql": 40, "docker": 30, "java": 20, "c++": 15},
            "hybrid_weights": {"python": 0.9, "sql": 0.8, "docker": 0.7, "java": 0.6, "c++": 0.5},
        }
        vacancies = [{"key_skills": [{"name": "python"}, {"name": "sql"}]}] * 100

        with patch("src.predictors.ltr_recommendation_engine.train_test_split",
                side_effect=self._make_mock_train_test_split()), \
            patch("src.predictors.ltr_recommendation_engine.xgb.XGBRegressor") as MockXGB, \
            patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer"), \
            patch("src.predictors.ltr_recommendation_engine.plt.figure"), \
            patch("src.predictors.ltr_recommendation_engine.plt.savefig", side_effect=Exception("save error")), \
            patch("src.predictors.ltr_recommendation_engine.plt.close"), \
            patch("src.predictors.ltr_recommendation_engine.joblib.dump"), \
            patch("src.predictors.ltr_recommendation_engine.ArtifactManifest"), \
            patch("src.predictors.ltr_recommendation_engine.r2_score", return_value=0.9), \
            patch("src.predictors.ltr_recommendation_engine.mean_absolute_error", return_value=0.1), \
            patch("src.predictors.ltr_recommendation_engine.ndcg_score", return_value=0.8):
            mock_instance = MockXGB.return_value
            mock_instance.predict.return_value = np.array([0.5] * 100)
            engine.fit(vacancies)

        assert engine.is_fitted is True

    def test_fit_insufficient_vacancies(self, engine_with_mocks):
        engine = engine_with_mocks
        vacancies = [{"key_skills": [{"name": "python"}], "experience": "middle"}] * 10
        engine.fit(vacancies)
        assert engine.is_fitted is False

    def test_fit_too_few_skills(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.vacancy_parser.extract_skills_from_vacancies.return_value = {
            "frequencies": {"python": 10},
            "hybrid_weights": {"python": 0.9},
        }
        vacancies = [{"key_skills": [{"name": "python"}]}] * 100
        engine.fit(vacancies)
        assert engine.is_fitted is False

    # в классе TestLTRFit
    def test_fit_ndcg_nan(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.vacancy_parser.extract_skills_from_vacancies.return_value = {
            "frequencies": {"python": 50, "sql": 40, "docker": 30, "java": 20, "c++": 15},
            "hybrid_weights": {"python": 0.9, "sql": 0.8, "docker": 0.7, "java": 0.6, "c++": 0.5},
        }
        vacancies = [{"key_skills": [{"name": "python"}, {"name": "sql"}]}] * 100

        with patch("src.predictors.ltr_recommendation_engine.train_test_split",
                side_effect=self._make_mock_train_test_split()), \
            patch("src.predictors.ltr_recommendation_engine.xgb.XGBRegressor") as MockXGB, \
            patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer"), \
            patch("src.predictors.ltr_recommendation_engine.plt.figure"), \
            patch("src.predictors.ltr_recommendation_engine.plt.savefig"), \
            patch("src.predictors.ltr_recommendation_engine.plt.close"), \
            patch("src.predictors.ltr_recommendation_engine.joblib.dump"), \
            patch("src.predictors.ltr_recommendation_engine.ArtifactManifest"), \
            patch("src.predictors.ltr_recommendation_engine.r2_score", return_value=0.9), \
            patch("src.predictors.ltr_recommendation_engine.mean_absolute_error", return_value=0.1), \
            patch("src.predictors.ltr_recommendation_engine.ndcg_score", return_value=float("nan")), \
            patch("src.predictors.ltr_recommendation_engine.logger") as mock_logger:
            mock_instance = MockXGB.return_value
            mock_instance.predict.return_value = np.array([0.5] * 100)
            engine.fit(vacancies)
        # Проверяем, что залогировалось ndcg_at_10='n/a'
        mock_logger.info.assert_any_call(
            "ltr_training_completed",
            r2=0.9, mae=0.1, ndcg_at_10="n/a",
            train_samples=ANY, val_samples=ANY, test_samples=ANY
        )

# ---------------------------------------------------------------------------
# Tests for predict_skill_impact & with_shap
# ---------------------------------------------------------------------------
class TestLTRPredict:
    def test_predict_single_missing_skill(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.5])
        result = engine.predict_skill_impact_with_shap(["python"], ["docker"])
        assert result.is_ok()
        recs, shap, X = result.unwrap()
        assert len(recs) == 1

    def test_predict_not_fitted(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = False
        result = engine.predict_skill_impact(["python"], ["docker"])
        assert result.is_err()

    def test_predict_fitted(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.5])
        result = engine.predict_skill_impact(["python"], ["docker"])
        assert result.is_ok()
        recs = result.unwrap()
        assert recs[0][0] == "docker"

    def test_predict_with_shap(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.5])
        with patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer") as mock_explainer:
            mock_explainer.return_value.shap_values.return_value = np.array([[0.1, 0.2]])
            result = engine.predict_skill_impact_with_shap(["python"], ["docker"])
        assert result.is_ok()
        recs, shap_vals, X = result.unwrap()
        assert len(recs) == 1
        assert shap_vals is not None

    def test_predict_shap_fails(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.5])
        with patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer",
                   side_effect=Exception("shap fail")):
            result = engine.predict_skill_impact_with_shap(["python"], ["docker"])
        assert result.is_ok()
        recs, shap_vals, X = result.unwrap()
        assert shap_vals is None

    def test_predict_student_emb_none(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.skill_embeddings = {}
        engine.model = MagicMock()
        result = engine.predict_skill_impact(["python"], ["docker"])
        assert result.is_err()
        assert "No embeddings" in result.err().message

    def test_predict_empty_missing(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        result = engine.predict_skill_impact_with_shap(["python"], [])
        assert result.is_ok()
        recs, shap_vals, X = result.unwrap()
        assert recs == []

    def test_fallback_impacts(self, engine_with_mocks):
        engine = engine_with_mocks
        impacts = engine._fallback_impacts(["docker", "fastapi"])
        assert len(impacts) == 2
        assert impacts[0][0] == "docker"

    def test_predict_missing_skill_not_in_metadata(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        result = engine.predict_skill_impact_with_shap(["python"], ["nonexistent"])
        assert result.is_ok()
        recs, shap_vals, X = result.unwrap()
        assert recs == []

    # extra: студент с пустыми навыками
    def test_predict_with_empty_student_skills(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.5])
        result = engine.predict_skill_impact([], ["docker"])
        assert result.is_ok()
        recs = result.unwrap()
        assert len(recs) == 1

    def test_predict_single_missing_skill(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.5])
        result = engine.predict_skill_impact_with_shap(["python"], ["docker"])
        assert result.is_ok()
        recs, shap, X = result.unwrap()
        # один навык → scores = raw_scores (без softmax)
        assert len(recs) == 1

    def test_predict_with_shap_empty_missing_returns_empty(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        result = engine.predict_skill_impact_with_shap(["python"], [])
        assert result.is_ok()
        recs, shap_vals, X = result.unwrap()
        assert recs == []
        assert shap_vals is None
        assert X is None

    def test_predict_single_skill_no_softmax(self, engine_with_mocks):
        engine = engine_with_mocks
        engine.is_fitted = True
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.5])
        result = engine.predict_skill_impact_with_shap(["python"], ["docker"])
        assert result.is_ok()
        recs, shap_vals, X = result.unwrap()
        # один навык -> scores = raw_scores (без softmax)
        assert len(recs) == 1


# ---------------------------------------------------------------------------
# Tests for _extract_features
# ---------------------------------------------------------------------------
class TestLTRExtractFeatures:
    def test_extract_features_basic(self, engine_with_mocks):
        engine = engine_with_mocks
        student_emb = np.random.rand(128)
        features = engine._extract_features("docker", student_emb, ["python", "sql"])
        assert "cosine_sim" in features
        assert features["in_student_profile"] == 0.0

    def test_extract_features_skill_missing_embedding(self, engine_with_mocks):
        engine = engine_with_mocks
        features = engine._extract_features("nonexistent", np.zeros(128), [])
        assert features["cosine_sim"] == 0.0

    def test_extract_features_student_emb_none(self, engine_with_mocks):
        engine = engine_with_mocks
        features = engine._extract_features("docker", None, ["python"])
        assert features["cosine_sim"] == 0.0


# ---------------------------------------------------------------------------
# Tests for _get_student_embedding
# ---------------------------------------------------------------------------
class TestLTRStudentEmbedding:
    def test_empty_skills(self, engine_with_mocks):
        assert engine_with_mocks._get_student_embedding([]) is None

    def test_no_valid_skills(self, engine_with_mocks):
        engine_with_mocks.skill_embeddings = {}
        assert engine_with_mocks._get_student_embedding(["python"]) is None

    def test_valid_skills(self, engine_with_mocks):
        emb = engine_with_mocks._get_student_embedding(["python", "sql"])
        assert emb is not None
        assert emb.shape == (128,)


# ---------------------------------------------------------------------------
# Tests for _co_occurrence_score
# ---------------------------------------------------------------------------
class TestCoOccurrence:
    def test_co_occurrence_empty_skills(self, engine_with_mocks):
        assert engine_with_mocks._co_occurrence_score("docker", []) == 0.0

    def test_co_occurrence_no_corpus(self, engine_with_mocks):
        engine_with_mocks.vacancy_skills_corpus = []
        assert engine_with_mocks._co_occurrence_score("docker", ["python"]) == 0.0

    def test_co_occurrence_normal(self, engine_with_mocks):
        score = engine_with_mocks._co_occurrence_score("docker", ["sql"])
        assert score > 0


# ---------------------------------------------------------------------------
# Tests for _prepare_vacancies_for_levels
# ---------------------------------------------------------------------------
class TestLTRPrepareVacancies:
    def test_prepare_vacancies_junior_senior(self, engine_with_mocks):
        vacs = [
            {"key_skills": [{"name": "python"}], "experience": {"name": "junior"}},
            {"key_skills": [{"name": "sql"}], "experience": {"name": "senior"}},
            {"key_skills": [{"name": "docker"}], "experience": {"name": "middle"}},
        ]
        processed = engine_with_mocks._prepare_vacancies_for_levels(vacs)
        assert processed[0]["experience"] == "junior"
        assert processed[1]["experience"] == "senior"
        assert processed[2]["experience"] == "middle"

    def test_prepare_vacancies_no_experience_field(self, engine_with_mocks):
        vacs = [{"key_skills": [{"name": "python"}]}]
        processed = engine_with_mocks._prepare_vacancies_for_levels(vacs)
        assert processed[0]["experience"] == "middle"

    def test_prepare_vacancies_string_experience(self, engine_with_mocks):
        vacs = [{"key_skills": [{"name": "python"}], "experience": "junior"}]
        processed = engine_with_mocks._prepare_vacancies_for_levels(vacs)
        assert processed[0]["experience"] == "junior"


# ---------------------------------------------------------------------------
# Tests for _extract_skills_from_vacancy
# ---------------------------------------------------------------------------
class TestLTRExtractSkillsFromVacancy:
    def test_key_skills_only(self, engine_with_mocks):
        vac = {"key_skills": [{"name": "Python"}, {"name": "SQL"}], "description": ""}
        skills = engine_with_mocks._extract_skills_from_vacancy(vac)
        assert "python" in skills and "sql" in skills

    def test_description_skills(self, engine_with_mocks):
        engine_with_mocks.vacancy_parser.extract_skills_from_description.return_value = ["docker"]
        vac = {"key_skills": [], "description": "Работа с Docker"}
        skills = engine_with_mocks._extract_skills_from_vacancy(vac)
        assert "docker" in skills

    def test_snippet_fallback(self, engine_with_mocks):
        engine_with_mocks.vacancy_parser.extract_skills_from_description.return_value = ["git"]
        vac = {"key_skills": [], "description": "", "snippet": {"requirement": "Знание git"}}
        skills = engine_with_mocks._extract_skills_from_vacancy(vac)
        assert "git" in skills


# ---------------------------------------------------------------------------
# Tests for _get_skill_category
# ---------------------------------------------------------------------------
class TestLTRGetSkillCategory:
    def test_from_taxonomy(self, engine_with_mocks):
        with patch.object(engine_with_mocks, '_taxonomy', MagicMock()) as mock_tax:
            mock_tax.get_category.return_value = "devops"
            assert engine_with_mocks._get_skill_category("docker") == "devops"

    def test_fallback_to_skill_filter(self, engine_with_mocks):
        engine_with_mocks.skill_filter.get_skill_categories.return_value = {"frontend": ["react"]}
        engine_with_mocks._taxonomy = MagicMock()
        engine_with_mocks._taxonomy.get_category.return_value = "other"
        assert engine_with_mocks._get_skill_category("react") == "frontend"

    def test_unknown(self, engine_with_mocks):
        engine_with_mocks.skill_filter.get_skill_categories.return_value = {}
        engine_with_mocks._taxonomy = MagicMock()
        engine_with_mocks._taxonomy.get_category.return_value = "other"
        assert engine_with_mocks._get_skill_category("zzz") == "other"


# ---------------------------------------------------------------------------
# Tests for _generate_explanation
# ---------------------------------------------------------------------------
class TestLTRGenerateExplanation:
    def test_cosine_sim_explanation(self, engine_with_mocks):
        X = pd.DataFrame([{"cosine_sim": 0.8}])
        shap_vals = np.array([[0.0, 0.9]])  # feature 1 is max
        engine_with_mocks.feature_names = ["level_encoded", "cosine_sim"]
        expl = engine_with_mocks._generate_explanation("docker", 0.7, shap_vals, 0, X)
        assert "🎯" in expl

    def test_co_occurrence_explanation(self, engine_with_mocks):
        X = pd.DataFrame([{"co_occurrence": 0.5}])
        shap_vals = np.array([[0.1, 0.0, 0.8, 0.0]])
        engine_with_mocks.feature_names = ["level_encoded", "cosine_sim", "co_occurrence", "in_student_profile"]
        expl = engine_with_mocks._generate_explanation("docker", 0.7, shap_vals, 0, X)
        assert "🔗" in expl

    def test_level_encoded_explanation(self, engine_with_mocks):
        X = pd.DataFrame([{"level_encoded": 2}])
        shap_vals = np.array([[0.9, 0.1]])
        engine_with_mocks.feature_names = ["level_encoded", "cosine_sim"]
        expl = engine_with_mocks._generate_explanation("docker", 0.7, shap_vals, 0, X)
        assert "📊" in expl

    def test_fallback_explanation(self, engine_with_mocks):
        X = pd.DataFrame([{"cosine_sim": 0.0}])
        shap_vals = np.array([[0.1, 0.2]])
        engine_with_mocks.feature_names = ["unknown", "cosine_sim"]
        expl = engine_with_mocks._generate_explanation("docker", 0.7, shap_vals, 0, X)
        assert "важность 70.0%" in expl

    def test_explanation_no_shap(self, engine_with_mocks):
        expl = engine_with_mocks._generate_explanation("docker", 0.7, None, 0, pd.DataFrame())
        assert "docker: важность 70.0%" in expl

    # дополнительный кейс: skill отсутствует в metadata
    def test_explanation_unknown_skill(self, engine_with_mocks):
        expl = engine_with_mocks._generate_explanation("nonexistent", 0.5, None, 0, pd.DataFrame())
        assert "nonexistent: важность 50.0%" in expl

    def test_explanation_from_skill_metadata_no_shap(self, engine_with_mocks):
        """Покрытие ветки, где shap_values is None (используется fallback)."""
        expl = engine_with_mocks._generate_explanation("docker", 0.7, None, 0, pd.DataFrame())
        assert "важность 70.0%" in expl

    def test_explanation_category_weight(self, engine_with_mocks):
        feature_order = engine_with_mocks.feature_names   # используем актуальный порядок
        X = pd.DataFrame([{name: 0.0 for name in feature_order}])
        # делаем так, чтобы top_idx указывал на category_avg_weight
        idx_of_cat_avg = feature_order.index("category_avg_weight")
        shap_vals = np.zeros((1, len(feature_order)))
        shap_vals[0, idx_of_cat_avg] = 0.9                # доминирует category_avg_weight
        X.iloc[0, idx_of_cat_avg] = 4.0

        expl = engine_with_mocks._generate_explanation("docker", 0.7, shap_vals, 0, X)
        assert "📁" in expl


# ---------------------------------------------------------------------------
# Tests for load_model
# ---------------------------------------------------------------------------
class TestLTRLoadModel:
    @pytest.fixture
    def dummy_model_file(self, tmp_path):
        model = xgb.XGBRegressor()
        model.fit(np.array([[0]]), np.array([0]))
        data = {
            "model": model,
            "feature_names": ["f1"],
            "skill_metadata": {},
            "skill_embeddings": {},
            "total_vacancies": 100,
            "vacancy_skills_corpus": [],
            "category_avg_weight": {},
        }
        filepath = tmp_path / "model.joblib"
        joblib.dump(data, filepath)
        return filepath

    def test_load_model_success(self, engine_with_mocks, dummy_model_file):
        engine_with_mocks.is_fitted = False
        with patch("src.predictors.ltr_recommendation_engine.ArtifactManifest"):
            engine_with_mocks.load_model(dummy_model_file)
        assert engine_with_mocks.is_fitted is True

    def test_load_model_file_missing(self, engine_with_mocks):
        engine_with_mocks.model_path = Path("/nonexistent.pkl")
        engine_with_mocks.is_fitted = False
        with patch("src.predictors.ltr_recommendation_engine.ArtifactManifest"):
            engine_with_mocks.load_model()
        assert engine_with_mocks.is_fitted is False

    def test_load_model_incompatible_manifest(self, engine_with_mocks, dummy_model_file):
        engine_with_mocks.is_fitted = False
        with patch("src.predictors.ltr_recommendation_engine.ArtifactManifest") as MockManifest:
            mock_instance = MockManifest.load.return_value
            mock_instance.is_compatible.return_value = False
            engine_with_mocks.load_model(dummy_model_file)
        assert engine_with_mocks.is_fitted is True

    def test_load_model_manifest_load_fails(self, engine_with_mocks, dummy_model_file):
        engine_with_mocks.is_fitted = False
        with patch("src.predictors.ltr_recommendation_engine.ArtifactManifest") as MockManifest:
            MockManifest.load.side_effect = Exception("bad manifest")
            engine_with_mocks.load_model(dummy_model_file)
        assert engine_with_mocks.is_fitted is True

    def test_load_model_with_manifest_present(self, engine_with_mocks, tmp_path):
        model = xgb.XGBRegressor()
        model.fit(np.array([[0]]), np.array([0]))
        data = {
            "model": model,
            "feature_names": ["f1"],
            "skill_metadata": {},
            "skill_embeddings": {},
            "total_vacancies": 100,
            "vacancy_skills_corpus": [],
            "category_avg_weight": {},
        }
        filepath = tmp_path / "model.joblib"
        joblib.dump(data, filepath)
        manifest_path = filepath.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps({"model_version": "same", "metrics": {}}))
        engine_with_mocks.is_fitted = False
        with patch("src.predictors.ltr_recommendation_engine.ArtifactManifest") as MockManifest:
            mock_inst = MockManifest.load.return_value
            mock_inst.is_compatible.return_value = True
            mock_inst.metrics = {"r2": 0.9}
            engine_with_mocks.load_model(filepath)
        assert engine_with_mocks.is_fitted is True

    def test_load_model_with_manifest_present(self, engine_with_mocks, tmp_path):
        model = xgb.XGBRegressor()
        model.fit(np.array([[0]]), np.array([0]))
        data = {
            "model": model,
            "feature_names": ["f1"],
            "skill_metadata": {},
            "skill_embeddings": {},
            "total_vacancies": 100,
            "vacancy_skills_corpus": [],
            "category_avg_weight": {},
        }
        filepath = tmp_path / "model.joblib"
        joblib.dump(data, filepath)
        manifest_path = filepath.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps({"model_version": "same", "metrics": {}}))
        engine_with_mocks.is_fitted = False
        with patch("src.predictors.ltr_recommendation_engine.ArtifactManifest") as MockManifest:
            mock_inst = MockManifest.load.return_value
            mock_inst.is_compatible.return_value = True
            mock_inst.metrics = {"r2": 0.9}
            engine_with_mocks.load_model(filepath)
        assert engine_with_mocks.is_fitted is True

    # в классе TestLTRLoadModel
    def test_load_model_logs_manifest_verified(self, engine_with_mocks, tmp_path):
        model = xgb.XGBRegressor()
        model.fit(np.array([[0]]), np.array([0]))
        data = {
            "model": model, "feature_names": ["f1"], "skill_metadata": {},
            "skill_embeddings": {}, "total_vacancies": 100,
            "vacancy_skills_corpus": [], "category_avg_weight": {},
        }
        filepath = tmp_path / "model.joblib"
        joblib.dump(data, filepath)
        manifest_path = filepath.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps({"model_version": "same", "metrics": {}}))
        engine_with_mocks.is_fitted = False
        with patch("src.predictors.ltr_recommendation_engine.ArtifactManifest") as MockManifest:
            mock_inst = MagicMock()
            mock_inst.is_compatible.return_value = True
            mock_inst.metrics = {"r2": 0.9}
            MockManifest.load.return_value = Ok(mock_inst)
            with patch("src.predictors.ltr_recommendation_engine.logger") as mock_logger:
                engine_with_mocks.load_model(filepath)
        mock_logger.info.assert_any_call("ltr_manifest_verified", metrics={"r2": 0.9})
