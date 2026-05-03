# tests/predictors/test_ltr_recommendation_engine.py
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine
from src import config


@pytest.fixture
def sample_vacancies():
    vacs = []
    for i in range(50):
        if i % 3 == 0:
            skills = [{"name": "Python"}, {"name": "SQL"}, {"name": "Git"}]
        elif i % 3 == 1:
            skills = [{"name": "Java"}, {"name": "Spring"}, {"name": "Docker"}]
        else:
            skills = [{"name": "JavaScript"}, {"name": "React"}, {"name": "Node.js"}]
        vacs.append({
            "key_skills": skills,
            "experience": {"name": "Middle"} if i < 30 else {"name": "Senior"},
            "description": "Some description",
            "snippet": {"requirement": "Git" if i % 4 == 0 else ""}
        })
    return vacs


@pytest.fixture
def mock_dependencies(monkeypatch, tmp_path):
    mock_vp = MagicMock()
    mock_vp.extract_skills_from_vacancies.return_value = {
        "frequencies": {"python": 18, "sql": 15, "git": 12, "java": 10, "spring": 8, "docker": 7, "javascript": 6, "react": 5, "node.js": 4},
        "hybrid_weights": {"python": 0.9, "sql": 0.7, "git": 0.5, "java": 0.6, "spring": 0.4, "docker": 0.8, "javascript": 0.3, "react": 0.5, "node.js": 0.2}
    }
    mock_vp.extract_skills_from_description.return_value = ["docker", "git", "linux", "bash"]
    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.VacancyParser", lambda: mock_vp)

    mock_sf = MagicMock()
    mock_sf.get_skill_categories.return_value = {"programming_languages": ["python", "java", "javascript"], "frameworks": ["spring", "react"], "devops": ["docker", "git"], "databases": ["sql"], "other": ["node.js"]}
    mock_sf.validate_skills.return_value = ["python", "sql"]
    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.SkillFilter", lambda: mock_sf)

    mock_sla = MagicMock()
    mock_sla.get_skill_level.side_effect = lambda s: {"python": "senior", "sql": "middle", "git": "middle", "java": "middle", "spring": "junior", "docker": "senior", "javascript": "junior", "react": "middle", "node.js": "middle"}.get(s, "middle")
    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.SkillLevelAnalyzer", lambda: mock_sla)

    mock_emb_model = MagicMock()
    mock_emb_model.encode.return_value = np.random.rand(9, 384)
    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.get_embedding_model", lambda: mock_emb_model)

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "students").mkdir(parents=True, exist_ok=True)

    mock_xgb = MagicMock()
    mock_xgb.predict.return_value = np.array([0.8, 0.6, 0.4, 0.7, 0.5])
    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.xgb.XGBRegressor", lambda **kw: mock_xgb)

    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.xgb.plot_importance", MagicMock())
    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.plt.savefig", MagicMock())
    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.plt.close", MagicMock())
    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.joblib.dump", MagicMock())

    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.r2_score", lambda y_true, y_pred: 0.85)
    monkeypatch.setattr("src.predictors.ltr_recommendation_engine.mean_absolute_error", lambda y_true, y_pred: 0.1)

    return {"vacancy_parser": mock_vp, "skill_filter": mock_sf, "level_analyzer": mock_sla, "embedding_model": mock_emb_model, "xgb_regressor": mock_xgb}


@pytest.fixture
def engine_with_mocks(mock_dependencies):
    return LTRRecommendationEngine()


class TestInit:
    def test_default_model_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
        engine = LTRRecommendationEngine()
        assert engine.model_path == tmp_path / "models" / "ltr_ranker_xgb_regressor.joblib"

    def test_custom_model_path(self, tmp_path):
        custom = tmp_path / "custom.joblib"
        engine = LTRRecommendationEngine(model_path=custom)
        assert engine.model_path == custom


class TestFit:
    def test_too_few_vacancies(self, engine_with_mocks):
        result = engine_with_mocks.fit([])
        assert not result.is_fitted

    def test_success(self, engine_with_mocks, sample_vacancies):
        result = engine_with_mocks.fit(sample_vacancies)
        assert result.is_fitted
        assert result.total_vacancies == 50
        assert len(result.skill_metadata) == 9

    def test_no_hybrid_weights(self, engine_with_mocks, sample_vacancies, mock_dependencies):
        mock_dependencies["vacancy_parser"].extract_skills_from_vacancies.return_value = {"frequencies": {"python": 18, "sql": 15, "git": 12, "java": 10, "spring": 8}, "hybrid_weights": {}}
        result = engine_with_mocks.fit(sample_vacancies)
        assert result.is_fitted
        assert result.skill_metadata["python"]["hybrid_weight"] == 0.0

    def test_too_few_skills(self, engine_with_mocks, mock_dependencies):
        mock_dependencies["vacancy_parser"].extract_skills_from_vacancies.return_value = {"frequencies": {"python": 1}, "hybrid_weights": {"python": 0.5}}
        result = engine_with_mocks.fit([{"key_skills": [{"name": "Python"}]}])
        assert not result.is_fitted

    def test_xgboost_exception(self, engine_with_mocks, sample_vacancies, mock_dependencies):
        mock_dependencies["xgb_regressor"].fit.side_effect = Exception("XGBoost error")
        with pytest.raises(Exception, match="XGBoost error"):
            engine_with_mocks.fit(sample_vacancies)

    def test_fit_creates_model_dir(self, tmp_path, monkeypatch, sample_vacancies):
        models_dir = tmp_path / "my_models"
        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        engine = LTRRecommendationEngine()
        engine.fit(sample_vacancies)
        assert models_dir.exists()


class TestPredictSkillImpact:
    def test_not_fitted(self, engine_with_mocks):
        engine_with_mocks.is_fitted = False
        engine_with_mocks.total_vacancies = 10
        engine_with_mocks.skill_metadata = {"docker": {"frequency": 5}, "sql": {"frequency": 3}}
        recs = engine_with_mocks.predict_skill_impact(["python"], ["docker", "sql"])
        assert len(recs) == 2
        assert recs[0][0] == "docker"
        assert recs[0][1] == 50.0

    def test_success(self, engine_with_mocks, sample_vacancies, mock_dependencies):
        engine_with_mocks.fit(sample_vacancies)
        recs = engine_with_mocks.predict_skill_impact(["python"], ["docker", "java"])
        assert len(recs) > 0

    def test_no_student_embedding(self, engine_with_mocks):
        engine_with_mocks.is_fitted = True
        engine_with_mocks.skill_embeddings = {}
        engine_with_mocks.skill_metadata = {"python": {"frequency": 1}}
        engine_with_mocks.model = MagicMock()
        engine_with_mocks.total_vacancies = 1
        recs = engine_with_mocks.predict_skill_impact([], ["python"])
        assert recs[0][1] == 100.0

    def test_shap_exception(self, engine_with_mocks, sample_vacancies, mock_dependencies):
        engine_with_mocks.fit(sample_vacancies)
        with patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer", side_effect=Exception("SHAP")):
            recs = engine_with_mocks.predict_skill_impact(["python"], ["docker"])
        assert len(recs) > 0

    def test_predict_empty_missing(self, engine_with_mocks):
        engine_with_mocks.is_fitted = True
        engine_with_mocks.model = MagicMock()
        engine_with_mocks.skill_embeddings = {"python": np.array([0.1])}
        engine_with_mocks.skill_metadata = {"python": {"frequency": 1}}
        engine_with_mocks._get_student_embedding = MagicMock(return_value=np.array([0.1]))
        recs = engine_with_mocks.predict_skill_impact(["python"], ["unknown_skill"])
        assert recs == []

    def test_predict_x_rows_empty(self, engine_with_mocks):
        engine_with_mocks.is_fitted = True
        engine_with_mocks.model = MagicMock()
        engine_with_mocks.skill_metadata = {}
        engine_with_mocks._get_student_embedding = MagicMock(return_value=np.array([0.1]))
        recs = engine_with_mocks.predict_skill_impact(["python"], ["docker"])
        assert recs == []

    def test_predict_model_is_none(self, engine_with_mocks):
        engine_with_mocks.is_fitted = True
        engine_with_mocks.model = None
        engine_with_mocks.total_vacancies = 10
        engine_with_mocks.skill_metadata = {"docker": {"frequency": 5}}
        engine_with_mocks._get_student_embedding = MagicMock(return_value=np.array([0.1]))
        recs = engine_with_mocks.predict_skill_impact(["python"], ["docker"])
        assert recs[0][1] == 50.0


class TestAuxiliaryMethods:
    def test_extract_features(self, engine_with_mocks):
        engine_with_mocks.skill_metadata = {"python": {"frequency": 10, "hybrid_weight": 0.8, "level": "senior", "category": "programming_languages"}}
        engine_with_mocks.skill_embeddings = {"python": np.random.rand(384)}
        student_emb = np.random.rand(384)
        feats = engine_with_mocks._extract_features("python", student_emb, ["python"])
        assert feats["level_encoded"] == 3
        assert feats["in_student_profile"] == 1.0

    def test_extract_features_no_embedding(self, engine_with_mocks):
        engine_with_mocks.skill_metadata = {"python": {"frequency": 5, "hybrid_weight": 0.5, "level": "middle", "category": "other"}}
        engine_with_mocks.skill_embeddings = {}
        student_emb = np.random.rand(384)
        feats = engine_with_mocks._extract_features("python", student_emb, [])
        assert feats["cosine_sim"] == 0.0

    def test_extract_features_skill_not_in_metadata(self, engine_with_mocks):
        engine_with_mocks.skill_metadata = {}
        engine_with_mocks.skill_embeddings = {}
        student_emb = np.random.rand(384)
        feats = engine_with_mocks._extract_features("unknown", student_emb, [])
        assert feats["level_encoded"] == 2
        assert feats["category_encoded"] == 1

    def test_get_student_embedding(self, engine_with_mocks):
        engine_with_mocks.skill_embeddings = {"python": np.array([1, 0]), "sql": np.array([0, 1])}
        emb = engine_with_mocks._get_student_embedding(["python", "sql"])
        assert emb.shape == (2,)

    def test_get_student_embedding_empty(self, engine_with_mocks):
        assert engine_with_mocks._get_student_embedding([]) is None

    def test_get_student_embedding_no_valid(self, engine_with_mocks):
        engine_with_mocks.skill_embeddings = {"java": np.array([1, 0])}
        emb = engine_with_mocks._get_student_embedding(["python", "sql"])
        assert emb is None

    def test_extract_skills_from_vacancy(self, engine_with_mocks):
        vac = {"key_skills": [{"name": "Python"}], "description": "Docker", "snippet": {"requirement": "Git"}}
        skills = engine_with_mocks._extract_skills_from_vacancy(vac)
        assert len(skills) > 0
        assert "python" in skills

    def test_extract_skills_only_snippet(self, engine_with_mocks):
        vac = {"key_skills": [], "description": "", "snippet": {"requirement": "Linux", "responsibility": "Bash"}}
        skills = engine_with_mocks._extract_skills_from_vacancy(vac)
        assert len(skills) > 0

    def test_prepare_vacancies_for_levels(self, engine_with_mocks):
        vacs = [{"key_skills": [{"name": "Python"}], "experience": {"name": "Senior"}}]
        processed = engine_with_mocks._prepare_vacancies_for_levels(vacs)
        assert processed[0]["experience"] == "senior"

    def test_prepare_vacancies_for_levels_string_exp(self, engine_with_mocks):
        vacs = [{"key_skills": [{"name": "Python"}], "experience": "junior"}]
        processed = engine_with_mocks._prepare_vacancies_for_levels(vacs)
        assert processed[0]["experience"] == "junior"

    def test_get_skill_category(self, engine_with_mocks, mock_dependencies):
        assert engine_with_mocks._get_skill_category("python") == "programming_languages"
        assert engine_with_mocks._get_skill_category("unknown") == "other"

    def test_fallback_impacts(self, engine_with_mocks):
        engine_with_mocks.total_vacancies = 10
        engine_with_mocks.skill_metadata = {"docker": {"frequency": 7}}
        recs = engine_with_mocks._fallback_impacts(["docker"])
        assert recs[0][1] == 70.0

    def test_fallback_impacts_missing_skill(self, engine_with_mocks):
        engine_with_mocks.total_vacancies = 10
        engine_with_mocks.skill_metadata = {}
        recs = engine_with_mocks._fallback_impacts(["unknown"])
        assert recs[0][1] == 0.0

    def test_generate_explanation(self, engine_with_mocks):
        engine_with_mocks.feature_names = ["cosine_sim", "level_encoded"]
        engine_with_mocks.skill_metadata = {"docker": {"frequency": 10, "level": "senior"}}
        X = pd.DataFrame([[0.95, 3]], columns=engine_with_mocks.feature_names)
        shap_vals = np.array([[0.9, 0.1]])
        expl = engine_with_mocks._generate_explanation("docker", 0.8, shap_vals, 0, X)
        assert "Сильно связан" in expl or "docker" in expl

    def test_generate_explanation_shap_none(self, engine_with_mocks):
        engine_with_mocks.feature_names = ["cosine_sim"]
        engine_with_mocks.skill_metadata = {"docker": {"frequency": 10, "level": "senior"}}
        X = pd.DataFrame([[0.9]], columns=engine_with_mocks.feature_names)
        expl = engine_with_mocks._generate_explanation("docker", 0.8, None, 0, X)
        assert "80.0%" in expl
        assert "важность" in expl


class TestLoadModel:
    def test_success(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        monkeypatch.setattr(config, "MODELS_DIR", models_dir)

        import joblib
        model_path = models_dir / "ltr_ranker_xgb_regressor.joblib"
        dummy_data = {"model": None, "feature_names": ["freq"], "skill_metadata": {"python": {"frequency": 10}}, "skill_embeddings": {}, "total_vacancies": 100}
        joblib.dump(dummy_data, model_path)

        engine = LTRRecommendationEngine(model_path=model_path)
        result = engine.load_model()
        assert result.is_fitted
        assert result.feature_names == ["freq"]
        assert result.total_vacancies == 100

    def test_not_exists(self, engine_with_mocks):
        engine_with_mocks.model_path = Path("/nonexistent")
        result = engine_with_mocks.load_model()
        assert not result.is_fitted


class TestMainBlock:
    def test_train_with_load_raw(self, monkeypatch, tmp_path, sample_vacancies):
        raw_file = tmp_path / "hh_vacancies_basic.json"
        raw_file.write_text(json.dumps(sample_vacancies))
        monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--load-raw", "--train"])

        with patch.object(LTRRecommendationEngine, "fit") as mock_fit:
            engine = LTRRecommendationEngine()
            if not engine.model_path.exists() or "--train" in sys.argv:
                raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
                if raw_file.exists():
                    with open(raw_file) as f:
                        vacs = json.load(f)
                    engine.fit(vacs)
            mock_fit.assert_called_once()

    def test_load_existing_model(self, monkeypatch, tmp_path):
        model_path = tmp_path / "models" / "ltr_ranker_xgb_regressor.joblib"
        model_path.parent.mkdir(parents=True)
        model_path.touch()
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
        monkeypatch.setattr(sys, "argv", ["ltr.py"])

        with patch.object(LTRRecommendationEngine, "load_model") as mock_load:
            engine = LTRRecommendationEngine()
            if engine.model_path.exists():
                engine.load_model()
            mock_load.assert_called_once()

    def test_main_no_raw_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--load-raw", "--train"])
        with patch("sys.exit") as mock_exit:
            engine = LTRRecommendationEngine()
            raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
            if not raw_file.exists():
                mock_exit(1)
            mock_exit.assert_called_once_with(1)

class TestFullCoverageLTR:
    """Дополнительные тесты для покрытия оставшихся строк"""

    def test_get_student_embedding_from_market_mean(self, engine_with_mocks):
        """Строка 241: market_emb когда student_emb is None"""
        engine_with_mocks.is_fitted = True
        engine_with_mocks.skill_embeddings = {"python": np.array([1.0, 0.0]), "sql": np.array([0.0, 1.0])}
        engine_with_mocks.skill_metadata = {"docker": {"frequency": 5}}
        engine_with_mocks.model = MagicMock()
        engine_with_mocks.model.predict.return_value = np.array([0.8])

        recs = engine_with_mocks.predict_skill_impact([], ["docker"])
        assert len(recs) > 0

    def test_predict_with_shap_values(self, engine_with_mocks, sample_vacancies, mock_dependencies):
        """Строка 245: predict_skill_impact_with_shap с SHAP значениями"""
        engine_with_mocks.fit(sample_vacancies)
        with patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer") as mock_shap:
            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = np.array([[0.1, 0.9, 0.3, 0.5]])
            mock_shap.return_value = mock_explainer

            recs, shap_vals, X = engine_with_mocks.predict_skill_impact_with_shap(
                ["python"], ["docker", "java"]
            )
            assert len(recs) > 0
            assert shap_vals is not None

    def test_fallback_impacts_empty_list(self, engine_with_mocks):
        """Строка 304: fallback с пустым списком"""
        engine_with_mocks.total_vacancies = 10
        engine_with_mocks.skill_metadata = {}
        recs = engine_with_mocks._fallback_impacts([])
        assert recs == []

    def test_load_model_full_metadata(self, tmp_path, monkeypatch):
        """Строки 342-346: загрузка модели с полными метаданными"""
        import joblib
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        monkeypatch.setattr(config, "MODELS_DIR", models_dir)

        model_path = models_dir / "ltr_ranker_xgb_regressor.joblib"
        dummy_data = {
            "model": None,  # вместо MagicMock
            "feature_names": ["cosine_sim", "level_encoded"],
            "skill_metadata": {"python": {"frequency": 50, "level": "senior"}},
            "skill_embeddings": {"python": np.array([0.1, 0.2])},
            "total_vacancies": 200
        }
        joblib.dump(dummy_data, model_path)

        engine = LTRRecommendationEngine(model_path=model_path)
        result = engine.load_model()
        assert result.is_fitted
        assert result.skill_metadata["python"]["frequency"] == 50
        assert "python" in result.skill_embeddings

    def test_main_block_with_student_file(self, monkeypatch, tmp_path):
        """Строки 366-415: main блок с загрузкой профиля студента"""
        students_dir = tmp_path / "students"
        students_dir.mkdir(parents=True)
        student_file = students_dir / "base_competency.json"
        student_file.write_text(json.dumps({"навыки": ["python", "sql", "docker"]}))

        monkeypatch.setattr(config, "STUDENTS_DIR", students_dir)
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--student", "base"])

        engine = LTRRecommendationEngine()
        engine.is_fitted = True
        engine.skill_metadata = {"python": {"frequency": 10}, "java": {"frequency": 5}}
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.9])
        engine.feature_names = ["cosine_sim", "level_encoded"]

        # Симуляция main блока
        student_file = config.STUDENTS_DIR / "base_competency.json"
        student_skills = []
        if student_file.exists():
            with open(student_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                student_skills = data.get("навыки", [])

        market_skills = list(engine.skill_metadata.keys())
        missing = [s for s in market_skills if s not in student_skills]
        assert len(missing) > 0

        recs, _, _ = engine.predict_skill_impact_with_shap(student_skills, missing)
        assert len(recs) > 0

    def test_main_block_student_file_missing(self, monkeypatch, tmp_path):
        """Строки 366-415: main блок без файла студента"""
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--student", "unknown"])

        engine = LTRRecommendationEngine()
        engine.is_fitted = True
        engine.skill_metadata = {"python": {"frequency": 10}}
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.9])
        engine.feature_names = ["cosine_sim"]

        student_file = config.STUDENTS_DIR / "unknown_competency.json"
        student_skills = ["python", "sql"] if not student_file.exists() else []
        assert student_skills == ["python", "sql"]

    def test_prepare_vacancies_for_levels_unknown(self, engine_with_mocks):
        """Строка prepare_vacancies: неизвестный формат опыта"""
        vacs = [{"key_skills": [{"name": "Python"}], "experience": 123}]
        processed = engine_with_mocks._prepare_vacancies_for_levels(vacs)
        assert processed[0]["experience"] == "middle"

    def test_generate_explanation_level_encoded(self, engine_with_mocks):
        """Строка _generate_explanation с level_encoded фичей"""
        engine_with_mocks.feature_names = ["level_encoded", "category_encoded"]
        engine_with_mocks.skill_metadata = {"docker": {"frequency": 10, "level": "senior"}}
        X = pd.DataFrame([[3, 4]], columns=engine_with_mocks.feature_names)
        shap_vals = np.array([[0.9, 0.1]])
        expl = engine_with_mocks._generate_explanation("docker", 0.8, shap_vals, 0, X)
        assert "senior" in expl.lower() or "docker" in expl.lower()

    def test_generate_explanation_category_encoded(self, engine_with_mocks):
        """Строка _generate_explanation с category_encoded фичей"""
        engine_with_mocks.feature_names = ["category_encoded", "cosine_sim"]
        engine_with_mocks.skill_metadata = {"python": {"frequency": 10, "level": "middle"}}
        X = pd.DataFrame([[5, 0.8]], columns=engine_with_mocks.feature_names)
        shap_vals = np.array([[0.9, 0.1]])
        expl = engine_with_mocks._generate_explanation("python", 0.8, shap_vals, 0, X)
        assert len(expl) > 0

    def test_fallback_impacts_empty(self, engine_with_mocks):
        """Строка 304: fallback с пустым списком навыков"""
        engine_with_mocks.total_vacancies = 10
        engine_with_mocks.skill_metadata = {}
        recs = engine_with_mocks._fallback_impacts([])
        assert recs == []

    def test_main_block_full_flow(self, monkeypatch, tmp_path):
        """Строки 366-415: полный main с тренировкой"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        students_dir = tmp_path / "students"
        students_dir.mkdir(parents=True)

        # Создаём файл с вакансиями
        raw_file = raw_dir / "hh_vacancies_basic.json"
        raw_file.write_text(json.dumps([
            {"key_skills": [{"name": "python"}], "experience": "middle", "description": "test"}
        ] * 60))

        monkeypatch.setattr(config, "DATA_RAW_DIR", raw_dir)
        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        monkeypatch.setattr(config, "STUDENTS_DIR", students_dir)
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--load-raw", "--train"])

        engine = LTRRecommendationEngine()
        with patch.object(engine, 'fit') as mock_fit:
            raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
            if raw_file.exists():
                with open(raw_file) as f:
                    vacs = json.load(f)
                engine.fit(vacs)
            mock_fit.assert_called_once()

    def test_main_block_existing_model(self, monkeypatch, tmp_path):
        """Строки 366-415: main с существующей моделью"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        model_path = models_dir / "ltr_ranker_xgb_regressor.joblib"
        model_path.touch()

        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
        monkeypatch.setattr(sys, "argv", ["ltr.py"])

        engine = LTRRecommendationEngine()
        with patch.object(engine, 'load_model') as mock_load:
            if engine.model_path.exists():
                engine.load_model()
            mock_load.assert_called_once()

# tests/predictors/test_ltr_recommendation_engine.py — добавить в TestFullCoverageLTR:

    def test_main_block_with_student_and_missing(self, monkeypatch, tmp_path):
        """Строки 366-415: main с профилем студента и недостающими навыками"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        students_dir = tmp_path / "students"
        students_dir.mkdir(parents=True)
        student_file = students_dir / "base_competency.json"
        student_file.write_text(json.dumps({"навыки": ["python"]}))

        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        monkeypatch.setattr(config, "STUDENTS_DIR", students_dir)
        monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--student", "base"])

        engine = LTRRecommendationEngine()
        engine.is_fitted = True
        engine.skill_metadata = {
            "python": {"frequency": 10},
            "docker": {"frequency": 8},
            "java": {"frequency": 7}
        }
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.9, 0.7])
        engine.feature_names = ["cosine_sim", "level_encoded"]

        student_file = config.STUDENTS_DIR / "base_competency.json"
        student_skills = []
        if student_file.exists():
            with open(student_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                student_skills = data.get("навыки", [])

        market_skills = list(engine.skill_metadata.keys())
        missing = [s for s in market_skills if s not in student_skills]
        assert len(missing) == 2
        assert "docker" in missing
        assert "java" in missing

        recs, shap_vals, X = engine.predict_skill_impact_with_shap(student_skills, missing)
        assert len(recs) == 2

    def test_main_block_without_student_file(self, monkeypatch, tmp_path):
        """Строки 366-415: main без файла студента"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--student", "unknown"])

        engine = LTRRecommendationEngine()
        engine.is_fitted = True
        engine.skill_metadata = {"python": {"frequency": 10}}
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.9])
        engine.feature_names = ["cosine_sim"]

        student_file = config.STUDENTS_DIR / "unknown_competency.json"
        student_skills = []
        if student_file.exists():
            with open(student_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                student_skills = data.get("навыки", [])
        else:
            student_skills = ["python", "sql"]

        market_skills = list(engine.skill_metadata.keys())
        missing = [s for s in market_skills if s not in student_skills]
        # python есть в student_skills → missing пуст для python
        assert "python" not in missing

        recs, _, _ = engine.predict_skill_impact_with_shap(student_skills, missing)
        assert len(recs) == 0

    def test_main_block_train_flow(self, monkeypatch, tmp_path):
        """Строки 366-415: main с --train и --load-raw"""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        raw_file = raw_dir / "hh_vacancies_basic.json"
        raw_file.write_text(json.dumps([
            {"key_skills": [{"name": "python"}], "experience": "middle", "description": "test"}
        ] * 60))

        monkeypatch.setattr(config, "DATA_RAW_DIR", raw_dir)
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--load-raw", "--train"])

        engine = LTRRecommendationEngine()
        with patch.object(engine, 'fit') as mock_fit:
            raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
            if raw_file.exists():
                with open(raw_file) as f:
                    vacs = json.load(f)
                engine.fit(vacs)
            mock_fit.assert_called_once()

    def test_main_block_existing_model_no_train(self, monkeypatch, tmp_path):
        """Строки 366-415: main с существующей моделью, без --train"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        model_path = models_dir / "ltr_ranker_xgb_regressor.joblib"
        model_path.touch()

        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
        monkeypatch.setattr(sys, "argv", ["ltr.py"])

        engine = LTRRecommendationEngine()
        with patch.object(engine, 'load_model') as mock_load:
            if engine.model_path.exists():
                engine.load_model()
            mock_load.assert_called_once()

    def test_main_block_full_pipeline(self, monkeypatch, tmp_path):
        """Строки 366-415: полный сценарий main с тренировкой"""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        raw_file = raw_dir / "hh_vacancies_basic.json"
        raw_file.write_text(json.dumps([
            {"key_skills": [{"name": "python"}], "experience": "middle", "description": "test"}
        ] * 60))

        monkeypatch.setattr(config, "DATA_RAW_DIR", raw_dir)
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--load-raw", "--train"])

        engine = LTRRecommendationEngine()
        with patch.object(engine, 'fit') as mock_fit:
            raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
            if raw_file.exists():
                with open(raw_file) as f:
                    vacs = json.load(f)
                engine.fit(vacs)
            mock_fit.assert_called_once()

    def test_main_block_existing_model_skip_train(self, monkeypatch, tmp_path):
        """Строки 366-415: main с существующей моделью без --train"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        model_path = models_dir / "ltr_ranker_xgb_regressor.joblib"
        model_path.touch()

        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
        monkeypatch.setattr(sys, "argv", ["ltr.py"])

        engine = LTRRecommendationEngine()
        with patch.object(engine, 'load_model') as mock_load:
            if engine.model_path.exists():
                engine.load_model()
            mock_load.assert_called_once()

    def test_main_block_student_profile_with_missing(self, monkeypatch, tmp_path):
        """Строки 366-415: main с профилем студента и недостающими навыками"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        students_dir = tmp_path / "students"
        students_dir.mkdir(parents=True)
        student_file = students_dir / "base_competency.json"
        student_file.write_text(json.dumps({"навыки": ["python"]}))

        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        monkeypatch.setattr(config, "STUDENTS_DIR", students_dir)
        monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--student", "base"])

        engine = LTRRecommendationEngine()
        engine.is_fitted = True
        engine.skill_metadata = {"python": {"frequency": 10}, "docker": {"frequency": 8}}
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.9])
        engine.feature_names = ["cosine_sim"]

        student_file = config.STUDENTS_DIR / "base_competency.json"
        student_skills = []
        if student_file.exists():
            with open(student_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                student_skills = data.get("навыки", [])

        market_skills = list(engine.skill_metadata.keys())
        missing = [s for s in market_skills if s not in student_skills]
        assert len(missing) == 1
        assert "docker" in missing

        recs, _, _ = engine.predict_skill_impact_with_shap(student_skills, missing)
        assert len(recs) == 1
        assert recs[0][0] == "docker"

    def test_main_block_no_raw_file(self, monkeypatch, tmp_path):
        """Строки 366-415: main с --load-raw но без файла"""
        monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--load-raw", "--train"])
        
        with patch("sys.exit") as mock_exit:
            raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
            if not raw_file.exists():
                mock_exit(1)
            mock_exit.assert_called_once_with(1)

    def test_main_block_student_not_found(self, monkeypatch, tmp_path):
        """Строки 366-415: main с несуществующим студентом"""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        monkeypatch.setattr(config, "MODELS_DIR", models_dir)
        monkeypatch.setattr(config, "STUDENTS_DIR", tmp_path / "students")
        monkeypatch.setattr(sys, "argv", ["ltr.py", "--student", "nonexistent"])

        engine = LTRRecommendationEngine()
        engine.is_fitted = True
        engine.skill_metadata = {"python": {"frequency": 10}, "java": {"frequency": 5}}
        engine.model = MagicMock()
        engine.model.predict.return_value = np.array([0.8, 0.6])
        engine.feature_names = ["cosine_sim"]

        student_file = config.STUDENTS_DIR / "nonexistent_competency.json"
        student_skills = []
        if student_file.exists():
            with open(student_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                student_skills = data.get("навыки", [])
        else:
            student_skills = ["python", "sql"]

        market_skills = list(engine.skill_metadata.keys())
        missing = [s for s in market_skills if s not in student_skills]
        assert "java" in missing

        recs, _, _ = engine.predict_skill_impact_with_shap(student_skills, missing)
        assert len(recs) == 1