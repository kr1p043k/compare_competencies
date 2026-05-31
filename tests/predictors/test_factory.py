from unittest.mock import MagicMock, patch

import pytest

from src import Err, Ok
from src.predictors.base import RankingPredictor
from src.predictors.factory import create_ranking_predictor, create_recommender
from src.predictors.models import RecommendationResult, SkillImpact


class MockRankingPredictor(RankingPredictor):
    def __init__(self):
        self._name = "mock_ranking"

    @property
    def name(self):
        return self._name

    def predict(self, skills, **kw):
        return [SkillImpact(skill=s, impact=0.5) for s in skills]

    def predict_impact(self, skills, **kw):
        return self.predict(skills, **kw)

    def fit(self, *a, **kw):
        return Ok(self)

    def load_model(self, path):
        return Ok(self)


class TestCreateRankingPredictor:
    def test_use_ltr_false(self):
        assert create_ranking_predictor(use_ltr=False) is None

    @patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine")
    def test_use_ltr_no_model_path(self, MockLTR):
        mock_engine = MagicMock()
        MockLTR.return_value = mock_engine
        with patch("src.predictors.factory.config") as mock_config:
            mock_config.MODELS_DIR = MagicMock()
            mock_config.MODELS_DIR.exists.return_value = False
            result = create_ranking_predictor()
            assert result is not None

    @patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine")
    def test_model_path_exists_ok(self, MockLTR):
        mock_engine = MagicMock()
        mock_engine.load_model.return_value = Ok(MagicMock())
        MockLTR.return_value = mock_engine
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        result = create_ranking_predictor(model_path=mock_path)
        assert result is not None

    @patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine")
    def test_model_path_exists_fail(self, MockLTR):
        mock_engine = MagicMock()
        mock_engine.load_model.return_value = Err(Exception("bad model"))
        MockLTR.return_value = mock_engine
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        result = create_ranking_predictor(model_path=mock_path)
        assert result is not None


class TestCreateRecommender:
    @patch("src.predictors.recommendation_engine.RecommendationEngine")
    def test_default_no_ltr(self, MockEngine):
        mock_engine = MagicMock()
        MockEngine.return_value = mock_engine
        engine = create_recommender(use_ltr=False)
        assert engine is not None

    @patch("src.predictors.recommendation_engine.RecommendationEngine")
    def test_with_ranking_predictor(self, MockEngine):
        mock_rp = MockRankingPredictor()
        mock_engine = MagicMock()
        MockEngine.return_value = mock_engine
        engine = create_recommender(use_ltr=True, ranking_predictor=mock_rp)
        assert engine.ltr_engine == mock_rp

    @patch("src.predictors.factory.create_ranking_predictor")
    @patch("src.predictors.recommendation_engine.RecommendationEngine")
    def test_auto_create_ltr(self, MockEngine, mock_create_rp):
        mock_ltr = MockRankingPredictor()
        mock_create_rp.return_value = mock_ltr
        mock_engine = MagicMock()
        MockEngine.return_value = mock_engine
        engine = create_recommender(use_ltr=True)
        assert engine is not None
