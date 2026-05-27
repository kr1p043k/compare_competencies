from pathlib import Path
from typing import Any

import structlog

from src import Err, Ok, config
from src.predictors.base import RankingPredictor, RecommenderPredictor
from src.predictors.models import RecommendationResult, SkillImpact

logger = structlog.get_logger(__name__)


def create_ranking_predictor(
    model_path: Path | None = None,
    use_ltr: bool = True,
) -> RankingPredictor | None:
    if not use_ltr:
        return None
    from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine

    engine = LTRRecommendationEngine(model_path=model_path)
    if model_path and model_path.exists():
        match engine.load_model(model_path):
            case Ok(_):
                logger.info("ranking_model_loaded", path=str(model_path))
            case Err(err):
                logger.warning("ranking_model_load_failed", error=str(err))
    elif config.MODELS_DIR.exists():
        default_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
        if default_path.exists():
            match engine.load_model(default_path):
                case Ok(_):
                    logger.info("ranking_model_loaded", path=str(default_path))
                case Err(err):
                    logger.warning("ranking_model_load_failed", error=str(err))
    return engine


def create_recommender(
    use_ltr: bool = True,
    use_llm: bool = False,
    profile_evaluator: Any = None,
    trend_analyzer: Any = None,
    ranking_predictor: RankingPredictor | None = None,
) -> "RecommendationEngine":
    from src.predictors.recommendation_engine import RecommendationEngine

    engine = RecommendationEngine(
        use_ltr=False,
        use_llm=use_llm,
        profile_evaluator=profile_evaluator,
        trend_analyzer=trend_analyzer,
    )
    if ranking_predictor is not None and isinstance(ranking_predictor, RankingPredictor):
        engine.ltr_engine = ranking_predictor
    elif use_ltr:
        ltr = create_ranking_predictor()
        if ltr is not None:
            engine.ltr_engine = ltr
    return engine
