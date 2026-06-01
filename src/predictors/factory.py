from pathlib import Path
from typing import Any

import structlog

from src import Err, Ok, Result, config
from src.errors import DomainError
from src.predictors.base import RankingPredictor, RecommenderPredictor
from src.predictors.models import RecommendationResult, SkillImpact

logger = structlog.get_logger(__name__)


def create_ranking_predictor(
    model_path: Path | None = None,
    use_ltr: bool = True,
) -> Result[RankingPredictor, DomainError]:
    if not use_ltr:
        return Err(DomainError(message="LTR disabled"))
    from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine

    engine = LTRRecommendationEngine(model_path=model_path)
    if model_path and model_path.exists():
        match engine.load_model(model_path):
            case Ok(_):
                logger.info("ranking_model_loaded", path=str(model_path))
            case Err(err):
                return Err(DomainError(message="Ranking model load failed", detail=str(err)))
    elif config.MODELS_DIR.exists():
        default_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
        if default_path.exists():
            match engine.load_model(default_path):
                case Ok(_):
                    logger.info("ranking_model_loaded", path=str(default_path))
                case Err(err):
                    return Err(DomainError(message="Ranking model load failed", detail=str(err)))
    return Ok(engine)


def create_recommender(
    use_ltr: bool = True,
    use_llm: bool = False,
    profile_evaluator: Any = None,
    trend_analyzer: Any = None,
    ranking_predictor: RankingPredictor | None = None,
) -> Result["RecommendationEngine", DomainError]:
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
        match create_ranking_predictor():
            case Ok(ltr):
                engine.ltr_engine = ltr
            case Err(err):
                logger.warning("ltr_engine_creation_failed", error=str(err))
    return Ok(engine)
