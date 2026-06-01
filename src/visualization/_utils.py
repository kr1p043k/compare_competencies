"""Вспомогательные функции загрузки данных."""

import json
from typing import Any

import structlog

from src import config

logger = structlog.get_logger(__name__)


def load_skill_weights() -> dict[str, float]:
    path = config.DATA_PROCESSED_DIR / "skill_weights.json"
    if not path.exists():
        logger.warning("skill_weights_file_not_found", path=str(path))
        return {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        logger.info("skill_weights_loaded", path=str(path), count=len(data))
        return data
    except Exception as e:
        logger.warning("skill_weights_load_failed", path=str(path), error=str(e))
        return {}


def load_hybrid_weights() -> dict[str, float]:
    path = config.DATA_PROCESSED_DIR / "hybrid_weights.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        logger.info("hybrid_weights_loaded", path=str(path), count=len(data))
        return data
    except Exception as e:
        logger.warning("hybrid_weights_load_failed", path=str(path), error=str(e))
        return {}


def load_ml_recommendations(profile: str) -> list[tuple[str, float, str]]:
    possible = [
        f"ltr_recommendations_{profile}.json",
        f"full_recommendations_{profile}.json",
    ]
    for fname in possible:
        path = config.DATA_DIR / "result" / profile / fname
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            recs = []
            for r in data.get("recommendations", []):
                skill = r.get("skill", "")
                score = r.get("importance_score", r.get("importance", r.get("score", 0.0)))
                explanation = r.get("explanation", r.get("why_important", ""))
                recs.append((skill, score, explanation))
            logger.info("ml_recommendations_loaded", profile=profile, count=len(recs))
            return recs
        except Exception as e:
            logger.warning("ml_recommendations_load_failed", path=str(path), error=str(e))
    logger.info("no_ml_recommendations_found", profile=profile)
    return []


def load_profile_evaluation(profile_name: str) -> dict[str, Any] | None:
    summary_path = config.DATA_RESULT_DIR / "profiles_comparison_summary.json"
    if not summary_path.exists():
        logger.warning("summary_file_not_found", path=str(summary_path))
        return None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        eval_data = data.get("evaluations", {}).get(profile_name)
        if eval_data:
            logger.info("profile_evaluation_loaded", profile=profile_name)
        else:
            logger.warning("profile_not_in_summary", profile=profile_name)
        return eval_data
    except Exception as e:
        logger.warning("profile_evaluation_load_failed", profile=profile_name, error=str(e))
        return None
