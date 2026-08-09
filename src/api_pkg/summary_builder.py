"""Сборка сводки профилей из per-profile full_recommendations_*.json."""

import json

import structlog

from src import config

logger = structlog.get_logger("summary_builder")

SUMMARY_DETAIL_KEYS = (
    "target_profession",
    "dominant_domain_name",
    "closest_roles",
    "gaps",
    "domain_coverage",
    "recommendations",
)


def load_recommendations_from_disk() -> dict[str, dict]:
    """Загрузить все full_recommendations_*.json из data/result/*/."""
    recs = {}
    result_dir = config.DATA_RESULT_DIR
    if not result_dir.exists():
        return recs
    for rec_file in sorted(result_dir.glob("*/full_recommendations_*.json")):
        pname = rec_file.parent.name
        try:
            with open(rec_file, encoding="utf-8") as f:
                recs[pname] = json.load(f)
        except Exception:
            logger.warning("recommendation_load_failed", path=str(rec_file))
    return recs


def build_summary_payload(recs: dict[str, dict]) -> dict:
    """Собрать обогащённую сводку из пер-профильных рекомендаций."""
    evaluations = {}
    for pname, rec in recs.items():
        ev = dict(rec.get("summary") or {})
        for key in SUMMARY_DETAIL_KEYS:
            if rec.get(key) is not None:
                ev[key] = rec[key]
        evaluations[pname] = ev
    return evaluations
