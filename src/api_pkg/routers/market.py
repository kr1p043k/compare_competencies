"""Market skills, top skills, skill info."""

import structlog
from fastapi import APIRouter, Depends, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.analyzers.skills.skill_taxonomy import SkillTaxonomy
from src.api_pkg import deps
from src.models.api_responses import (
    SkillInfoResponse,
    TopSkillsResponse,
)

logger = structlog.get_logger("api")

router = APIRouter(tags=["market"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/market/top-skills", response_model=TopSkillsResponse)
@limiter.limit("60/minute")
async def get_top_skills(
    request: Request,
    limit: int = Query(15, ge=1, le=50),
    weights: dict[str, float] = Depends(deps.get_skill_weights),
    freq: dict[str, int] = Depends(deps.get_skill_freq),
):
    """Топ навыков рынка. Вес — нормализованная метрика, frequency — сырое
    число упоминаний в вакансиях (понятнее для UI)."""
    top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {"skills": [{"skill": s, "weight": round(w, 4), "frequency": int(freq.get(s, 0))} for s, w in top]}


@router.get("/market/skill/{skill}", response_model=SkillInfoResponse)
@limiter.limit("60/minute")
async def get_skill_info(
    request: Request,
    skill: str,
    weights: dict[str, float] = Depends(deps.get_skill_weights),
    freq: dict[str, int] = Depends(deps.get_skill_freq),
    taxonomy_instance: SkillTaxonomy | None = Depends(deps.get_taxonomy),
):
    """Информация о навыке (частота, тренд)."""
    weight = weights.get(skill, 0.0)
    freq_val = freq.get(skill, 0)
    category = (
        taxonomy_instance.get_category_label(skill) if taxonomy_instance else "unknown"
    )
    icon = taxonomy_instance.get_category_icon(skill) if taxonomy_instance else ""
    return {
        "skill": skill,
        "frequency": freq_val,
        "weight": round(weight, 4),
        "category": category,
        "icon": icon,
    }


@router.get("/market-competencies", response_model=dict)
@limiter.limit("60/minute")
async def get_market_competencies(
    request: Request,
    weights: dict[str, float] = Depends(deps.get_skill_weights),
):
    """Компетенции рынка + объём выборки (для витрины рынка)."""
    from src.db import get_pool

    top_skills = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:100]
    payload = {
        "skills": [{"skill": s, "weight": w} for s, w in top_skills],
        "total": len(weights),
        "vacancy_count": None,
        "date_from": None,
        "date_to": None,
    }
    try:
        pool = get_pool()
        if pool is not None:
            row = await pool.fetchrow(
                "SELECT COUNT(*) AS n, MIN(published_at)::date AS mn, "
                "MAX(published_at)::date AS mx FROM vacancies "
                "WHERE published_at IS NOT NULL"
            )
            if row:
                payload["vacancy_count"] = int(row["n"] or 0)
                payload["date_from"] = str(row["mn"]) if row["mn"] else None
                payload["date_to"] = str(row["mx"]) if row["mx"] else None
    except Exception as e:
        logger.warning("market_competencies_meta_failed", error=str(e))
    return payload
