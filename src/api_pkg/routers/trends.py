"""Trends analysis."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import func, select

from src import Err, Ok
from src.analyzers.skills.trends import TrendAnalyzer
from src.database import async_session_factory
from src.models.api_responses import TrendsResponse
from src.models.krm_models import Competency, CompetencyTrend

from src.api_pkg import deps

logger = structlog.get_logger("api")

router = APIRouter(tags=["trends"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/api/trends", response_model=TrendsResponse)
@limiter.limit("60/minute")
async def get_trends(
    request: Request,
    top_n: int = Query(15),
    min_change: float = Query(3.0),
    trend_analyzer_instance: TrendAnalyzer = Depends(deps.get_trend_analyzer),
):
    match trend_analyzer_instance.get_trending_skills(
        top_n=top_n, min_change_percent=min_change
    ):
        case Ok(trends):
            return {"trends": trends}
        case Err(err):
            raise HTTPException(status_code=500, detail=str(err))


@router.get("/api/competency-trends")
@limiter.limit("60/minute")
async def get_competency_trends(
    request: Request,
    direction: str | None = Query(None, regex="^(rising|falling|stable)$"),
    limit: int = Query(200, ge=1, le=500),
):
    """Get competency trends from DB, optionally filtered by direction."""
    async with async_session_factory() as session:
        query = (
            select(CompetencyTrend, Competency.code, Competency.name)
            .join(Competency, Competency.id == CompetencyTrend.competency_id)
            .order_by(CompetencyTrend.snapshot_date.desc(), func.abs(CompetencyTrend.change_pct).desc())
        )
        if direction:
            query = query.where(CompetencyTrend.trend_direction == direction)
        rows = await session.execute(query.limit(limit))
        seen: set[str] = set()
        results = []
        for ct, code, name in rows.unique().all():
            if code in seen:
                continue
            seen.add(code)
            results.append({
                "competency_id": str(ct.competency_id),
                "code": code,
                "name": name or "",
                "direction": ct.trend_direction,
                "change_pct": ct.change_pct,
                "skill_count": ct.skill_count,
                "snapshot_date": str(ct.snapshot_date),
            })
        return {"total": len(results), "trends": results}
