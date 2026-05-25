"""Trends analysis."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.analyzers.skills.trends import TrendAnalyzer
from src.models.api_responses import TrendsResponse

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
    try:
        trends = trend_analyzer_instance.get_trending_skills(
            top_n=top_n, min_change_percent=min_change
        )
        return {"trends": trends}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
