from typing import Any
import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import Ok, Err, config
from src.predictors.skill_forecast import SkillForecastEngine
from src.utils import safe_read_json

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["forecast"])
limiter = Limiter(key_func=get_remote_address)

_forecast_cache: dict[str, float] | None = None


def _get_forecast_data() -> dict[str, float]:
    global _forecast_cache
    if _forecast_cache is not None:
        return _forecast_cache
    data: dict[str, float] = {}
    freq_path = config.COMPETENCY_FREQ_PATH
    if freq_path.exists():
        raw = safe_read_json(freq_path)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, (int, float)):
                    data[k] = float(v)
    weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
    if weights_path.exists():
        raw = safe_read_json(weights_path)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k not in data and isinstance(v, (int, float)):
                    data[k] = float(v)
    _forecast_cache = data
    return data


def _build_forecast_engine() -> SkillForecastEngine:
    freqs = _get_forecast_data()
    engine = SkillForecastEngine()
    match engine.fit(freqs):
        case Ok(_):
            return engine
        case Err(e):
            logger.error("forecast_engine_fit_failed", error=str(e))
            return engine


def _serialize(r, direction: str | None = None) -> dict:
    change_pct = round(r.predicted_growth * 100, 2)
    return {
        "skill": r.skill,
        "current_frequency": round(r.current_frequency, 4),
        "predicted_growth": round(r.predicted_growth, 4),
        "predicted_change_pct": change_pct,
        "confidence": round(r.confidence, 4),
        "next_year_frequency": round(r.next_year_frequency, 4),
        "method": "prophet",
        "trend_direction": direction or ("growing" if r.predicted_growth > 0 else "declining"),
    }


@router.get("/api/forecast/all")
@limiter.limit("30/minute")
async def get_all_forecasts(request: Request, months: int = Query(12, ge=1, le=24)):
    engine = _build_forecast_engine()
    result = engine.forecast_all(months=months)
    match result:
        case Ok(forecasts):
            items = [_serialize(r) for r in forecasts]
            return {"total": len(items), "months": months, "forecasts": items}
        case Err(e):
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/forecast/top")
@limiter.limit("30/minute")
async def get_top_forecasts(
    request: Request,
    n: int = Query(25, ge=1, le=50),
    months: int = Query(12, ge=1, le=24),
    direction: str = Query("growing", regex="^(growing|declining)$"),
):
    engine = _build_forecast_engine()
    result = engine.top_growing(n=n * 2, months=months)
    match result:
        case Ok(all_results):
            if direction == "declining":
                all_results = sorted(all_results, key=lambda x: x.predicted_growth)[:n]
            items = [_serialize(r, direction) for r in all_results[:n]]
            return {"direction": direction, "n": n, "months": months, "forecasts": items}
        case Err(e):
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/forecast/{skill}")
@limiter.limit("60/minute")
async def get_skill_forecast(skill: str, request: Request, months: int = Query(12, ge=1, le=24)):
    engine = _build_forecast_engine()
    match engine.forecast(skill, months=months):
        case Ok(r):
            return _serialize(r)
        case Err(e):
            raise HTTPException(status_code=404, detail=f"Skill not found: {skill}")


@router.get("/api/forecast/krm/roles")
async def get_krm_roles(request: Request, sheet: str = Query("КРМ_09")):
    return {"sheet": sheet, "roles": [], "total": 0}


@router.get("/api/forecast/krm/competencies")
async def get_krm_competencies(request: Request, role: str, sheet: str = Query("КРМ_09")):
    return {"role": role, "sheet": sheet, "competencies": [], "total": 0, "categories": {}, "domain_breakdown": {}}
