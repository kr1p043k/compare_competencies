import re
from datetime import date
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text

from src import Ok, Err, Result, config, DomainError
from src.predictors.prophet_forecast import ProphetForecastEngine
from src.predictors.skill_forecast import SkillForecastEngine, ForecastResult
from src.utils import safe_read_json
import src.api_pkg.deps as deps

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["forecast"])
limiter = Limiter(key_func=get_remote_address)


def _get_forecast_engine() -> Result[ProphetForecastEngine | SkillForecastEngine, DomainError]:
    if deps.prophet_engine is not None and deps.prophet_engine.is_fitted:
        return Ok(deps.prophet_engine)
    freqs = _get_forecast_data()
    if not freqs:
        return Err(DomainError("No frequency data available for forecast"))
    engine = SkillForecastEngine()
    return engine.fit(freqs)


def _get_forecast_data() -> dict[str, float]:
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
    return data


async def _get_vacancy_meta() -> dict:
    meta = {"vacancies_count": 0, "data_from": None, "data_to": date.today().isoformat()}
    try:
        from src.database import async_session_factory

        async with async_session_factory() as session:
            row = (
                await session.execute(
                    text("""
                        SELECT
                            COUNT(*) AS cnt,
                            MIN(published_at::date) AS min_date
                        FROM vacancies
                        WHERE parsed_skills IS NOT NULL
                          AND parsed_skills != '[]'::jsonb
                    """)
                )
            ).one()
            if row.cnt:
                meta["vacancies_count"] = row.cnt
                meta["data_from"] = row.min_date.isoformat() if row.min_date else None
                return meta
    except Exception:
        logger.warning("vacancy_meta_db_failed_falling_back_to_files")
    freq_path = config.COMPETENCY_FREQ_PATH
    if freq_path.exists():
        raw = safe_read_json(freq_path)
        if isinstance(raw, dict):
            meta["vacancies_count"] = len(raw)
    history_dir: Path = config.DATA_DIR / "history"
    if history_dir.is_dir():
        snaps = sorted(history_dir.glob("freq_*.json"))
        if snaps:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", snaps[-1].stem)
            if m:
                meta["data_from"] = m.group(1)
    return meta


def _serialize(r: ForecastResult, direction: str | None = None, method: str = "genetic") -> dict:
    change_pct = round(r.predicted_growth * 100, 2)
    return {
        "skill": r.skill,
        "current_frequency": round(r.current_frequency, 4),
        "predicted_growth": round(r.predicted_growth, 4),
        "predicted_change_pct": change_pct,
        "confidence": round(r.confidence, 4),
        "next_year_frequency": round(r.next_year_frequency, 4),
        "method": method,
        "trend_direction": direction or ("growing" if r.predicted_growth > 0 else "declining"),
    }


def _detect_method(engine: ProphetForecastEngine | SkillForecastEngine, skill: str | None = None) -> str:
    if isinstance(engine, ProphetForecastEngine):
        if skill is not None:
            return "prophet" if skill in engine._models else "genetic"
        return "prophet" if engine._models else "genetic"
    return "genetic"


@router.get("/forecast/all")
@limiter.limit("30/minute")
async def get_all_forecasts(request: Request, months: int = Query(12, ge=1, le=24)):
    match _get_forecast_engine():
        case Ok(engine):
            match engine.forecast_all(months=months):
                case Ok(forecasts):
                    method = _detect_method(engine)
                    items = [_serialize(r, method=method) for r in forecasts]
                    return {"total": len(items), "months": months, "forecasts": items}
                case Err(e):
                    raise HTTPException(status_code=500, detail=str(e))
        case Err(e):
            raise HTTPException(status_code=503, detail=str(e))


@router.get("/forecast/top")
@limiter.limit("30/minute")
async def get_top_forecasts(
    request: Request,
    n: int = Query(25, ge=1, le=50),
    months: int = Query(12, ge=1, le=24),
    direction: str = Query("growing", regex="^(growing|declining)$"),
):
    match _get_forecast_engine():
        case Ok(engine):
            match engine.top_growing(n=n * 2, months=months):
                case Ok(all_results):
                    if direction == "declining":
                        all_results = sorted(all_results, key=lambda x: x.predicted_growth)[:n]
                    method = _detect_method(engine)
                    items = [_serialize(r, direction, method) for r in all_results[:n]]
                    meta = await _get_vacancy_meta()
                    return {"direction": direction, "n": n, "months": months, "forecasts": items, **meta}
                case Err(e):
                    raise HTTPException(status_code=500, detail=str(e))
        case Err(e):
            raise HTTPException(status_code=503, detail=str(e))


@router.get("/forecast/{skill}")
@limiter.limit("60/minute")
async def get_skill_forecast(skill: str, request: Request, months: int = Query(12, ge=1, le=24)):
    match _get_forecast_engine():
        case Ok(engine):
            result = engine.forecast(skill, months) if hasattr(engine, "forecast") else engine.predict(skill, months)
            match result:
                case Ok(r):
                    return _serialize(r, method=_detect_method(engine, skill))
                case Err(e):
                    raise HTTPException(status_code=404, detail=f"Skill not found: {skill}")
        case Err(e):
            raise HTTPException(status_code=503, detail=str(e))
