"""Trends by profession - live SQL query + historical snapshots."""
from __future__ import annotations

import json
import re

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src import config
from src.api_pkg.rate_limit import limiter
from src.api_pkg.routers.auth import require_authenticated
from src.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["trends"], dependencies=[Depends(require_authenticated())])


@router.get("/trends/professions")
@limiter.limit("60/minute")
async def list_professions(request: Request):
    """List available professions that have snapshots."""
    history = config.HISTORY_DIR
    files = list(history.glob("freq_profession_*.json"))
    seen: set[str] = set()
    result: list[dict] = []
    for f in files:
        # Мусорные файлы без профессии (freq_profession_2026-09.json) пропускаем.
        if not re.match(r"^freq_profession_.+_\d{4}-\d{2}\.json$", f.name):
            continue
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            meta = raw.get("_meta", {})
            prof = meta.get("profession") or f.name.replace("freq_profession_", "").rsplit("_", 1)[0]
            if prof not in seen:
                seen.add(prof)
                result.append({"name": prof, "snapshot": f.name, "date": meta.get("snapshot_date", "")})
        except Exception:
            pass
    return {"professions": sorted(result, key=lambda x: x["name"])}


@router.get("/trends/by-profession")
@limiter.limit("60/minute")
async def get_profession_trends(
    request: Request,
    profession: str = Query(..., description="Profession name from taxonomy"),
    limit: int = Query(50, ge=1, le=500),
):
    """Get top skills for a profession from the latest snapshot."""
    history = config.HISTORY_DIR
    # Find latest snapshot for this profession
    pattern = f"freq_profession_{profession}_*.json"
    files = sorted(history.glob(pattern))
    if not files:
        # Try falling back to live SQL
        return await _live_profession_query(profession, limit)

    latest = files[-1]
    raw = json.loads(latest.read_text(encoding="utf-8"))
    meta = raw.get("_meta", {})
    skills = sorted(
        [(k, v) for k, v in raw.items() if k != "_meta"],
        key=lambda x: -x[1],
    )[:limit]
    return {
        "profession": profession,
        "source": "snapshot",
        "snapshot_date": meta.get("snapshot_date", ""),
        "vacancy_count": meta.get("vacancy_count"),
        "synthetic": bool(meta.get("synthetic", False)),
        "methodology_version": meta.get("methodology_version", ""),
        "skills": [{"skill": s, "frequency": f} for s, f in skills],
    }


async def _live_profession_query(profession: str, limit: int = 50) -> dict:
    """Live SQL query when no snapshot exists."""
    try:
        pool = get_pool()
        rows = await pool.fetch(
            """SELECT LOWER(TRIM(value)) AS skill, COUNT(DISTINCT v.id) AS cnt
               FROM vacancies v,
                    jsonb_array_elements_text(
                        COALESCE(v.parsed_skills, v.key_skills, '[]'::jsonb)
                    ) AS value
               WHERE v.parsed_skills IS NOT NULL
                 AND LOWER(v.name) LIKE '%' || $1 || '%'
               GROUP BY skill
               ORDER BY cnt DESC
               LIMIT $2""",
            profession.lower(), limit,
        )
        skills = [{"skill": r["skill"], "frequency": r["cnt"]} for r in rows if r["cnt"] > 0]
        return {"profession": profession, "source": "live", "skills": skills}
    except Exception as e:
        logger.error("live_profession_query_failed", profession=profession, error=str(e))
        raise HTTPException(status_code=503, detail="Trend data unavailable") from None
