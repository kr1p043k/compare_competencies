from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["teacher"])
limiter = Limiter(key_func=get_remote_address)


# ---------- models ----------

class RecommendationIn(BaseModel):
    discipline: str
    competency: str
    suggestion: str
    type: str = "modify"


class RecommendationOut(RecommendationIn):
    id: int


# ---------- helpers ----------

def _load_json(path) -> dict | list:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# ---------- endpoints ----------

@router.get("/api/teacher/stats")
@limiter.limit("30/minute")
async def teacher_stats(request: Request):
    result_dir = config.DATA_RESULT_DIR
    if not result_dir.exists():
        return {"total_reports": 0, "by_profession": []}

    reports = []
    for f in result_dir.glob("*recommendations_*.json"):
        parts = f.stem.replace("recommendations_", "").split("_")
        prof = parts[0] if parts else "unknown"
        reports.append(prof)

    prof_counts: dict[str, int] = {}
    for p in reports:
        prof_counts[p] = prof_counts.get(p, 0) + 1

    by_profession = [{"profession": k, "reports": v} for k, v in sorted(prof_counts.items(), key=lambda x: -x[1])]
    return {"total_reports": len(reports), "by_profession": by_profession}


@router.get("/api/teacher/krm/stats")
@limiter.limit("30/minute")
async def krm_stats(request: Request):
    data = _load_json(config.KRM_DISCIPLINES_PATH)
    d = data.get("09.03.02", {}).get("disciplines", {})

    total_comps: set[str] = set()
    total_skills = 0
    for disc in d.values():
        total_comps.update(disc.get("competencies", []))
        for skills in disc.get("skills", {}).values():
            total_skills += len(skills)

    return {
        "total_disciplines": len(d),
        "total_competencies": len(total_comps),
        "total_skills": total_skills,
    }


@router.get("/api/teacher/krm/directions")
@limiter.limit("30/minute")
async def krm_directions(request: Request):
    data = _load_json(config.KRM_DISCIPLINES_PATH)
    info = data.get("09.03.02", {})
    return {
        "code": "09.03.02",
        "name": info.get("direction_name", ""),
        "profile": info.get("profile", ""),
    }


@router.get("/api/teacher/krm/disciplines")
@limiter.limit("30/minute")
async def krm_disciplines(request: Request):
    data = _load_json(config.KRM_DISCIPLINES_PATH)
    d = data.get("09.03.02", {}).get("disciplines", {})
    return [
        {
            "name": name,
            "competencies_count": len(info.get("competencies", [])),
            "skills_count": sum(len(s) for s in info.get("skills", {}).values()),
        }
        for name, info in sorted(d.items())
    ]


@router.get("/api/teacher/krm/disciplines/{discipline_name:path}")
@limiter.limit("30/minute")
async def krm_discipline_detail(request: Request, discipline_name: str):
    data = _load_json(config.KRM_DISCIPLINES_PATH)
    d = data.get("09.03.02", {}).get("disciplines", {})

    # fuzzy match because encoding may vary
    info = d.get(discipline_name)
    if not info:
        for k, v in d.items():
            if k.replace(" ", "").lower() == discipline_name.replace(" ", "").lower():
                info = v
                break

    if not info:
        raise HTTPException(404, f"Discipline '{discipline_name}' not found")

    return {
        "name": discipline_name,
        "competencies": [
            {
                "code": comp,
                "skills": info.get("skills", {}).get(comp, []),
            }
            for comp in info.get("competencies", [])
        ],
    }


@router.get("/api/teacher/krm/recommendations")
@limiter.limit("30/minute")
async def krm_get_recommendations(request: Request):
    recs = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if isinstance(recs, list):
        return [{"id": i, **r} for i, r in enumerate(recs)]
    return []


@router.post("/api/teacher/krm/recommendations")
@limiter.limit("30/minute")
async def krm_add_recommendation(request: Request, rec: RecommendationIn):
    recs = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if not isinstance(recs, list):
        recs = []
    recs.append(rec.model_dump())
    _save_json(config.TEACHER_RECOMMENDATIONS_PATH, recs)
    return {"status": "ok", "id": len(recs) - 1}


@router.delete("/api/teacher/krm/recommendations/{index}")
@limiter.limit("30/minute")
async def krm_delete_recommendation(request: Request, index: int):
    recs = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if not isinstance(recs, list) or index < 0 or index >= len(recs):
        raise HTTPException(404, "Recommendation not found")
    recs.pop(index)
    _save_json(config.TEACHER_RECOMMENDATIONS_PATH, recs)
    return {"status": "ok"}
