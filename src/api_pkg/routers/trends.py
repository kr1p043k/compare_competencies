"""Trends analysis — per-competency aggregate + per-skill trends from trend_snapshots."""
from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select

from src import Err, Ok
from src.analyzers.skills.trends import TrendAnalyzer
from src.database import async_session_factory
from src.models.api_responses import TrendsResponse
from src.models.krm_models import Competency, CompetencySkill, Skill, TrendSnapshot

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


def _classify(change_pct: float) -> str:
    if change_pct > 5:
        return "rising"
    if change_pct < -5:
        return "falling"
    return "stable"


@router.get("/api/competency-trends")
@limiter.limit("60/minute")
async def get_competency_trends(
    request: Request,
    direction: str | None = Query(None, pattern="^(rising|falling|stable)$"),
    limit: int = Query(200, ge=1, le=500),
):
    """Compute per-competency trends directly from trend_snapshots."""
    async with async_session_factory() as session:
        # 1. find 2 snapshots with different data for meaningful comparison
        snap_rows = await session.execute(
            select(TrendSnapshot).order_by(TrendSnapshot.snapshot_date.desc())
        )
        all_snaps = snap_rows.scalars().all()
        if len(all_snaps) < 2:
            raise HTTPException(400, "Need at least 2 trend snapshots")

        def parse_freq(raw: Any) -> dict[str, int]:
            if isinstance(raw, str):
                return json.loads(raw)
            return {k: int(v) for k, v in (raw or {}).items()}

        cur = all_snaps[0]
        cur_freq = parse_freq(cur.skill_freq)
        snap_latest = str(cur.snapshot_date)

        # find a previous snapshot with different data (or at least 7 days older)
        prev = None
        for s in all_snaps[1:]:
            prev_freq = parse_freq(s.skill_freq)
            if prev_freq != cur_freq:
                prev = s
                break
            days_diff = (cur.snapshot_date - s.snapshot_date).days
            if days_diff >= 7:
                prev = s
                break
        if prev is None:
            prev = all_snaps[-1]
        prev_freq = parse_freq(prev.skill_freq)
        snap_prev = str(prev.snapshot_date)

        # 2. all competencies + their skills
        comp_rows = await session.execute(
            select(
                Competency.id,
                Competency.code,
                Competency.name,
                Skill.name.label("skill_name"),
            )
            .outerjoin(CompetencySkill, CompetencySkill.competency_id == Competency.id)
            .outerjoin(Skill, Skill.id == CompetencySkill.skill_id)
            .order_by(Competency.code, Skill.name)
        )
        comp_map: dict[str, dict] = {}
        for row in comp_rows:
            code = row.code
            if code not in comp_map:
                comp_map[code] = {
                    "id": str(row.id),
                    "name": row.name or "",
                    "skills": [],
                }
            if row.skill_name:
                comp_map[code]["skills"].append(row.skill_name)

        # 3. compute trends per competency
        results = []
        for code, data in comp_map.items():
            skill_list = []
            changes: list[float] = []
            rising = 0
            falling = 0
            for sk in data["skills"]:
                cv = cur_freq.get(sk, 0) or 0
                pv = prev_freq.get(sk, 0) or 0
                if pv > 0:
                    chg = round((cv - pv) / pv * 100, 1)
                    changes.append(chg)
                    if chg >= 5:
                        rising += 1
                    elif chg <= -5:
                        falling += 1
                else:
                    chg = 0.0
                skill_list.append({
                    "name": sk,
                    "direction": _classify(chg),
                    "change_pct": chg,
                })

            if not changes:
                continue

            agg = round(sum(changes) / len(changes), 1)
            total_ch = len(changes)
            stable = total_ch - rising - falling
            if rising > falling and rising > stable:
                comp_dir = "rising"
            elif falling > rising and falling > stable:
                comp_dir = "falling"
            else:
                comp_dir = "stable"

            results.append({
                "competency_id": data["id"],
                "code": code,
                "name": data["name"],
                "direction": comp_dir,
                "change_pct": agg,
                "skill_count": len(data["skills"]),
                "active_skills_count": len(changes),
                "snapshot_date": snap_latest,
                "skills": skill_list,
            })

        # 4. sort: rising first, then by |change_pct| desc
        results.sort(
            key=lambda r: (
                0 if r["direction"] == "rising" else 1 if r["direction"] == "falling" else 2,
                -abs(r["change_pct"]),
            )
        )

        # 5. filter
        if direction:
            results = [r for r in results if r["direction"] == direction]

        return {"total": len(results), "trends": results[:limit]}
