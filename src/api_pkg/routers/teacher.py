from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["teacher"])
limiter = Limiter(key_func=get_remote_address)


# ---------- models ----------

class RecommendationIn(BaseModel):
    discipline_id: str
    competency_id: str | None = None
    suggestion: str
    suggestion_type: str = "modify"


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

@router.get("/teacher/stats")
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


@router.get("/teacher/krm/stats")
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


@router.get("/teacher/krm/directions")
@limiter.limit("30/minute")
async def krm_directions(request: Request):
    data = _load_json(config.KRM_DISCIPLINES_PATH)
    info = data.get("09.03.02", {})
    return {
        "code": "09.03.02",
        "name": info.get("direction_name", ""),
        "profile": info.get("profile", ""),
    }


@router.get("/teacher/krm/disciplines")
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


@router.get("/teacher/krm/disciplines/{discipline_name:path}")
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


@router.get("/teacher/krm/recommendations")
@limiter.limit("30/minute")
async def krm_get_recommendations(request: Request):
    recs = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if isinstance(recs, list):
        return [{"id": i, **r} for i, r in enumerate(recs)]
    return []


@router.post("/teacher/krm/recommendations")
async def krm_add_recommendation(request: Request):
    raw = await request.json()
    rec = RecommendationIn(**raw)
    recs = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if not isinstance(recs, list):
        recs = []
    recs.append(rec.model_dump())
    _save_json(config.TEACHER_RECOMMENDATIONS_PATH, recs)
    return {"status": "ok", "id": len(recs) - 1}


@router.delete("/teacher/krm/recommendations/{index}")
async def krm_delete_recommendation(request: Request, index: int):
    recs = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if not isinstance(recs, list) or index < 0 or index >= len(recs):
        raise HTTPException(404, "Recommendation not found")
    recs.pop(index)
    _save_json(config.TEACHER_RECOMMENDATIONS_PATH, recs)
    return {"status": "ok"}


# ---------- DB-backed coverage analysis ----------


@router.get("/teacher/krm/coverage")
@limiter.limit("30/minute")
async def krm_coverage(request: Request):
    """Coverage per discipline (latest analysis)."""
    from src.database import async_session_factory
    from src.models.krm_models import CoverageAnalysis as CAModel, Discipline
    from sqlalchemy import select, func

    async with async_session_factory() as session:
        # latest analysis date
        latest = await session.execute(select(func.max(CAModel.analysis_date)))
        latest_date = latest.scalar()

        result = await session.execute(
            select(CAModel, Discipline.name)
            .join(Discipline, CAModel.discipline_id == Discipline.id)
            .where(CAModel.analysis_date == latest_date)
            .order_by(CAModel.coverage_ratio.asc())
        )
        rows = result.all()

    return {
        "analysis_date": latest_date.isoformat() if latest_date else None,
        "disciplines": [
            {
                "name": name,
                "total_skills": ca.total_skills,
                "matched_skills": ca.market_matched_skills,
                "coverage_ratio": ca.coverage_ratio,
            }
            for ca, name in rows
        ],
    }


@router.get("/teacher/krm/coverage/history")
@limiter.limit("30/minute")
async def krm_coverage_history(request: Request, discipline: str | None = None, limit: int = 20):
    """Coverage history across analyses."""
    from src.database import async_session_factory
    from src.models.krm_models import CoverageAnalysis as CAModel, Discipline
    from sqlalchemy import select

    async with async_session_factory() as session:
        query = select(CAModel, Discipline.name).join(Discipline, CAModel.discipline_id == Discipline.id)
        if discipline:
            query = query.where(Discipline.name.ilike(f"%{discipline}%"))
        query = query.order_by(CAModel.analysis_date.desc()).limit(limit)
        result = await session.execute(query)

    return [
        {
            "discipline": name,
            "coverage_ratio": ca.coverage_ratio,
            "total_skills": ca.total_skills,
            "matched_skills": ca.market_matched_skills,
            "analysis_date": ca.analysis_date.isoformat() if ca.analysis_date else None,
        }
        for ca, name in result.all()
    ]


@router.get("/teacher/krm/market-skills")
@limiter.limit("30/minute")
async def krm_market_skills(request: Request, limit: int = 50):
    """Top market-demanded skills (from it_skills.json)."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "it_skills.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        skills = json.load(f)
    return [{"skill": s, "frequency": 1} for s in list(skills)[:limit]]

@router.get("/teacher/krm/search-runs")
async def krm_search_runs(request: Request, limit: int = 20):
    from src.database import async_session_factory
    from src.models.krm_models import PipelineRun
    from sqlalchemy import select

    async with async_session_factory() as session:
        result = await session.execute(
            select(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

    return [
        {
            "id": str(r.id),
            "action": r.action,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "stats": r.stats or {},
        }
        for r in rows
    ]


@router.get("/teacher/krm/search-runs/{run_id}")
async def krm_search_run_detail(run_id: str):
    from src.database import async_session_factory
    from src.models.krm_models import PipelineRun, AnalysisResult
    from sqlalchemy import select

    async with async_session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        if not run:
            raise HTTPException(404, "Run not found")

        result = await session.execute(
            select(AnalysisResult)
            .where(AnalysisResult.pipeline_run_id == run_id)
            .order_by(AnalysisResult.created_at.desc())
        )
        analysis = result.scalars().first()

    return {
        "run": {
            "id": str(run.id),
            "action": run.action,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "stats": run.stats or {},
        },
        "analysis": analysis.data if analysis else None,
    }


@router.post("/teacher/krm/run-analysis")
async def run_teacher_analysis_endpoint(
    background_tasks: BackgroundTasks,
    dir_code: str = "09.03.02",
):
    async def _run():
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "src.cli", "teacher-analysis",
                "--direction", dir_code,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).resolve().parent.parent.parent.parent,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
            logger.info("teacher_analysis_cli_done", returncode=proc.returncode,
                         stderr=stderr.decode("utf-8", errors="ignore")[-500:])
        except asyncio.TimeoutError:
            logger.error("teacher_analysis_cli_timeout")
            if proc and proc.returncode is None:
                proc.kill()
        except Exception as exc:
            logger.error("teacher_analysis_cli_error", error=str(exc))

    background_tasks.add_task(_run)
    return {"status": "started", "direction": dir_code}


@router.get("/teacher/analysis")
async def get_analysis(dir_code: str = "09.03.02"):
    summary_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "result" / "teacher" / dir_code / "_summary.json"
    if not summary_path.exists():
        raise HTTPException(404, f"Analysis not found for {dir_code}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


@router.get("/teacher/analysis/{discipline_name:path}")
async def get_analysis_discipline(discipline_name: str, dir_code: str = "09.03.02"):
    import re
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent.parent.parent / "data" / "result" / "teacher" / dir_code
    safe = re.sub(r'[\\/*?:"<>|]', "_", discipline_name).strip()[:80]
    files = [f for f in base.rglob(f"{safe}.json") if f.name != "_summary.json"]
    if not files:
        for f in base.rglob("*.json"):
            if f.name == "_summary.json": continue
            if discipline_name.lower() in f.stem.lower(): files.append(f)
    if not files:
        raise HTTPException(404, f"'{discipline_name}' not found")
    return json.loads(files[0].read_text(encoding="utf-8"))
