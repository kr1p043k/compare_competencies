from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.db import get_pool

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


_DIR_CODE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}(?:_\w+)?$")


def _validate_dir_code(dir_code: str) -> None:
    if not _DIR_CODE_RE.match(dir_code):
        raise HTTPException(status_code=400, detail="Invalid direction code format")


def _list_krm_directions() -> list[dict]:
    """Список направлений из data/reference/krm_disciplines_*.json.

    dir_code = имя файла без префикса (например 09.03.01_ai_och).
    Файлы *_clean игнорируем, чтобы не дублировать направления.
    """
    dirs: list[dict] = []
    if not config.REFERENCE_DIR.exists():
        return dirs
    for path in sorted(config.REFERENCE_DIR.glob("krm_disciplines_*.json")):
        if "_clean" in path.name:
            continue
        dir_code = path.name[len("krm_disciplines_"):-len(".json")]
        if not dir_code or not _DIR_CODE_RE.match(dir_code):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        info = next(iter(data.values()), {}) if isinstance(data, dict) else {}
        dirs.append({
            "dir_code": dir_code,
            "name": info.get("direction_name", ""),
            "profile": info.get("profile", ""),
            "disciplines_count": len(info.get("disciplines", {}) or {}),
        })
    dirs.sort(key=lambda d: d["dir_code"])
    return dirs


def _load_krm_direction(dir_code: str) -> dict:
    """Загружает данные одного направления (direction_name/profile/disciplines)."""
    _validate_dir_code(dir_code)
    path = config.REFERENCE_DIR / f"krm_disciplines_{dir_code}.json"
    if not path.exists() or "_clean" in path.name:
        raise HTTPException(status_code=404, detail=f"Direction '{dir_code}' not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=503, detail="KRM data unavailable") from None
    if not isinstance(data, dict) or not data:
        return {}
    return next(iter(data.values()))


# ---------- endpoints ----------

@router.get("/teacher/stats")
@limiter.limit("30/minute")
async def teacher_stats(request: Request):
    """Real discipline/competency/skill counts from DB + vacancy stats."""
    result = {"disciplines": 0, "competencies": 0, "skills": 0, "vacancies": 0}

    try:
        pool = get_pool()
        r = await pool.fetchrow("""
            SELECT
                COUNT(DISTINCT disc.id) AS disciplines,
                COUNT(DISTINCT c.id) AS competencies,
                COUNT(DISTINCT k.id) AS skills
            FROM disciplines disc
            JOIN directions d ON d.id = disc.direction_id
            JOIN competencies c ON c.discipline_id = disc.id
            LEFT JOIN ksa_entries k ON k.competency_id = c.id
        """)
        if r:
            result["disciplines"] = r["disciplines"] or 0
            result["competencies"] = r["competencies"] or 0
            result["skills"] = r["skills"] or 0
    except Exception:
        pass

    try:
        pool = get_pool()
        result["vacancies"] = await pool.fetchval("SELECT COUNT(*) FROM vacancies") or 0
    except Exception:
        pass

    return result


@router.get("/teacher/krm/stats")
@limiter.limit("30/minute")
async def krm_stats(request: Request, dir_code: str = "09.03.02"):
    """Статистика KRM направления."""
    pool = get_pool()
    r = await pool.fetchrow("""
        SELECT
            COUNT(DISTINCT disc.id) AS total_disciplines,
            COUNT(DISTINCT c.id) AS total_competencies,
            COUNT(DISTINCT k.id) AS total_skills
        FROM directions d
        JOIN disciplines disc ON disc.direction_id = d.id
        JOIN competencies c ON c.discipline_id = disc.id
        LEFT JOIN ksa_entries k ON k.competency_id = c.id
        WHERE d.code = $1
    """, dir_code)
    if not r:
        raise HTTPException(status_code=404, detail=f"Direction '{dir_code}' not found")
    return {
        "dir_code": dir_code,
        "total_disciplines": r["total_disciplines"] or 0,
        "total_competencies": r["total_competencies"] or 0,
        "total_skills": r["total_skills"] or 0,
    }


@router.get("/teacher/krm/directions")
@limiter.limit("30/minute")
async def krm_directions(request: Request):
    """Список направлений KRM."""
    pool = get_pool()
    rows = await pool.fetch("""
        SELECT d.code AS dir_code, d.name, d.profile,
               COUNT(disc.id) AS disciplines_count
        FROM directions d
        LEFT JOIN disciplines disc ON disc.direction_id = d.id
        GROUP BY d.id
        ORDER BY d.code
    """)
    return [
        {
            "dir_code": r["dir_code"],
            "name": r["name"],
            "profile": r["profile"],
            "disciplines_count": r["disciplines_count"],
        }
        for r in rows
    ]


@router.get("/teacher/krm/disciplines")
@limiter.limit("30/minute")
async def krm_disciplines(request: Request, dir_code: str = "09.03.02"):
    """Дисциплины направления."""
    pool = get_pool()
    rows = await pool.fetch("""
        SELECT disc.name,
               COUNT(DISTINCT c.id) AS competencies_count,
               COUNT(DISTINCT k.id) AS skills_count
        FROM directions d
        JOIN disciplines disc ON disc.direction_id = d.id
        LEFT JOIN competencies c ON c.discipline_id = disc.id
        LEFT JOIN ksa_entries k ON k.competency_id = c.id
        WHERE d.code = $1
        GROUP BY disc.id, disc.name
        ORDER BY disc.name
    """, dir_code)
    return [
        {
            "name": r["name"],
            "competencies_count": r["competencies_count"],
            "skills_count": r["skills_count"],
        }
        for r in rows
    ]


@router.get("/teacher/krm/disciplines/{discipline_name:path}")
@limiter.limit("30/minute")
async def krm_discipline_detail(request: Request, discipline_name: str, dir_code: str = "09.03.02"):
    """Детали дисциплины (компетенции, KSA)."""
    pool = get_pool()

    disc = await pool.fetchrow("""
        SELECT disc.id, disc.name
        FROM disciplines disc
        JOIN directions d ON d.id = disc.direction_id
        WHERE d.code = $1
          AND REPLACE(disc.name, ' ', '') ILIKE REPLACE($2, ' ', '')
        LIMIT 1
    """, dir_code, discipline_name)
    if not disc:
        raise HTTPException(404, f"Discipline '{discipline_name}' not found")

    comps = await pool.fetch("""
        SELECT c.code,
               ARRAY_AGG(k.cleaned_text) FILTER (WHERE k.cleaned_text IS NOT NULL) AS skills
        FROM competencies c
        LEFT JOIN ksa_entries k ON k.competency_id = c.id
        WHERE c.discipline_id = $1 AND c.parent_id IS NULL
        GROUP BY c.id, c.code
        ORDER BY c.sort_order, c.code
    """, disc["id"])

    return {
        "name": disc["name"],
        "dir_code": dir_code,
        "competencies": [
            {
                "code": c["code"],
                "skills": c["skills"] or [],
            }
            for c in comps
        ],
    }


@router.get("/teacher/krm/recommendations")
@limiter.limit("30/minute")
async def krm_get_recommendations(request: Request):
    """Рекомендации KRM."""
    recs = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if isinstance(recs, list):
        return [{"id": i, **r} for i, r in enumerate(recs)]
    return []


@router.post("/teacher/krm/recommendations")
async def krm_add_recommendation(request: Request):
    """Добавить рекомендацию KRM."""
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
    """Удалить рекомендацию KRM."""
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
    """История запусков поиска."""
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
    """Детали запуска поиска."""
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
    """Запустить teacher analysis."""
    _validate_dir_code(dir_code)
    async def _run():
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "src.cli", "teacher-analysis",
                "--direction", dir_code,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).resolve().parent.parent.parent.parent,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
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


@router.get("/teacher/export/vacancies")
@limiter.limit("3/minute")
async def export_vacancies_excel(request: Request):
    """Export vacancies to Excel (accessible by teachers)."""
    import json
    import pandas as pd

    detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
    raw_file = detailed_file if detailed_file.exists() else basic_file
    if not raw_file.exists():
        raise HTTPException(status_code=404, detail="No vacancy data found")

    import asyncio
    vacancies = await asyncio.to_thread(lambda: json.loads(raw_file.read_bytes()))

    rows = []
    for vac in vacancies:
        name = vac.get("name", "")
        employer = vac.get("employer", {})
        employer_name = employer.get("name", "") if isinstance(employer, dict) else ""
        area = vac.get("area", {})
        area_name = area.get("name", "") if isinstance(area, dict) else ""
        salary = vac.get("salary") or {}
        salary_from = salary.get("from") if isinstance(salary, dict) else None
        salary_to = salary.get("to") if isinstance(salary, dict) else None
        skills = vac.get("extracted_skills", []) or vac.get("key_skills", [])
        if isinstance(skills, list):
            skill_names = [s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in skills if s]
            skills_str = ", ".join(skill_names[:15])
        else:
            skills_str = ""
        exp = vac.get("experience") or {}
        exp_text = exp.get("name", "") if isinstance(exp, dict) else str(exp)
        vid = str(vac.get("id", ""))
        is_spam_flag = vac.get("is_spam", False)
        spam_reason = vac.get("spam_reason", "") or ""
        is_spam = "Да" if is_spam_flag else "Нет"
        rows.append({"ID": vid, "Название": name, "Работодатель": employer_name, "Город": area_name,
                     "Зарплата от": salary_from, "Зарплата до": salary_to, "Опыт": exp_text,
                     "Спам": is_spam, "Причина спама": spam_reason,
                     "Навыки": skills_str, "Ссылка": vac.get("alternate_url", "")})

    df = pd.DataFrame(rows)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = config.REPORTS_DIR / "vacancies_export.xlsx"
    df.to_excel(excel_path, index=False, engine="openpyxl")
    if not excel_path.exists():
        raise HTTPException(status_code=500, detail="Excel export failed")
    return FileResponse(str(excel_path), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename="vacancies_export.xlsx")


@router.get("/teacher/krm/competencies/tree")
async def competency_tree(dir_code: str = "09.03.02"):
    """Competency tree with hierarchy built from parent_id and coverage data."""
    import re
    if not re.match(r"^\d{2}\.\d{2}\.\d{2}(?:_\w+)?$", dir_code):
        raise HTTPException(status_code=400, detail="Invalid direction code format")
    from src.database import async_session_factory
    from src.models.krm_models import Competency as CompModel, CoverageAnalysis, Direction, Discipline
    from sqlalchemy import select, func

    async with async_session_factory() as session:
        latest = await session.execute(select(func.max(CoverageAnalysis.analysis_date)))
        latest_date = latest.scalar()

        cov_subq = select(
            CoverageAnalysis.discipline_id,
            CoverageAnalysis.competency_id,
            func.avg(CoverageAnalysis.coverage_ratio).label("weighted_coverage"),
        ).where(CoverageAnalysis.analysis_date == latest_date).group_by(
            CoverageAnalysis.discipline_id, CoverageAnalysis.competency_id
        ).subquery()

        comps = await session.execute(
            select(CompModel, cov_subq.c.weighted_coverage)
            .outerjoin(cov_subq, cov_subq.c.competency_id == CompModel.id)
            .join(Discipline, Discipline.id == CompModel.discipline_id)
            .join(Direction, Direction.id == Discipline.direction_id)
            .where(Direction.code == dir_code)
            .order_by(CompModel.sort_order, CompModel.code)
        )
        rows = comps.all()
    node_map: dict[str, dict] = {}
    roots: list[dict] = []

    for (comp, wc) in rows:
        node = {
            "id": comp.id,
            "code": comp.code,
            "name": comp.name or "",
            "category": comp.category,
            "number": comp.number,
            "weighted_coverage": round(float(wc) if wc else 0, 4),
            "discipline_id": comp.discipline_id,
            "children": [],
        }
        node_map[comp.id] = node

    for comp_id, node in node_map.items():
        comp_next = next((c for c in rows if c[0].id == comp_id), None)
        if comp_next and comp_next[0].parent_id and comp_next[0].parent_id in node_map:
            node_map[comp_next[0].parent_id]["children"].append(node)
        else:
            roots.append(node)

    def sort_key(n: dict) -> tuple:
        m = re.match(r"(\D+)(\d+(?:\.\d+)?)", n["code"])
        return (m.group(1), float(m.group(2))) if m else (n["code"], 0)

    for node in node_map.values():
        node["children"].sort(key=sort_key)
    roots.sort(key=sort_key)

    return {"direction_code": dir_code, "competencies": roots}


@router.get("/teacher/analysis")
async def get_analysis(dir_code: str = "09.03.02"):
    """Сводка teacher analysis направления."""
    import re
    if not re.match(r"^\d{2}\.\d{2}\.\d{2}(?:_\w+)?$", dir_code):
        raise HTTPException(status_code=400, detail="Invalid direction code format")
    base = Path(__file__).resolve().parent.parent.parent.parent / "data" / "result" / "teacher"
    resolved = (base / dir_code / "_summary.json").resolve()
    if base.resolve() not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not resolved.exists():
        raise HTTPException(404, f"Analysis not found for {dir_code}")
    return json.loads(resolved.read_text(encoding="utf-8"))


@router.get("/teacher/analysis/{discipline_name:path}")
async def get_analysis_discipline(discipline_name: str, dir_code: str = "09.03.02"):
    """Teacher analysis дисциплины."""
    import re
    from pathlib import Path
    if not re.match(r"^\d{2}\.\d{2}\.\d{2}(?:_\w+)?$", dir_code):
        raise HTTPException(status_code=400, detail="Invalid direction code format")
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
