from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.api_pkg.request_logger import audit_action
from src.api_pkg.routers.auth import require_any_role
from src.api_pkg.routers.profiles import _load_self_profile, _self_profile_name
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


def _foundational_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "data" / "manual_foundational_skills.json"


def _norm_skill(skill: str) -> str:
    """Каноническое имя навыка для ручных флагов: lower + trim + схлоп пробелов."""
    return " ".join((skill or "").lower().split())


def _load_foundational() -> list[str]:
    try:
        data = _load_json(_foundational_path())
    except Exception:
        return []
    if isinstance(data, dict):
        items = data.get("skills", [])
    elif isinstance(data, list):
        items = data
    else:
        return []
    seen: list[str] = []
    for s in items:
        n = _norm_skill(str(s))
        if n and n not in seen:
            seen.append(n)
    return seen


def _save_foundational(skills: list[str]) -> None:
    _save_json(_foundational_path(), {"skills": skills})


def _apply_foundational_filter(
    recs: list[dict], flags: list[str] | set[str]
) -> tuple[list[dict], list[dict]]:
    """Убрать фундаментальное из выдачи (pure, тестируется).

    Скрывается: помеченное вручную (любой тип) + авто-foundational.
    Подписей "foundational" на странице нет — всё лежит в скрытом списке.
    Возвращает (visible, hidden); hidden-элементы — копии с флагом manual.
    Реки без skill не трогаем.
    """
    flag_set = set(flags or [])
    visible: list[dict] = []
    hidden: list[dict] = []
    for r in recs or []:
        if not isinstance(r, dict):
            visible.append(r)
            continue
        name = _norm_skill(r.get("skill", ""))
        manual = bool(name) and name in flag_set
        auto = r.get("type") == "foundational"
        if manual or auto:
            hidden.append({**r, "manual": manual})
        else:
            visible.append(r)
    return visible, hidden


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
    from src.teacher_scope import effective_in_scope, load_scope_overrides, scope_source
    pool = get_pool()
    over = await load_scope_overrides(pool, dir_code)
    rows = await pool.fetch("""
        SELECT disc.name,
               COUNT(DISTINCT c.id) AS competencies_count,
               COUNT(DISTINCT k.id) AS skills_count,
               COUNT(DISTINCT k.id) FILTER (WHERE k.ksa_type::text = 'knowledge') AS knowledge_count,
               COUNT(DISTINCT k.id) FILTER (WHERE k.ksa_type::text = 'abilities') AS abilities_count,
               MAX(disc.semester) AS semester
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
            "knowledge_count": r["knowledge_count"],
            "abilities_count": r["abilities_count"],
            "semester": r["semester"],
            "course": (r["semester"] + 1) // 2 if r["semester"] else None,
            "in_scope": effective_in_scope(r["name"], over),
            "scope_source": scope_source(r["name"], over),
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
        SELECT c.id, c.code,
               ARRAY_AGG(k.cleaned_text ORDER BY k.sort_order) FILTER (WHERE k.cleaned_text IS NOT NULL) AS skills,
               ARRAY_AGG(k.cleaned_text ORDER BY k.sort_order) FILTER (WHERE k.ksa_type::text = 'knowledge' AND k.cleaned_text IS NOT NULL) AS knowledge,
               ARRAY_AGG(k.id ORDER BY k.sort_order) FILTER (WHERE k.ksa_type::text = 'knowledge' AND k.cleaned_text IS NOT NULL) AS knowledge_ids,
               ARRAY_AGG(k.cleaned_text ORDER BY k.sort_order) FILTER (WHERE k.ksa_type::text = 'abilities' AND k.cleaned_text IS NOT NULL) AS abilities,
               ARRAY_AGG(k.id ORDER BY k.sort_order) FILTER (WHERE k.ksa_type::text = 'abilities' AND k.cleaned_text IS NOT NULL) AS abilities_ids,
               ARRAY_AGG(k.cleaned_text ORDER BY k.sort_order) FILTER (WHERE k.ksa_type::text = 'skills' AND k.cleaned_text IS NOT NULL) AS prof_skills,
               ARRAY_AGG(k.id ORDER BY k.sort_order) FILTER (WHERE k.ksa_type::text = 'skills' AND k.cleaned_text IS NOT NULL) AS prof_skills_ids
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
                "id": str(c["id"]),
                "code": c["code"],
                "skills": c["skills"] or [],
                "ksa": {
                    "knowledge": [{"id": str(i), "text": t} for i, t in zip(c["knowledge_ids"] or [], c["knowledge"] or [])],
                    "abilities": [{"id": str(i), "text": t} for i, t in zip(c["abilities_ids"] or [], c["abilities"] or [])],
                    "skills": [{"id": str(i), "text": t} for i, t in zip(c["prof_skills_ids"] or [], c["prof_skills"] or [])],
                },
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


@router.post("/teacher/krm/recommendations", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
async def krm_add_recommendation(request: Request):
    """Добавить рекомендацию KRM."""
    raw = await request.json()
    rec = RecommendationIn(**raw)
    recs = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if not isinstance(recs, list):
        recs = []
    recs.append(rec.model_dump())
    _save_json(config.TEACHER_RECOMMENDATIONS_PATH, recs)
    await audit_action(
        request, "krm.recommendation.add",
        f"discipline={rec.discipline_id} competency={rec.competency_id} "
        f"type={rec.suggestion_type} suggestion={rec.suggestion[:120]!r}",
    )
    return {"status": "ok", "id": len(recs) - 1}


@router.delete("/teacher/krm/recommendations/{index}", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
async def krm_delete_recommendation(request: Request, index: int):
    """Удалить рекомендацию KRM."""
    recs = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if not isinstance(recs, list) or index < 0 or index >= len(recs):
        raise HTTPException(404, "Recommendation not found")
    removed = recs.pop(index)
    _save_json(config.TEACHER_RECOMMENDATIONS_PATH, recs)
    await audit_action(
        request, "krm.recommendation.delete",
        f"index={index} discipline={removed.get('discipline_id')} "
        f"suggestion={str(removed.get('suggestion', ''))[:120]!r}",
    )
    return {"status": "ok"}


class FoundationalIn(BaseModel):
    skill: str


@router.get("/teacher/krm/foundational")
@limiter.limit("30/minute")
async def krm_list_foundational(request: Request):
    """Ручные пометки 'фундаментальный навык'."""
    return {"skills": _load_foundational()}


@router.post("/teacher/krm/foundational", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
async def krm_add_foundational(request: Request):
    """Пометить навык фундаментальным вручную (идемпотентно)."""
    raw = await request.json()
    name = _norm_skill((raw or {}).get("skill", "") if isinstance(raw, dict) else "")
    if not name:
        raise HTTPException(status_code=400, detail="Empty skill name")
    skills = _load_foundational()
    if name not in skills:
        skills.append(name)
        _save_foundational(skills)
        await audit_action(request, "krm.foundational.add", f"skill={name}")
    return {"status": "ok", "skills": skills}


@router.delete("/teacher/krm/foundational/{skill:path}", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
async def krm_delete_foundational(request: Request, skill: str):
    """Снять ручную пометку."""
    from urllib.parse import unquote

    name = _norm_skill(unquote(skill))
    skills = _load_foundational()
    if name not in skills:
        raise HTTPException(404, "Skill not flagged")
    skills = [s for s in skills if s != name]
    _save_foundational(skills)
    await audit_action(request, "krm.foundational.delete", f"skill={name}")
    return {"status": "ok", "skills": skills}


@router.post("/teacher/skills/suggest", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
@limiter.limit("10/minute")
async def suggest_skill(request: Request):
    """Предложить навык в таксономию (на модерацию админу)."""
    from src.api_pkg.routers.auth import get_current_user
    from src.api_pkg.skill_suggestions import add as suggest_add

    raw = await request.json()
    skill = ((raw or {}).get("skill", "") if isinstance(raw, dict) else "")
    hint = ((raw or {}).get("category_hint", "") if isinstance(raw, dict) else "")
    me = await get_current_user(request) or {}
    try:
        entry = suggest_add(skill, hint, me.get("u", ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await audit_action(
        request, "skills.suggest",
        f"skill={entry.get('skill')} hint={hint[:64]} id={entry.get('id')}",
    )
    return {"status": "ok", "suggestion": entry}


@router.get("/teacher/skills/suggestions")
@limiter.limit("30/minute")
async def list_own_suggestions(request: Request):
    """Мои предложения (по токену)."""
    from src.api_pkg.routers.auth import get_current_user
    from src.api_pkg.skill_suggestions import load_all

    me = await get_current_user(request) or {}
    mine = [s for s in load_all() if not me.get("u") or s.get("created_by") == me.get("u")]
    return {"suggestions": mine}


def _teacher_result_base() -> Path:
    """Base dir of teacher analysis read-model (extracted for test isolation)."""
    return Path(__file__).resolve().parent.parent.parent.parent / "data" / "result" / "teacher"


@router.post("/teacher/krm/recommendations/seed/auto", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
@limiter.limit("10/minute")
async def krm_seed_auto_recommendations(request: Request, dir_code: str = "09.03.02",
                                        per_discipline: int = 3):
    """Seed curated store with top auto-generated recommendations (idempotent)."""
    import re
    if not re.match(r"^\d{2}\.\d{2}\.\d{2}(?:_\w+)?$", dir_code):
        raise HTTPException(status_code=400, detail="Invalid direction code format")
    per_discipline = max(1, min(int(per_discipline), 10))
    base = _teacher_result_base()
    resolved = (base / dir_code).resolve()
    if base.resolve() not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not resolved.is_dir():
        raise HTTPException(404, f"Analysis not found for {dir_code}")
    prio = {"high": 0, "medium": 1, "low": 2}
    type_rank = {"major_revision": -1, "add_new_content": 0, "cross_reference": 1,
                 "review_content": 2, "foundational": 3}
    seeded: list = []
    for sub in sorted(resolved.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        files = list(sub.glob("*.json"))
        if not files:
            continue
        try:
            disc = json.loads(files[0].read_text(encoding="utf-8"))
        except Exception:
            continue
        recs = [r for r in (disc.get("recommendations") or [])
                if (r.get("message") or "").strip()]
        recs.sort(key=lambda r: (prio.get(r.get("priority"), 9),
                                 type_rank.get(r.get("type"), 9),
                                 -(r.get("skill") and len(r.get("skill")) or 0)))
        dname = disc.get("discipline", sub.name)
        for r in recs[:per_discipline]:
            seeded.append({"discipline_id": dname, "competency_id": None,
                           "suggestion": r["message"], "suggestion_type": "auto"})
    store = _load_json(config.TEACHER_RECOMMENDATIONS_PATH)
    if not isinstance(store, list):
        store = []
    kept = [r for r in store if r.get("suggestion_type") != "auto"]
    removed = len(store) - len(kept)
    kept.extend(seeded)
    _save_json(config.TEACHER_RECOMMENDATIONS_PATH, kept)
    await audit_action(
        request, "krm.recommendations.seed_auto",
        f"dir={dir_code} seeded={len(seeded)} removed_auto={removed} total={len(kept)}",
    )
    return {"status": "ok", "seeded": len(seeded),
            "removed_auto": removed, "total": len(kept)}


# ---------- DB-backed coverage analysis ----------


@router.get("/teacher/krm/coverage")
@limiter.limit("30/minute")
async def krm_coverage(request: Request):
    """Coverage per discipline (latest analysis)."""
    from sqlalchemy import func, select

    from src.database import async_session_factory
    from src.models.krm_models import CoverageAnalysis as CAModel
    from src.models.krm_models import Discipline

    async with async_session_factory() as session:
        # latest analysis date
        latest = await session.execute(select(func.max(CAModel.analysis_date)))
        latest_date = latest.scalar()
        if latest_date is not None and getattr(latest_date, "tzinfo", None):
            latest_date = latest_date.replace(tzinfo=None)

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
    from sqlalchemy import select

    from src.database import async_session_factory
    from src.models.krm_models import CoverageAnalysis as CAModel
    from src.models.krm_models import Discipline

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
    with open(path, encoding="utf-8") as f:
        skills = json.load(f)
    return [{"skill": s, "frequency": 1} for s in list(skills)[:limit]]

@router.get("/teacher/krm/search-runs")
async def krm_search_runs(request: Request, limit: int = 20):
    """История запусков поиска."""
    from sqlalchemy import select

    from src.database import async_session_factory
    from src.models.krm_models import PipelineRun

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
    from sqlalchemy import select

    from src.database import async_session_factory
    from src.models.krm_models import AnalysisResult, PipelineRun

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


@router.post("/teacher/krm/run-analysis", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
async def run_teacher_analysis_endpoint(
    background_tasks: BackgroundTasks,
    dir_code: str = "09.03.02",
):
    """Запустить teacher analysis (с run_id, прогрессом и ошибками)."""
    from src.api_pkg.routers.rpd import _run_cli, _update_run, short_cli_error
    from src.pipeline.db_writer import complete_pipeline_run, create_pipeline_run

    _validate_dir_code(dir_code)
    run_id = await create_pipeline_run("teacher-analysis")
    await _update_run(run_id, {"stage": "analysis", "status": "running", "dir_code": dir_code})

    async def _run():
        try:
            code, out = await _run_cli(
                [sys.executable, "-m", "src.cli", "teacher-analysis",
                 "--direction", dir_code],
                timeout=1800, run_id=run_id,
            )
            if code != 0:
                raise RuntimeError(f"teacher-analysis failed: {short_cli_error(out)}")
            await complete_pipeline_run(
                run_id, status="completed",
                stats={"stage": "done", "status": "completed", "dir_code": dir_code},
            )
        except Exception as exc:
            logger.error("teacher_analysis_cli_error", run_id=run_id,
                         exc_type=type(exc).__name__, exc_repr=repr(exc))
            try:
                await complete_pipeline_run(run_id, status="failed", error=str(exc),
                                            stats={"stage": "error", "status": "failed"})
            except Exception:
                pass

    background_tasks.add_task(_run)
    return {"status": "started", "direction": dir_code, "run_id": run_id}


@router.get("/teacher/export/vacancies", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
@limiter.limit("3/minute")
async def export_vacancies_excel(request: Request, search: str | None = None,
                                       experience: str | None = None,
                                       region: str | None = None,
                                       months: int | None = Query(None, ge=1, le=24),
                                       date_from: str | None = None,
                                       date_to: str | None = None):
    """Export vacancies to Excel with list filters (v45). No filters = full dump."""
    import json

    import pandas as pd

    from src.api_pkg.routers.vacancies import _classify_experience, _parse_day, build_vacancy_where
    pool = get_pool()
    for label, val in (("date_from", date_from), ("date_to", date_to)):
        if val and _parse_day(val) is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"{label} must be YYYY-MM-DD")
    where, params = build_vacancy_where(search=search, experience=experience,
                                        region=region, months=months,
                                        date_from=date_from, date_to=date_to)
    recs = await pool.fetch(
        """SELECT hh_id, name, employer_name, area_name, salary_from, salary_to,
                  experience, alternate_url, parsed_skills, key_skills
           FROM vacancies v WHERE %s ORDER BY v.published_at DESC NULLS LAST""" % where,
        *params)
    vacancies = [dict(r) for r in recs]

    rows = []
    for vac in vacancies:
        skills = vac.get("parsed_skills") or vac.get("key_skills") or []
        if isinstance(skills, str):
            try:
                skills = json.loads(skills)
            except Exception:
                skills = []
        skill_names = [s.get("name", str(s)) if isinstance(s, dict) else str(s)
                       for s in (skills or []) if s]
        skills_str = ", ".join(skill_names[:15])
        exp_text = _classify_experience(vac.get("experience"), vac.get("name") or "")
        rows.append({"ID": str(vac.get("hh_id") or ""), "Название": vac.get("name") or "",
                     "Работодатель": vac.get("employer_name") or "", "Город": vac.get("area_name") or "",
                     "Зарплата от": vac.get("salary_from"), "Зарплата до": vac.get("salary_to"),
                     "Опыт": exp_text, "Навыки": skills_str,
                     "Ссылка": vac.get("alternate_url") or ""})
    if not rows:
        raise HTTPException(status_code=404, detail="No vacancies match filters")
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
    from sqlalchemy import func, select

    from src.database import async_session_factory
    from src.models.krm_models import Competency as CompModel
    from src.models.krm_models import CoverageAnalysis, Direction, Discipline

    async with async_session_factory() as session:
        latest = await session.execute(select(func.max(CoverageAnalysis.analysis_date)))
        latest_date = latest.scalar()
        if latest_date is not None and getattr(latest_date, "tzinfo", None):
            latest_date = latest_date.replace(tzinfo=None)

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




def _report_meta_path(dir_code: str) -> Path:
    """Resolve _report_meta.json with the same traversal guard as get_analysis."""
    _validate_dir_code(dir_code)
    base = Path(__file__).resolve().parent.parent.parent.parent / "data" / "result" / "teacher"
    resolved = (base / dir_code / "_report_meta.json").resolve()
    if base.resolve() not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved


def report_staleness(meta: dict, current_vac_hash: str | None,
                     code_version: int) -> tuple[bool, str]:
    """Pure: is the stored read-model stale vs current code/data? (unit-tested)."""
    if not meta:
        return True, "no-report"
    if meta.get("report_schema") != 1:
        return True, "schema-mismatch"
    if meta.get("code_version") != code_version:
        return True, f"code-changed:{meta.get('code_version')}->{code_version}"
    if current_vac_hash is None:
        return False, "db-unavailable-assumed-fresh"
    if meta.get("vac_hash") != current_vac_hash:
        return True, "vacancies-changed"
    return False, "fresh"


@router.get("/teacher/analysis/meta")
async def get_analysis_meta(dir_code: str = "09.03.02"):
    """CQRS read-model lineage: which code/data produced the stored report + staleness."""
    from src.pipeline.teacher_analysis_runner import CODE_VERSION
    path = _report_meta_path(dir_code)
    if not path.exists():
        raise HTTPException(404, f"No report meta for {dir_code}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    current_hash = None
    try:
        pool = get_pool()
        current_hash = await pool.fetchval(
            "SELECT MD5(COALESCE(MAX(created_at)::text, '0')) FROM vacancies "
            "WHERE parsed_skills IS NOT NULL AND jsonb_array_length(parsed_skills) > 0"
        )
    except Exception:
        current_hash = None
    stale, reason = report_staleness(meta, current_hash, CODE_VERSION)
    return {"meta": meta, "stale": stale, "stale_reason": reason,
            "current_vac_hash": current_hash}

# ---------- Students skills view (teacher/rop/admin) ----------
# NOTE: placed here, not in admin.py — the admin router guard is
# admin+teacher only, while these endpoints must also serve rop.
# Existing /admin/students and /admin/users guards are untouched.


@router.get("/admin/students/skills", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
@limiter.limit("30/minute")
async def admin_student_skills(request: Request, email: str = ""):
    """Навыки + компетенции студента из его self-профиля.

    Self file resolved via profiles helpers (no duplication).
    404 when the user has no self file; never 500 for a missing file.
    """
    target = (email or "").strip().lower()
    if not target:
        raise HTTPException(status_code=400, detail="email is required")
    name, fpath = _self_profile_name(target)
    probe = Path(fpath)
    if not probe.exists():
        raise HTTPException(status_code=404, detail=f"No self profile for '{target}'")
    data = _load_self_profile(name, fpath)
    try:
        updated = datetime.fromtimestamp(probe.stat().st_mtime).isoformat()
    except Exception:
        updated = None
    try:
        row = await get_pool().fetchrow(
            "SELECT full_name FROM users WHERE email = $1", target)
        full_name = (row["full_name"] if row else "") or ""
    except Exception as exc:
        logger.warning("admin_student_skills_db_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="User directory unavailable")
    return {
        "email": target,
        "full_name": full_name,
        "target_level": data.get("target_level", "middle"),
        "skills": data.get("skills", []),
        "user_added": data.get("user_added", []),
        "competencies": data.get("competencies", []),
        "updated_at": updated,
    }


@router.get("/teacher/students", dependencies=[Depends(require_any_role("admin", "teacher", "rop"))])
@limiter.limit("30/minute")
async def teacher_students(request: Request):
    """Students (role=student) with self-profile stats for the Students tab."""
    try:
        rows = await get_pool().fetch(
            "SELECT email, full_name FROM users "
            "WHERE role = 'student' AND is_active = true ORDER BY email")
    except Exception as exc:
        logger.warning("teacher_students_db_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="User directory unavailable")
    items: list[dict] = []
    for r in rows:
        email = str(r["email"] or "")
        try:
            name, fpath = _self_profile_name(email)
            probe = Path(fpath)
            if probe.exists():
                data = _load_self_profile(name, fpath)
                items.append({
                    "email": email,
                    "full_name": (r["full_name"] or ""),
                    "target_level": data.get("target_level", "middle"),
                    "skills_count": len(data.get("skills", [])),
                    "competencies_count": len(data.get("competencies", [])),
                    "has_profile": True,
                })
            else:
                items.append({
                    "email": email,
                    "full_name": (r["full_name"] or ""),
                    "target_level": None,
                    "skills_count": 0,
                    "competencies_count": 0,
                    "has_profile": False,
                })
        except Exception:
            items.append({
                "email": email,
                "full_name": (r["full_name"] or ""),
                "target_level": None,
                "skills_count": 0,
                "competencies_count": 0,
                "has_profile": False,
            })
    return {"students": items, "total": len(items)}


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
    data = json.loads(files[0].read_text(encoding="utf-8"))
    # Ручные пометки foundational применяются на выдаче: помеченное исчезает
    # со страницы сразу, без рерана пайплайна.
    try:
        flags = _load_foundational()
    except Exception:
        flags = []
    if isinstance(data, dict) and isinstance(data.get("recommendations"), list):
        visible, hidden = _apply_foundational_filter(data["recommendations"], flags)
        data["recommendations"] = visible
        if config.LLM_ENABLED and config.LLM_ENHANCE_TEACHER:
            try:
                from src.services.llm_recommend import enhance_teacher_recs
                enhanced = enhance_teacher_recs(
                    discipline=discipline_name,
                    gaps=data.get("gaps", []),
                    base_recs=data,
                )
                if isinstance(enhanced, dict):
                    data = enhanced
            except Exception:
                logger.warning("llm_teacher_enhance_failed", discipline=discipline_name)
        data["hidden_foundational"] = [
            {"skill": (r.get("skill", "") if isinstance(r, dict) else ""),
             "type": r.get("type", ""), "manual": bool(r.get("manual", False))}
            for r in hidden if isinstance(r, dict)
        ]
    return data
