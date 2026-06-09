"""Admin: whitelist management, student profiles, pipeline, Excel export."""

import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.parsing.utils import load_it_skills
from src.api_pkg.request_logger import get_logs, get_logs_by_user
from src.api_pkg.routers.auth import require_role

from src.api_pkg import deps
from src.monitoring.metrics import (
    pipeline_duration, pipeline_errors, api_latency, api_requests_total,
    recommendations_generated, ltr_training_duration, ltr_model_metrics,
    vacancies_loaded, active_profiles, skill_count,
)

logger = structlog.get_logger("api")

router = APIRouter(tags=["admin"], dependencies=[Depends(require_role("admin"))])
limiter = Limiter(key_func=get_remote_address)


class WhitelistAddRequest(BaseModel):
    skills: list[str]


class WhitelistRemoveRequest(BaseModel):
    skills: list[str]


class WhitelistBackupResponse(BaseModel):
    status: str
    message: str
    backup_path: str | None = None
    total_skills: int = 0


IT_SKILLS_PATH = config.DATA_DIR / "reference" / "it_skills.json"


def _load_whitelist() -> list[str]:
    return sorted(load_it_skills())


def _save_whitelist(skills: list[str]):
    IT_SKILLS_PATH.write_text(
        json.dumps(sorted(set(skills)), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    deps.current_skills_set = set(skills)


# ---------- Whitelist ----------


@router.get("/api/admin/whitelist")
@limiter.limit("30/minute")
async def get_whitelist(request: Request):
    skills = _load_whitelist()
    return {"skills": skills, "total": len(skills)}


@router.post("/api/admin/whitelist/add")
@limiter.limit("10/minute")
async def whitelist_add(request: Request, body: WhitelistAddRequest):
    current = set(_load_whitelist())
    before = len(current)
    current.update(body.skills)
    if len(current) == before:
        return {
            "status": "ok",
            "message": "No new skills to add",
            "added": 0,
            "total": len(current),
        }
    _save_whitelist(list(current))
    logger.info("Whitelist extended", added=len(current) - before, total=len(current))
    return {
        "status": "ok",
        "message": f"Added {len(current) - before} skills",
        "added": len(current) - before,
        "total": len(current),
    }


@router.post("/api/admin/whitelist/remove")
@limiter.limit("10/minute")
async def whitelist_remove(request: Request, body: WhitelistRemoveRequest):
    current = set(_load_whitelist())
    before = len(current)
    current -= set(body.skills)
    if len(current) == before:
        return {
            "status": "ok",
            "message": "No skills removed",
            "removed": 0,
            "total": len(current),
        }
    _save_whitelist(list(current))
    logger.info("Whitelist trimmed", removed=before - len(current), total=len(current))
    return {
        "status": "ok",
        "message": f"Removed {before - len(current)} skills",
        "removed": before - len(current),
        "total": len(current),
    }


@router.post("/api/admin/whitelist/backup", response_model=WhitelistBackupResponse)
@limiter.limit("5/minute")
async def whitelist_backup(request: Request):
    backup_dir = config.DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"it_skills_backup_{ts}.json"
    shutil.copy2(IT_SKILLS_PATH, backup_path)
    skills = _load_whitelist()
    return WhitelistBackupResponse(
        status="ok",
        message=f"Backup saved to {backup_path.name}",
        backup_path=str(backup_path),
        total_skills=len(skills),
    )


# ---------- Student profiles ----------


@router.get("/api/admin/students")
@limiter.limit("30/minute")
async def list_students(request: Request):
    students_dir = config.DATA_DIR / "students"
    if not students_dir.exists():
        return {"students": [], "total": 0}
    files = sorted(students_dir.iterdir())
    students = []
    for f in files:
        if f.suffix in (".json",):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                skills = (
                    data.get("competencias_actuales")
                    or data.get("навыки")
                    or data.get("skills")
                    or []
                )
                students.append(
                    {
                        "filename": f.name,
                        "skills_count": len(skills) if isinstance(skills, list) else 0,
                        "modified": datetime.fromtimestamp(
                            f.stat().st_mtime
                        ).isoformat(),
                    }
                )
            except Exception:
                students.append({"filename": f.name, "skills_count": 0, "modified": ""})
    return {"students": students, "total": len(students)}


# ---------- Pipeline trigger ----------


class PipelineTriggerRequest(BaseModel):
    action: str = "full-cycle"
    regions: str = "0"
    skip_collection: bool = False
    run_gap_analysis: bool = True


@router.post("/api/admin/pipeline/trigger")
@limiter.limit("2/minute")
async def admin_trigger_pipeline(
    request: Request, body: PipelineTriggerRequest, background_tasks: BackgroundTasks
):
    from src.api_pkg.routers.pipeline import PipelineAction, run_pipeline_task

    action_map = {
        "full-cycle": PipelineAction.FULL_CYCLE,
        "rebuild": PipelineAction.REBUILD,
        "train-clusters": PipelineAction.TRAIN_CLUSTERS,
        "train-model": PipelineAction.TRAIN_MODEL,
        "gap-analysis": PipelineAction.GAP_ANALYSIS,
    }
    action = action_map.get(body.action)
    if not action:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {body.action}. Valid: {', '.join(action_map.keys())}",
        )
    task_id = f"{action.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    background_tasks.add_task(
        run_pipeline_task,
        action,
        task_id,
        skip_collection=body.skip_collection,
        run_gap_analysis=body.run_gap_analysis,
        regions=body.regions,
    )
    return {
        "status": "started",
        "task_id": task_id,
        "check_url": f"/api/pipeline/task/{task_id}",
    }


# ---------- Excel export ----------


def _format_experience(exp: Any) -> str:
    if isinstance(exp, dict):
        eid = exp.get("id", "")
        ename = exp.get("name", "")
        if eid == "noExperience":
            return "не требуется"
        return f"требуется: {ename}"
    if isinstance(exp, str):
        return exp
    return ""


@router.get("/api/admin/export/excel")
@limiter.limit("3/minute")
async def export_excel(request: Request):
    import json
    import pandas as pd

    detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
    raw_file = detailed_file if detailed_file.exists() else basic_file
    if not raw_file.exists():
        raise HTTPException(status_code=404, detail="No vacancy data found")

    with open(raw_file, encoding="utf-8") as f:
        vacancies = json.load(f)

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

        exp_text = _format_experience(vac.get("experience"))
        vid = str(vac.get("id", ""))
        is_spam_flag = vac.get("is_spam", False)
        spam_reason = vac.get("spam_reason", "") or ""
        is_spam = "Да" if is_spam_flag else "Нет"

        rows.append({
            "ID": vid,
            "Название": name,
            "Работодатель": employer_name,
            "Город": area_name,
            "Зарплата от": salary_from,
            "Зарплата до": salary_to,
            "Опыт": exp_text,
            "Спам": is_spam,
            "Причина спама": spam_reason,
            "Навыки": skills_str,
            "Ссылка": vac.get("alternate_url", ""),
        })

    df = pd.DataFrame(rows)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = config.REPORTS_DIR / "vacancies_export.xlsx"
    df.to_excel(excel_path, index=False, engine="openpyxl")

    if not excel_path.exists():
        raise HTTPException(status_code=500, detail="Excel export failed")

    return FileResponse(
        str(excel_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="vacancies_export.xlsx",
    )


@router.get("/api/admin/export/full-report")
@limiter.limit("2/minute")
async def export_full_report(request: Request):
    from fastapi.responses import StreamingResponse

    result_dir = config.DATA_RESULT_DIR
    if not result_dir.exists() or not any(result_dir.rglob("*")):
        raise HTTPException(status_code=404, detail="No report data found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(result_dir.rglob("*")):
            if fpath.is_file() and fpath.suffix in {".json", ".png", ".xlsx"}:
                arcname = str(fpath.relative_to(result_dir.parent))
                zf.write(str(fpath), arcname)
    zip_buffer.seek(0)

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="full_report.zip"'},
    )


@router.get("/api/admin/users")
@limiter.limit("30/minute")
async def admin_users(request: Request):
    from sqlalchemy import text
    from src.database import async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(text("SELECT * FROM users ORDER BY created_at"))
        users = result.fetchall()

    log_counts = get_logs_by_user()
    return {
        "users": [
            {
                "username": u.email,
                "role": u.role,
                "name": u.full_name,
                "total_requests": log_counts.get(u.email, 0),
                "is_active": u.is_active,
            }
            for u in users
        ]
    }


@router.get("/api/admin/monitoring")
async def admin_monitoring(request: Request):
    from prometheus_client.parser import text_string_to_metric_families
    from src.monitoring.metrics import get_metrics
    raw, _ = get_metrics()
    families = {}
    for f in text_string_to_metric_families(raw.decode()):
        samples = []
        for s in f.samples:
            samples.append({"name": s.name, "labels": s.labels, "value": s.value})
        families[f.name] = {"name": f.name, "type": f.type, "samples": samples}
    return families


@router.get("/api/admin/logs")
@limiter.limit("30/minute")
async def admin_logs(request: Request, user: str | None = None, limit: int = 100):
    if user and user == "all":
        user = None
    entries = get_logs(user=user, limit=limit)
    return {"logs": entries, "total": len(entries)}


# ---------- DB Management (CLI → API) ----------


class SeedDBRequest(BaseModel):
    drop: bool = False


@router.post("/api/admin/db/seed")
@limiter.limit("1/minute")
async def admin_seed_db(request: Request, body: SeedDBRequest, background_tasks: BackgroundTasks):
    """Seed database from JSON files (skills, disciplines, competencies)."""
    background_tasks.add_task(_run_seed, drop=body.drop)
    return {"status": "started", "message": "Seeding database in background"}


def _run_seed(drop: bool = False) -> None:
    import asyncio
    from src.cli.seed_db import main as seed_main
    asyncio.run(seed_main(drop=drop))
    logger.info("db_seed_completed", drop=drop)


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str = "teacher"
    name: str = ""


@router.post("/api/admin/users/create")
@limiter.limit("10/minute")
async def admin_create_user(request: Request, body: CreateUserRequest):
    """Create a new user with bcrypt-hashed password."""
    from src.cli.create_user import main as create_user_main
    await create_user_main(body.email, body.password, body.role, body.name)
    return {"status": "ok", "email": body.email, "role": body.role}


class EmbeddingsRequest(BaseModel):
    force: bool = False


@router.post("/api/admin/embeddings/generate")
@limiter.limit("1/minute")
async def admin_generate_embeddings(request: Request, body: EmbeddingsRequest, background_tasks: BackgroundTasks):
    """Generate sentence-transformers embeddings for all skills."""
    background_tasks.add_task(_run_embeddings, force=body.force)
    return {"status": "started", "message": "Generating embeddings in background"}


def _run_embeddings(force: bool = False) -> None:
    import asyncio
    from src.cli.embeddings import main as emb_main
    asyncio.run(emb_main(force=force))
    logger.info("embeddings_generated", force=force)


class StudentImportItem(BaseModel):
    full_name: str
    group_name: str
    direction_code: str = "09.03.02"
    skills: str = ""


@router.post("/api/admin/students/import")
@limiter.limit("5/minute")
async def admin_import_students(request: Request, body: list[StudentImportItem]):
    """Import students from JSON array."""
    import csv
    import io
    from src.cli.import_students import main as import_main

    # Write to temp CSV, then run import
    tmp = io.StringIO()
    w = csv.DictWriter(tmp, fieldnames=["full_name", "group_name", "direction_code", "skills"])
    w.writeheader()
    for item in body:
        w.writerow(item.model_dump())
    tmp.seek(0)

    from pathlib import Path
    tmp_path = Path(config.DATA_DIR) / "cache" / "_import_tmp.csv"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(tmp.getvalue(), encoding="utf-8")

    await import_main(str(tmp_path))
    tmp_path.unlink(missing_ok=True)
    return {"status": "ok", "imported": len(body)}


@router.post("/api/admin/skills/extend")
@limiter.limit("1/minute")
async def admin_extend_skills(request: Request, yes: bool = True):
    """Analyze vacancies and extend it_skills with new skills."""
    import argparse
    from src.cli.extend_skills import main as extend_main

    args = argparse.Namespace(interactive=False, yes=yes, coverage=False, dead=False, min_frequency=2)
    extend_main(args)
    return {"status": "ok", "message": "Skills analysis completed"}


@router.get("/api/admin/export/db")
@limiter.limit("2/minute")
async def admin_export_db(request: Request, background_tasks: BackgroundTasks):
    """Export all DB tables to JSON files in data/export/."""
    background_tasks.add_task(_run_export)
    return {"status": "started", "message": "Exporting database in background"}


def _run_export() -> None:
    import asyncio
    from src.cli.export_json import main as export_main
    asyncio.run(export_main())
    logger.info("db_export_completed")


@router.post("/api/admin/db/backup")
@limiter.limit("1/minute")
async def admin_backup_db(request: Request):
    """Create a PostgreSQL dump in data/backups/."""
    from src.cli.backup_db import main as backup_main
    backup_main()
    return {"status": "ok", "message": "Backup created"}


@router.post("/api/admin/db/export-results")
@limiter.limit("1/minute")
async def admin_export_results(request: Request, background_tasks: BackgroundTasks):
    """Migrate existing JSON pipeline results to PostgreSQL."""
    background_tasks.add_task(_run_export_results)
    return {"status": "started", "message": "Exporting results in background"}


def _run_export_results() -> None:
    import asyncio
    from src.cli.export_results import main as er_main
    asyncio.run(er_main())


# ---------- Frontend log endpoint ----------


class FrontendLogRequest(BaseModel):
    action: str
    detail: str = ""


@router.post("/api/log")
@limiter.limit("30/minute")
async def frontend_log(request: Request, body: FrontendLogRequest):
    """Log frontend actions (button clicks, page views, errors)."""
    from src.api_pkg.request_logger import _log_buffer, LogEntry
    _log_buffer.append(LogEntry(
        method="ACTION",
        path=body.action,
        status=200,
        duration_ms=0,
        user_email=getattr(request.state, "user", None),
        source="frontend",
    ))
    return {"status": "ok"}
