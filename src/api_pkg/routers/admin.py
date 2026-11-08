"""Admin: whitelist management, student profiles, pipeline, Excel export."""

import asyncio
import contextlib
import io
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text

from src import config
from src.api_pkg import deps
from src.api_pkg.request_logger import actor_from_request, audit_action, get_logs_by_user
from src.api_pkg.routers.auth import (
    _hash_token,
    _request_token,
    get_current_user,
    require_any_role,
)
from src.db import get_pool
from src.parsing.utils import load_it_skills

logger = structlog.get_logger("api")

#: Таблица admin_totp подтверждённо существует (кэш True, см. _admin_totp_present).
_ADMIN_TOTP_PRESENT = False

router = APIRouter(tags=["admin"], dependencies=[Depends(require_any_role("admin", "teacher"))])
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


#: (mtime, skills) - файл whitelist читался с диска на каждый запрос.
_whitelist_cache: tuple[float, list[str]] | None = None


def _load_whitelist() -> list[str]:
    global _whitelist_cache
    try:
        mtime = IT_SKILLS_PATH.stat().st_mtime
    except OSError:
        _whitelist_cache = None
        return sorted(load_it_skills())
    if _whitelist_cache is not None and _whitelist_cache[0] == mtime:
        return list(_whitelist_cache[1])
    skills = sorted(load_it_skills())
    _whitelist_cache = (mtime, skills)
    return list(skills)


def _save_whitelist(skills: list[str]):
    global _whitelist_cache
    IT_SKILLS_PATH.write_text(
        json.dumps(sorted(set(skills)), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    deps.current_skills_set = set(skills)
    _whitelist_cache = None


def _write_json(path: str | Path, data: Any) -> None:
    """Синхронная запись JSON (запуск через asyncio.to_thread)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_log_chunk(path: str | Path, pos: int) -> tuple[str, int]:
    """Читает новые строки лога от позиции pos (в потоке, не на event loop)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        f.seek(pos)
        chunk = f.read()
        return chunk, f.tell()


def _read_log_tail_lines(path: str | Path, n: int) -> list[str]:
    """Последние n строк лога (в потоке, не на event loop)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()[-n:]


# ---------- Whitelist ----------


@router.get("/admin/whitelist", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def get_whitelist(request: Request):
    """Текущий whitelist навыков."""
    skills = _load_whitelist()
    return {"skills": skills, "total": len(skills)}


@router.post("/admin/whitelist/add", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("10/minute")
async def whitelist_add(request: Request, body: WhitelistAddRequest):
    """Добавить навыки в whitelist."""
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
    await audit_action(
        request, "skills.whitelist.add",
        f"added={len(current) - before} skills={sorted(set(body.skills))[:20]}",
    )
    return {
        "status": "ok",
        "message": f"Added {len(current) - before} skills",
        "added": len(current) - before,
        "total": len(current),
    }


@router.post("/admin/whitelist/remove", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("10/minute")
async def whitelist_remove(request: Request, body: WhitelistRemoveRequest):
    """Убрать навыки из whitelist."""
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
    await audit_action(
        request, "skills.whitelist.remove",
        f"removed={before - len(current)} skills={sorted(set(body.skills))[:20]}",
    )
    return {
        "status": "ok",
        "message": f"Removed {before - len(current)} skills",
        "removed": before - len(current),
        "total": len(current),
    }


@router.post("/admin/whitelist/backup", response_model=WhitelistBackupResponse, dependencies=[Depends(require_any_role("admin"))])  # noqa: E501
@limiter.limit("5/minute")
async def whitelist_backup(request: Request):
    """Бэкап whitelist в файл."""
    backup_dir = config.DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"it_skills_backup_{ts}.json"
    shutil.copy2(IT_SKILLS_PATH, backup_path)
    skills = _load_whitelist()
    await audit_action(request, "admin.skills.whitelist.backup")
    return WhitelistBackupResponse(
        status="ok",
        message=f"Backup saved to {backup_path.name}",
        backup_path=str(backup_path),
        total_skills=len(skills),
    )


# ---------- Student profiles ----------


def _scan_students_dir(students_dir: Path) -> list[dict[str, Any]]:
    """Сканирует каталог студентов и собирает профили (в потоке, не на event loop)."""
    students: list[dict[str, Any]] = []
    for f in sorted(students_dir.iterdir()):
        if f.suffix not in (".json",):
            continue
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
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                }
            )
        except Exception:
            students.append({"filename": f.name, "skills_count": 0, "modified": ""})
    return students


@router.get("/admin/students", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def list_students(request: Request):
    """Список студентов."""
    students_dir = config.DATA_DIR / "students"
    if not students_dir.exists():
        return {"students": [], "total": 0}
    students = await asyncio.to_thread(_scan_students_dir, students_dir)
    return {"students": students, "total": len(students)}


# ---------- Pipeline trigger ----------


class PipelineTriggerRequest(BaseModel):
    action: str = "full-cycle"
    regions: str = "0"
    skip_collection: bool = False
    run_gap_analysis: bool = True


@router.post("/admin/pipeline/trigger", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("2/minute")
async def admin_trigger_pipeline(
    request: Request, body: PipelineTriggerRequest, background_tasks: BackgroundTasks
):
    """Запустить пайплайн из админки."""
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
    from src.api_pkg.request_logger import audit_action
    await audit_action(request, "pipeline.trigger.manual",
                       f"action={body.action} task={task_id} regions={body.regions}")
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


@router.get("/admin/export/excel", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("3/minute")
async def export_excel(request: Request):
    """Экспорт вакансий в Excel."""
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
    # Уникальный временный файл: при двух одновременных выгрузках общий путь
    # vacancies_export.xlsx перезаписывался, и первый запрос отдавал чужой файл.
    import os
    import tempfile

    fd, name = tempfile.mkstemp(
        prefix="vacancies_export_", suffix=".xlsx", dir=str(config.REPORTS_DIR)
    )
    os.close(fd)
    excel_path = Path(name)
    try:
        await asyncio.to_thread(df.to_excel, excel_path, index=False, engine="openpyxl")
    except Exception:
        excel_path.unlink(missing_ok=True)
        raise

    if not excel_path.exists():
        raise HTTPException(status_code=500, detail="Excel export failed")

    from starlette.background import BackgroundTask

    await audit_action(request, "admin.export.excel")
    return FileResponse(
        str(excel_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="vacancies_export.xlsx",
        background=BackgroundTask(excel_path.unlink, missing_ok=True),
    )


def _build_full_report_zip(result_dir: Path) -> bytes:
    """Собрать ZIP с отчётами (синхронно, вызывается через asyncio.to_thread)."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(result_dir.rglob("*")):
            if fpath.is_file() and fpath.suffix in {".json", ".png", ".xlsx"}:
                arcname = str(fpath.relative_to(result_dir.parent))
                zf.write(str(fpath), arcname)
    return zip_buffer.getvalue()


@router.get("/admin/export/full-report", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("2/minute")
async def export_full_report(request: Request):
    """Полный отчёт (Excel/ZIP)."""
    from fastapi.responses import StreamingResponse

    result_dir = config.DATA_RESULT_DIR
    if not result_dir.exists() or not any(result_dir.rglob("*")):
        raise HTTPException(status_code=404, detail="No report data found")

    # Сборка архива - CPU и диски: уносим с event loop.
    payload = await asyncio.to_thread(_build_full_report_zip, result_dir)

    await audit_action(request, "admin.export.full-report")
    return StreamingResponse(
        iter([payload]),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="full_report.zip"'},
    )


@router.get("/admin/users", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_users(
    request: Request,
    q: str | None = Query(None, max_length=100),
    role: str | None = None,
    is_active: bool | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Список пользователей (фильтры q/role/is_active, пагинация)."""
    from sqlalchemy import select, text

    from src.database import async_session_factory
    from src.models.krm_models import UserDirection

    if role is not None and role not in ("admin", "teacher", "rop", "student"):
        raise HTTPException(
            status_code=400, detail="Invalid role. Valid: admin, teacher, rop, student"
        )

    where_parts: list[str] = []
    params: dict[str, Any] = {}
    if q:
        params["q"] = f"%{_escape_like(q.strip())}%"
        where_parts.append(
            "(lower(u.email) LIKE :q ESCAPE '\\' OR lower(u.full_name) LIKE :q ESCAPE '\\')"
        )
    if role is not None:
        params["role"] = role
        where_parts.append("u.role = :role")
    if is_active is not None:
        params["is_active"] = is_active
        where_parts.append("u.is_active = :is_active")
    where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

    async with async_session_factory() as session:
        totp_expr = (
            "COALESCE((SELECT true FROM admin_totp t "
            "WHERE t.user_id = u.id AND t.confirmed LIMIT 1), false)"
            if await _admin_totp_present(session)
            else "false"
        )
        result = await session.execute(
            text(
                f"""
                SELECT u.id, u.email, u.full_name, u.role, u.is_active,
                       {totp_expr} AS totp_enabled,
                       (SELECT COUNT(1) FROM sessions s
                         WHERE s.user_id = u.id AND s.logged_out_at IS NULL) AS open_sessions,
                       (SELECT MAX(s.last_activity) FROM sessions s
                         WHERE s.user_id = u.id) AS last_activity
                  FROM users u{where_clause}
                 ORDER BY u.created_at
                 LIMIT :limit OFFSET :offset
                """
            ),
            {**params, "limit": limit, "offset": offset},
        )
        rows = result.fetchall()
        total = (
            await session.execute(text(f"SELECT COUNT(*) FROM users u{where_clause}"), params)
        ).scalar_one()
        # Направления грузим только для страницы: раньше select без WHERE тянул
        # всю таблицу user_directions на каждый запрос списка.
        page_ids = [str(u.id) for u in rows]
        dir_map: dict[str, list[str]] = {}
        if page_ids:
            ud_result = await session.execute(
                select(UserDirection.user_id, UserDirection.dir_code).where(
                    UserDirection.user_id.in_(page_ids)
                )
            )
            for uid, code in ud_result.all():
                dir_map.setdefault(str(uid), []).append(code)

    log_counts = get_logs_by_user()
    return {
        "users": [
            _user_payload(
                user_id=str(u.id),
                email=u.email,
                full_name=u.full_name,
                role=u.role,
                is_active=u.is_active,
                directions=dir_map.get(str(u.id), []),
                log_counts=log_counts,
                totp_enabled=bool(getattr(u, "totp_enabled", False)),
                open_sessions=int(getattr(u, "open_sessions", 0) or 0),
                last_activity=getattr(u, "last_activity", None),
            )
            for u in rows
        ],
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


@router.get("/admin/directions", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_directions(request: Request):
    """Список всех направлений (dir_code) из KRM-файлов для привязки РОП."""
    from src.api_pkg.routers.teacher import _list_krm_directions
    return _list_krm_directions()


class UserDirectionsBody(BaseModel):
    directions: list[str] = []


class UserUpdateRequest(BaseModel):
    role: str | None = None
    is_active: bool | None = None
    full_name: str | None = Field(default=None, max_length=200)
    caller_password: str | None = Field(default=None, max_length=72)


class PasswordResetRequest(BaseModel):
    # bcrypt обрезает пароль на 72 байтах: длиннее принимать бессмысленно.
    new_password: str = Field(min_length=10, max_length=72)
    caller_password: str | None = Field(default=None, max_length=72)


class AdminStepUpRequest(BaseModel):
    """Подтверждение опасной операции над другим администратором."""

    caller_password: str = Field(min_length=1, max_length=72)


def _validate_uuid(value: str, label: str = "id") -> str:
    """Проверяет, что value - валидный UUID; иначе 400."""
    try:
        from uuid import UUID

        return str(UUID(value))
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {label}: not a valid UUID") from None


async def _admin_totp_present(session) -> bool:
    """Есть ли таблица admin_totp. Старая БД могла не применить миграцию.

    True кэшируется навсегда (таблица не исчезает), False перепроверяется:
    после /auth/2fa/setup таблица создаётся на лету и панель должна это увидеть
    без рестарта API.
    """
    global _ADMIN_TOTP_PRESENT
    if _ADMIN_TOTP_PRESENT:
        return True
    try:
        present = bool(
            (
                await session.execute(text("SELECT to_regclass('public.admin_totp') IS NOT NULL"))
            ).scalar_one()
        )
    except Exception:
        return False
    if present:
        _ADMIN_TOTP_PRESENT = True
    return present


def _escape_like(value: str) -> str:
    """Экранирование спецсимволов LIKE: \\, %, _ (ESCAPE '\\')."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def _assert_caller_password(session, caller: dict[str, Any] | None, password: str | None) -> None:
    """Step-up для операций над другим администратором: подтвердить свой пароль.

    Одна украденная сессия администратора не должна позволять сбросить пароль
    или снять 2FA у второго администратора и тем самым забрать его аккаунт.
    """
    if caller is None or not caller.get("uid"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not password:
        raise HTTPException(
            status_code=403,
            detail="Для операции с другим администратором подтвердите свой пароль",
        )
    from sqlalchemy import text

    ok = (
        await session.execute(
            text(
                "SELECT EXISTS(SELECT 1 FROM users "
                "WHERE id = :uid AND is_active = true AND crypt(:pw, password_hash) = password_hash)"
            ),
            {"uid": caller["uid"], "pw": password},
        )
    ).scalar_one()
    if not ok:
        raise HTTPException(status_code=403, detail="Неверный пароль администратора")


def _user_payload(
    user_id: str,
    email: str,
    full_name: str,
    role: str,
    is_active: bool,
    directions: list[str],
    log_counts: dict[str, int],
    totp_enabled: bool = False,
    open_sessions: int = 0,
    last_activity: Any = None,
) -> dict[str, Any]:
    """Единый вид пользователя для списка/деталей/обновления (backward-compatible)."""
    return {
        "id": str(user_id),
        "username": email,
        "role": role,
        "name": full_name,
        "total_requests": log_counts.get(email, 0),
        "is_active": is_active,
        "directions": sorted(directions),
        "totp_enabled": bool(totp_enabled),
        "open_sessions": open_sessions,
        "last_activity": last_activity.isoformat() if last_activity is not None else None,
    }


@router.get("/admin/users/{user_id}/directions", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_user_directions(request: Request, user_id: str):
    """Направления пользователя."""
    from sqlalchemy import select

    from src.database import async_session_factory
    from src.models.krm_models import UserDirection

    user_id = _validate_uuid(user_id, "user_id")
    async with async_session_factory() as session:
        result = await session.execute(
            select(UserDirection.dir_code).where(UserDirection.user_id == user_id)
        )
        return {"user_id": user_id, "directions": sorted(row[0] for row in result)}


@router.put("/admin/users/{user_id}/directions", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_set_user_directions(request: Request, user_id: str, body: UserDirectionsBody):
    """Назначить направления пользователю."""
    from sqlalchemy import delete

    from src.database import async_session_factory
    from src.models.krm_models import User, UserDirection

    user_id = _validate_uuid(user_id, "user_id")
    async with async_session_factory() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        await session.execute(delete(UserDirection).where(UserDirection.user_id == user_id))
        for code in body.directions:
            session.add(UserDirection(user_id=user_id, dir_code=code))
        await session.commit()
    await audit_action(
        request, "admin.user.directions",
        f"uid={user_id} dirs={','.join(sorted(body.directions))}",
    )
    return {"status": "ok", "user_id": user_id, "directions": sorted(body.directions)}


@router.get("/admin/users/{user_id}", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_user_detail(request: Request, user_id: str):
    """Профиль пользователя: поля, направления, сессии, статус 2FA."""
    from sqlalchemy import select, text

    from src.database import async_session_factory
    from src.models.krm_models import Session, User, UserDirection

    user_id = _validate_uuid(user_id, "user_id")
    async with async_session_factory() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        dirs_result = await session.execute(
            select(UserDirection.dir_code).where(UserDirection.user_id == user_id)
        )
        directions = sorted(row[0] for row in dirs_result)
        sess_result = await session.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.logged_in_at.desc())
        )
        sessions = sess_result.scalars().all()
        totp_row = None
        if await _admin_totp_present(session):
            totp = await session.execute(
                text("SELECT confirmed FROM admin_totp WHERE user_id = :uid"), {"uid": user_id}
            )
            totp_row = totp.first()

    log_counts = get_logs_by_user()
    open_sessions = sum(1 for s in sessions if s.logged_out_at is None)
    last_activity = max(
        (s.last_activity for s in sessions if s.last_activity is not None), default=None
    )
    totp_enabled = bool(totp_row and totp_row[0])
    return {
        "user": _user_payload(
            user_id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            directions=directions,
            log_counts=log_counts,
            totp_enabled=totp_enabled,
            open_sessions=open_sessions,
            last_activity=last_activity,
        ),
        "directions": directions,
        "sessions": [
            {
                "id": str(s.id),
                "ip_address": s.ip_address,
                "user_agent": s.user_agent,
                "logged_in_at": s.logged_in_at.isoformat() if s.logged_in_at else None,
                "last_activity": s.last_activity.isoformat() if s.last_activity else None,
                "is_open": s.logged_out_at is None,
            }
            for s in sessions
        ],
        "totp_enabled": totp_enabled,
    }


@router.patch("/admin/users/{user_id}", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_user_update(request: Request, user_id: str, body: UserUpdateRequest):
    """Обновить роль/active/имя с защитой от потери последнего админа."""
    from sqlalchemy import text

    from src.database import async_session_factory
    from src.models.krm_models import User

    user_id = _validate_uuid(user_id, "user_id")
    if body.role is not None and body.role not in ("admin", "teacher", "rop", "student"):
        raise HTTPException(status_code=400, detail="Invalid role")

    caller = await get_current_user(request)
    async with async_session_factory() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        # Охрана 1: нельзя снять себе роль админа или деактивировать себя.
        if (
            caller is not None
            and str(user.id) == caller.get("uid")
            and ((body.role is not None and body.role != "admin") or body.is_active is False)
        ):
            raise HTTPException(
                status_code=400,
                detail="Нельзя изменить собственную роль или деактивировать себя",
            )

        target_is_admin = user.role == "admin"
        will_demote = target_is_admin and body.role is not None and body.role != "admin"
        will_deactivate = target_is_admin and bool(user.is_active) and body.is_active is False

        # Охрана 2: нельзя оставить систему без активного администратора.
        # Проверка и запись идут в одной транзакции под advisory-lock: иначе два
        # параллельных понижения двух последних админов оба увидят «останется
        # один» и система останется без администратора.
        if will_demote or will_deactivate:
            await session.execute(text("SELECT pg_advisory_xact_lock(hashtext('admin_guard'))"))
            active_admins = (
                await session.execute(
                    text(
                        "SELECT COUNT(*) FROM users "
                        "WHERE role = 'admin' AND is_active = true AND id <> :uid"
                    ),
                    {"uid": user_id},
                )
            ).scalar_one()
            if int(active_admins) == 0:
                raise HTTPException(
                    status_code=409,
                    detail="Нельзя оставить систему без активного администратора",
                )
            # Демонтаж другого администратора - действие с последствиями для чужого
            # аккаунта: без подтверждения пароля это способ тихо понизить коллегу.
            if caller is not None and str(user.id) != caller.get("uid"):
                await _assert_caller_password(session, caller, body.caller_password)

        changes: list[str] = []
        if body.role is not None and body.role != user.role:
            user.role = body.role
            changes.append(f"role={body.role}")
        if body.is_active is not None and bool(body.is_active) != bool(user.is_active):
            user.is_active = body.is_active
            changes.append(f"is_active={body.is_active}")
        if body.full_name is not None and body.full_name != user.full_name:
            user.full_name = body.full_name
            changes.append(f"full_name={body.full_name!r}")

        user.updated_at = datetime.now(UTC)
        await session.commit()

        totp_row = None
        if await _admin_totp_present(session):
            totp = await session.execute(
                text("SELECT confirmed FROM admin_totp WHERE user_id = :uid"), {"uid": user_id}
            )
            totp_row = totp.first()
        counts = (
            await session.execute(
                text(
                    "SELECT COUNT(*) FILTER (WHERE logged_out_at IS NULL) AS open_sessions, "
                    "MAX(last_activity) AS last_activity FROM sessions WHERE user_id = :uid"
                ),
                {"uid": user_id},
            )
        ).first()
        # Row из text() индексируется позиционно, не по имени колонки.
        open_sessions_value = int(counts[0] or 0) if counts else 0
        last_activity_value = counts[1] if counts else None
        dirs_result = await session.execute(
            text("SELECT dir_code FROM user_directions WHERE user_id = :uid ORDER BY dir_code"),
            {"uid": user_id},
        )
        directions = [row[0] for row in dirs_result]

    target = f"uid={user_id}" + ((" " + " ".join(changes)) if changes else "")
    await audit_action(request, "admin.user.update", target)
    return {
        "user": _user_payload(
            user_id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            directions=directions,
            log_counts=get_logs_by_user(),
            totp_enabled=bool(totp_row and totp_row[0]),
            open_sessions=open_sessions_value,
            last_activity=last_activity_value,
        )
    }


@router.post("/admin/users/{user_id}/password", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("10/minute")
async def admin_user_password(request: Request, user_id: str, body: PasswordResetRequest):
    """Сбросить пароль; все открытые сессии, кроме текущей, отзываются."""
    from sqlalchemy import text

    from src.database import async_session_factory
    from src.models.krm_models import User

    user_id = _validate_uuid(user_id, "user_id")
    caller = await get_current_user(request)
    async with async_session_factory() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        if user.role == "admin" and caller is not None and str(user.id) != caller.get("uid"):
            await _assert_caller_password(session, caller, body.caller_password)
        result = await session.execute(
            text("SELECT crypt(:pw, gen_salt('bf', 12)) AS pw_hash"),
            {"pw": body.new_password},
        )
        pw_hash = result.scalar_one()
        user.password_hash = pw_hash
        user.updated_at = datetime.now(UTC)
        mine = _hash_token(_request_token(request))
        res = await session.execute(
            text(
                "UPDATE sessions SET logged_out_at = NOW() "
                "WHERE user_id = :uid AND logged_out_at IS NULL AND token_hash <> :mine"
            ),
            {"uid": user_id, "mine": mine},
        )
        revoked = res.rowcount or 0
        await session.commit()

    await audit_action(request, "admin.user.password_reset", f"uid={user_id}")
    return {"status": "ok", "revoked_sessions": revoked}


@router.post("/admin/users/{user_id}/sessions/revoke", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_user_sessions_revoke(request: Request, user_id: str):
    """Отозвать все открытые сессии пользователя, кроме сессии вызывающего."""
    from sqlalchemy import text

    from src.database import async_session_factory
    from src.models.krm_models import User

    user_id = _validate_uuid(user_id, "user_id")
    async with async_session_factory() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        mine = _hash_token(_request_token(request))
        res = await session.execute(
            text(
                "UPDATE sessions SET logged_out_at = NOW() "
                "WHERE user_id = :uid AND logged_out_at IS NULL AND token_hash <> :mine"
            ),
            {"uid": user_id, "mine": mine},
        )
        revoked = res.rowcount or 0
        await session.commit()

    await audit_action(request, "admin.user.sessions_revoke", f"uid={user_id}")
    return {"status": "ok", "revoked": revoked}


@router.post("/admin/users/{user_id}/2fa/reset", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("10/minute")
async def admin_user_2fa_reset(request: Request, user_id: str, body: AdminStepUpRequest | None = None):
    """Сбросить 2FA пользователя (удалить admin_totp)."""
    from sqlalchemy import text

    from src.database import async_session_factory
    from src.models.krm_models import User

    user_id = _validate_uuid(user_id, "user_id")
    caller = await get_current_user(request)
    async with async_session_factory() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        if user.role == "admin" and caller is not None and str(user.id) != caller.get("uid"):
            await _assert_caller_password(session, caller, body.caller_password if body else None)
        if not await _admin_totp_present(session):
            raise HTTPException(status_code=400, detail="2FA не настроена")
        res = await session.execute(
            text("DELETE FROM admin_totp WHERE user_id = :uid"), {"uid": user_id}
        )
        if (res.rowcount or 0) == 0:
            raise HTTPException(status_code=400, detail="2FA не настроена")
        await session.commit()

    await audit_action(request, "admin.user.2fa_reset", f"uid={user_id}")
    return {"status": "ok"}


@router.get("/admin/sessions", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_sessions(
    request: Request,
    user: str = Query("", description="Email пользователя"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Все сессии с привязкой к пользователям (новые первыми)."""
    from sqlalchemy import text

    from src.database import async_session_factory

    where_clause = ""
    params: dict[str, Any] = {}
    if user.strip():
        params["user"] = user.strip().lower()
        where_clause = " WHERE lower(u.email) = :user"

    async with async_session_factory() as session:
        rows = (
            await session.execute(
                text(
                    f"""
                    SELECT s.id, u.email, u.role, s.ip_address, s.user_agent,
                           s.logged_in_at, s.last_activity,
                           (s.logged_out_at IS NULL) AS is_open
                      FROM sessions s JOIN users u ON u.id = s.user_id
                    {where_clause}
                     ORDER BY s.logged_in_at DESC
                     LIMIT :limit OFFSET :offset
                    """
                ),
                {**params, "limit": limit, "offset": offset},
            )
        ).fetchall()
        total = (
            await session.execute(
                text(
                    f"SELECT COUNT(*) FROM sessions s JOIN users u ON u.id = s.user_id{where_clause}"
                ),
                params,
            )
        ).scalar_one()

    return {
        "sessions": [
            {
                "id": str(r.id),
                "email": r.email,
                "role": r.role,
                "ip_address": r.ip_address,
                "user_agent": r.user_agent,
                "logged_in_at": r.logged_in_at.isoformat() if r.logged_in_at else None,
                "last_activity": r.last_activity.isoformat() if r.last_activity else None,
                "is_open": bool(r.is_open),
            }
            for r in rows
        ],
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


@router.get("/admin/monitoring", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("60/minute")
async def admin_monitoring(
    request: Request,
    names: str = Query("", description="Фильтр по именам метрик через запятую"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """JSON-мониторинг (метрики) с фильтром и пагинацией.

    Без параметров отдавал ВСЕ семейства метрик (~435 строк и растёт с каждым
    новым эндпоинтом), что тяжело и для панели, и для клиента. Теперь: фильтр
    по именам, limit/offset и общее число семейств в ответе.
    """
    wanted = {n.strip() for n in names.split(",") if n.strip()}
    # Разбор ~сотен семейств метрик - в потоке, а не на event loop.
    families = await asyncio.to_thread(_parse_metric_families, wanted)

    total = len(families)
    page = dict(sorted(families.items())[offset: offset + limit])
    return {"total": total, "limit": limit, "offset": offset, "families": page}


def _parse_metric_families(wanted: set[str]) -> dict[str, dict]:
    """Разобрать экспозицию Prometheus в словарь семейств (в потоке)."""
    from prometheus_client.parser import text_string_to_metric_families

    from src.monitoring.metrics import get_metrics

    raw, _ = get_metrics()
    families: dict[str, dict] = {}
    for f in text_string_to_metric_families(raw.decode()):
        if wanted and f.name not in wanted:
            continue
        families[f.name] = {
            "name": f.name,
            "type": f.type,
            "samples": [
                {"name": s.name, "labels": s.labels, "value": s.value} for s in f.samples
            ],
        }
    return families


def _p95_from_buckets(buckets: list[tuple[float, float]]) -> float | None:
    """p95 по гистограммным бакетам [(upper, cumulative)] (прометеев формат).

    Возвращает наименьшую конечную верхнюю границу бакета, чей накопительный
    счётчик достигает 95% (total * 0.95). Если порог достигается только в бакете
    +Inf, возвращает наибольшую КОНЕЧНУЮ границу (семантика: p95 >= максимальный
    измеренный бакет), но никогда 1e18 или бесконечность.
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = total * 0.95
    for upper, cum in buckets:
        if upper != float("inf") and cum >= target:
            return upper
    finite = [upper for upper, _ in buckets if upper != float("inf")]
    return max(finite) if finite else None


def _metrics_p95_and_busiest() -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Топ-5 по p95 и топ-5 по числу запросов (синхронно, через to_thread)."""
    from prometheus_client.parser import text_string_to_metric_families

    from src.monitoring.metrics import get_metrics

    raw, _ = get_metrics()
    buckets: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for f in text_string_to_metric_families(raw.decode()):
        if f.name == "api_request_duration_seconds":
            for s in f.samples:
                if s.name.endswith("_bucket"):
                    key = (s.labels.get("method", ""), s.labels.get("path", ""))
                    # Бакет +Inf держим как настоящий inf (не подменяем на 1e18),
                    # иначе p95 из +Inf-бакета вернул бы абсурдное число.
                    buckets.setdefault(key, []).append(
                        (float(s.labels.get("le", "+Inf")), s.value)
                    )
    p95_by_endpoint: dict[str, float] = {}
    requests_by_endpoint: dict[str, int] = {}
    for (method, path), b in buckets.items():
        b_sorted = sorted(b, key=lambda x: x[0])
        p95 = _p95_from_buckets(b_sorted)
        name = f"{method} {path}"
        if p95 is not None:
            p95_by_endpoint[name] = p95
        requests_by_endpoint[name] = int(b_sorted[-1][1]) if b_sorted else 0
    top_latency = dict(
        sorted(p95_by_endpoint.items(), key=lambda kv: kv[1], reverse=True)[:5]
    )
    top_busiest = [
        {"endpoint": name, "requests": count}
        for name, count in sorted(
            requests_by_endpoint.items(), key=lambda kv: kv[1], reverse=True
        )[:5]
    ]
    return top_latency, top_busiest


@router.get("/admin/monitoring/summary", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_monitoring_summary(request: Request):
    """Сводка для панели здоровья: латентность, ошибки, сессии, прогоны, свежесть.

    latency_p95_by_endpoint - топ-5 САМЫХ МЕДЛЕННЫХ эндпоинтов (по p95, с ненулевой
    выборкой), busiest_by_requests - топ-5 самых нагруженных (по числу запросов):
    ключ по имени больше не вводит в заблуждение, информация о нагрузке сохраняется.
    """
    out: dict[str, Any] = {"latency_p95_by_endpoint": {}, "busiest_by_requests": [],
                           "errors_1h": {"4xx": 0, "5xx": 0}, "sessions_active": None,
                           "sessions_total_open": None, "recent_runs": [], "freshness": {}}
    try:
        latency, busiest = await asyncio.to_thread(_metrics_p95_and_busiest)
        out["latency_p95_by_endpoint"] = latency
        out["busiest_by_requests"] = busiest
    except Exception as e:
        logger.warning("monitoring_summary_metrics_failed", error=str(e))

    try:
        pool = get_pool()
        if pool is not None:
            out["sessions_active"] = await pool.fetchval(
                "SELECT COUNT(*) FROM sessions WHERE logged_out_at IS NULL "
                "AND last_activity > NOW() - INTERVAL '15 minutes'")
            out["sessions_total_open"] = await pool.fetchval(
                "SELECT COUNT(*) FROM sessions WHERE logged_out_at IS NULL")
            rows = await pool.fetch(
                "SELECT action, status, started_at, completed_at, error_message "
                "FROM pipeline_runs ORDER BY started_at DESC LIMIT 10")
            out["recent_runs"] = [dict(r) | {"started_at": str(r["started_at"]),
                                             "completed_at": str(r["completed_at"])} for r in rows]
            vac = await pool.fetchrow(
                "SELECT COUNT(*) AS n, MIN(published_at)::date AS mn, "
                "MAX(published_at)::date AS mx FROM vacancies WHERE published_at IS NOT NULL")
            if vac:
                out["freshness"] = {"vacancies": int(vac["n"] or 0),
                                    "date_from": str(vac["mn"]) if vac["mn"] else None,
                                    "date_to": str(vac["mx"]) if vac["mx"] else None}
            err = await pool.fetch(
                "SELECT CASE WHEN status BETWEEN 400 AND 499 THEN '4xx' ELSE '5xx' END AS cls, "
                "COUNT(*) AS n FROM request_logs WHERE created_at > NOW() - INTERVAL '1 hour' "
                "AND status >= 400 GROUP BY 1")
            for r in err:
                out["errors_1h"][r["cls"]] = int(r["n"])
    except Exception as e:
        logger.warning("monitoring_summary_db_failed", error=str(e))
    return out


@router.get("/admin/logs", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_logs(
    request: Request,
    user: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    action: str | None = None,
):
    """Логи запросов. AUDIT-строки несут detail 'action | target | by email (role)'.

    action - префикс-фильтр по audit-действию, напр. action=krm.recommendation.add.
    Только admin: строки содержат email пользователей, пути и коды ответов.
    """
    if user and user == "all":
        user = None
    from src.api_pkg.request_logger import get_logs_merged
    entries = await get_logs_merged(user=user, limit=limit, action=action)
    return {"logs": entries, "total": len(entries)}


@router.get("/admin/student-actions", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_student_actions(
    request: Request,
    user: str | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    """Действия студентов (история запросов + правки профилей). Видно админу/преподавателю."""
    from src.api_pkg.student_actions import get_actions

    entries = await get_actions(username=user, limit=limit)
    return {"actions": entries, "total": len(entries)}


@router.get("/admin/logs/file", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_logs_file(
    request: Request,
    lines: int = Query(300, ge=1, le=5000),
    level: str = "all",
):
    """Read the tail of logs/backend.log (system log file)."""
    from src.config import LOG_FILE

    if not LOG_FILE.exists():
        return {"file": str(LOG_FILE), "lines": [], "total": 0}
    try:
        raw_lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {"file": str(LOG_FILE), "lines": [], "total": 0}

    # Последняя строка может быть неполной (пишется на лету) - берём хвост целиком
    wanted = raw_lines[-max(lines, 1):]
    if level in ("error", "warning", "info", "debug"):
        wanted = [line for line in wanted if f"level={level}" in line or line.startswith(f"{level.upper()} ")]
        wanted = wanted[-max(lines, 1):]
    return {"file": str(LOG_FILE), "lines": wanted, "total": len(wanted)}


# ---------- DB Management (CLI → API) ----------


class SeedDBRequest(BaseModel):
    drop: bool = False


@router.post("/admin/db/seed", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("1/minute")
async def admin_seed_db(request: Request, body: SeedDBRequest, background_tasks: BackgroundTasks):
    """Seed database from JSON files (skills, disciplines, competencies)."""
    background_tasks.add_task(_run_seed, drop=body.drop)
    await audit_action(request, "admin.db.seed", f"drop={body.drop}")
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
    directions: list[str] = []


@router.post("/admin/users/create", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("10/minute")
async def admin_create_user(request: Request, body: CreateUserRequest):
    """Create a new user with bcrypt-hashed password (optionally bind directions)."""
    from src.cli.create_user import main as create_user_main
    await create_user_main(body.email, body.password, body.role, body.name, body.directions)
    await audit_action(request, "admin.user.create", f"email={body.email} role={body.role}")
    return {"status": "ok", "email": body.email, "role": body.role, "directions": body.directions}


class EmbeddingsRequest(BaseModel):
    force: bool = False


@router.post("/admin/embeddings/generate", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("1/minute")
async def admin_generate_embeddings(request: Request, body: EmbeddingsRequest, background_tasks: BackgroundTasks):
    """Generate sentence-transformers embeddings for all skills."""
    background_tasks.add_task(_run_embeddings, force=body.force)
    await audit_action(request, "admin.embeddings.generate", f"force={body.force}")
    return {"status": "started", "message": "Generating embeddings in background"}


def _run_embeddings(force: bool = False) -> None:
    import asyncio

    from src.cli.embeddings import main as emb_main
    asyncio.run(emb_main(force=force))
    logger.info("embeddings_generated", force=force)


class StudentImportItem(BaseModel):
    full_name: str = Field(max_length=200)
    group_name: str = Field(max_length=100)
    direction_code: str = Field(default="09.03.02", max_length=20)
    skills: str = Field(default="", max_length=10000)


@router.post("/admin/students/import", dependencies=[Depends(require_any_role("admin"))])
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

    tmp_path = Path(config.DATA_DIR) / "cache" / "_import_tmp.csv"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(tmp.getvalue(), encoding="utf-8")

    await import_main(str(tmp_path))
    tmp_path.unlink(missing_ok=True)
    await audit_action(request, "admin.students.import", f"count={len(body)}")
    return {"status": "ok", "imported": len(body)}


@router.post("/admin/skills/extend", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("1/minute")
async def admin_extend_skills(request: Request, background_tasks: BackgroundTasks, yes: bool = True):
    """Analyze vacancies and extend it_skills with new skills."""
    import argparse

    from src.cli.extend_skills import main as extend_main

    args = argparse.Namespace(interactive=False, yes=yes, coverage=False, dead=False, min_frequency=2)
    background_tasks.add_task(extend_main, args)
    await audit_action(request, "skills.extend.start", f"yes={yes}")
    return {"status": "ok", "message": "Skills analysis started in background"}


class CategorizeSkillsRequest(BaseModel):
    assignments: list[dict[str, str]]  # [{"skill": "...", "category": "..."}]


@router.get("/admin/skills/uncategorized", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("20/minute")
async def admin_skills_uncategorized(request: Request):
    """Некатегоризованные навыки it_skills + предложенная категория (эмбеддинг)."""
    from src.cli.taxonomy_audit import (
        MANUAL_OVERRIDES,
        build_prototypes,
        load_it_skills,
        load_taxonomy,
        suggest_categories,
        taxonomy_skill_set,
    )

    taxonomy = load_taxonomy()
    it_skills = load_it_skills()
    known = taxonomy_skill_set(taxonomy)
    uncategorized = sorted(it_skills - known)
    suggestions: dict[str, tuple[str, float]] = {}
    if uncategorized:
        try:
            prototypes = build_prototypes(taxonomy)
            suggestions = suggest_categories(uncategorized, prototypes)
        except Exception as exc:
            logger.warning("uncategorized_suggest_failed", error=str(exc))

    items = []
    for skill in uncategorized:
        cat_id, score = suggestions.get(skill, ("other", 0.0))
        items.append({
            "skill": skill,
            "category": MANUAL_OVERRIDES.get(skill, cat_id),
            "score": round(score, 3),
            "manual": skill in MANUAL_OVERRIDES,
        })
    return {"total": len(items), "items": items}


@router.post("/admin/skills/categorize", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_skills_categorize(request: Request, body: CategorizeSkillsRequest):
    """Записать навыки в указанные категории skill_taxonomy.json."""
    from src.cli.taxonomy_audit import TAXONOMY_PATH, load_taxonomy

    taxonomy = load_taxonomy()
    cats = taxonomy.get("categories", {})
    added = 0
    invalid = []
    for item in body.assignments:
        skill = (item.get("skill") or "").strip().lower()
        cat_id = (item.get("category") or "").strip()
        if not skill or cat_id not in cats:
            invalid.append({"skill": skill, "category": cat_id})
            continue
        lst = cats[cat_id].setdefault("skills", [])
        existing = {s.strip().lower() for s in lst}
        if skill in existing:
            continue
        lst.append(skill)
        added += 1
    if added:
        await asyncio.to_thread(_write_json, TAXONOMY_PATH, taxonomy)
        await audit_action(
            request, "skills.categorize",
            f"added={added} invalid={len(invalid)} "
            f"skills={[a.get('skill') for a in body.assignments][:20]}",
        )
    return {"status": "ok", "added": added, "invalid": invalid}


@router.get("/admin/skills/suggestions", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_skills_suggestions(request: Request, status: str = "pending"):
    """Очередь предложений навыков от преподавателей."""
    from src.api_pkg.skill_suggestions import load_all

    items = load_all()
    if status in ("pending", "approved", "rejected"):
        items = [s for s in items if s.get("status") == status]
    return {"suggestions": items}


@router.post("/admin/skills/suggestions/{suggestion_id}/approve", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_skills_approve(suggestion_id: str, request: Request):
    """Одобрить: в it_skills.json + категория таксономии."""
    from src.api_pkg.skill_suggestions import decide, load_all
    from src.cli.taxonomy_audit import TAXONOMY_PATH, load_taxonomy

    raw = await request.json()
    category = ((raw or {}).get("category", "") if isinstance(raw, dict) else "").strip()
    pending = [s for s in load_all() if s.get("id") == suggestion_id and s.get("status") == "pending"]
    if not pending:
        raise HTTPException(status_code=404, detail="Suggestion not found or decided")
    skill = pending[0]["skill"]

    taxonomy = load_taxonomy()
    cats = taxonomy.get("categories", {})
    if category and category not in cats:
        raise HTTPException(status_code=400, detail=f"Unknown category '{category}'")

    if category:
        lst = cats[category].setdefault("skills", [])
        if skill not in {s.strip().lower() for s in lst}:
            lst.append(skill)
        await asyncio.to_thread(_write_json, TAXONOMY_PATH, taxonomy)

    import json as _json

    current = _json.loads(await asyncio.to_thread(IT_SKILLS_PATH.read_text, encoding="utf-8"))
    if skill not in {str(s).lower() for s in current}:
        current.append(skill)
        current = sorted(set(current), key=lambda x: str(x).lower())
        await asyncio.to_thread(
            IT_SKILLS_PATH.write_text,
            _json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    decider, _role = await actor_from_request(request)
    entry = decide(suggestion_id, True, decided_by=decider)
    await audit_action(
        request, "skills.suggestion.approve",
        f"id={suggestion_id} skill={skill} category={category or '-'}",
    )
    return {"status": "ok", "suggestion": entry, "category": category or None}


@router.post("/admin/skills/suggestions/{suggestion_id}/reject", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def admin_skills_reject(request: Request, suggestion_id: str):
    """Отклонить предложение."""
    from src.api_pkg.skill_suggestions import decide

    decider, _role = await actor_from_request(request)
    entry = decide(suggestion_id, False, decided_by=decider)
    if not entry:
        raise HTTPException(status_code=404, detail="Suggestion not found or decided")
    await audit_action(
        request, "skills.suggestion.reject",
        f"id={suggestion_id} skill={entry.get('skill')}",
    )
    return {"status": "ok", "suggestion": entry}


@router.get("/admin/export/db", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("2/minute")
async def admin_export_db(request: Request, background_tasks: BackgroundTasks):
    """Export all DB tables to JSON files in data/export/."""
    background_tasks.add_task(_run_export)
    await audit_action(request, "admin.db.export")
    return {"status": "started", "message": "Exporting database in background"}


def _run_export() -> None:
    import asyncio

    from src.cli.export_json import main as export_main
    asyncio.run(export_main())
    logger.info("db_export_completed")


@router.post("/admin/db/backup", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("1/minute")
async def admin_backup_db(request: Request):
    """Create a PostgreSQL dump in data/backups/."""
    from src.cli.backup_db import main as backup_main

    # pg_dump - блокирующий внешний процесс: на event loop он останавливает весь
    # воркер на всё время дампа.
    await asyncio.to_thread(backup_main)
    await audit_action(request, "admin.db.backup")
    return {"status": "ok", "message": "Backup created"}


# ---------- Frontend log endpoint ----------


class FrontendLogRequest(BaseModel):
    action: str
    detail: str = ""


@router.post("/log")
@limiter.limit("30/minute")
async def frontend_log(request: Request, body: FrontendLogRequest):
    """Log frontend actions (button clicks, page views, errors)."""
    from src.api_pkg.request_logger import LogEntry, _log_buffer
    _log_buffer.append(LogEntry(
        method="ACTION",
        path=body.action,
        status=200,
        duration_ms=0,
        user_email=getattr(request.state, "user", None),
        source="frontend",
    ))
    return {"status": "ok"}


# ---------- Real-time system log streaming (WebSocket) ----------
# Отдельный роутер БЕЗ dependencies (require_any_role ждёт HTTP Request,
# а у WebSocket нет Request) - авторизация выполняется вручную в эндпоинте.

ws_log_router = APIRouter(tags=["logs-ws"])

#: Период keepalive-опроса сокета и пере-проверки сессии.
_WS_LOOP_INTERVAL_SEC = 30.0
_WS_SESSION_RECHECK_SEC = 30.0
#: Жёсткий лимит жизни соединения: долгий сокет переживает отзыв токена.
_WS_MAX_LIFETIME_SEC = 30.0 * 60.0

_TAIL_POLL_SEC = 0.7
_TAIL_INITIAL_LINES = 400
_TAIL_MAX_SEND = 200


async def _tail_backend_log(websocket: WebSocket, stop: asyncio.Event) -> None:
    """Читает logs/backend.log, шлёт хвост сразу и новые строки по мере появления."""
    from src.config import LOG_FILE

    pos = 0
    rotated = False
    while not stop.is_set():
        try:
            if not LOG_FILE.exists():
                await asyncio.sleep(_TAIL_POLL_SEC)
                continue
            size = LOG_FILE.stat().st_size
            if size < pos:
                rotated = True
                pos = 0
            chunk, pos = await asyncio.to_thread(_read_log_chunk, LOG_FILE, pos)
            if chunk:
                lines = chunk.splitlines()
                if rotated:
                    await websocket.send_json({"type": "rotated", "lines": lines})
                else:
                    for i in range(0, len(lines), _TAIL_MAX_SEND):
                        batch = lines[i:i + _TAIL_MAX_SEND]
                        await websocket.send_json({"type": "lines", "lines": batch})
                rotated = False
        except Exception as exc:
            try:
                await websocket.send_json({"type": "error", "error": str(exc)})
            except Exception:
                break
        await asyncio.sleep(_TAIL_POLL_SEC)


@ws_log_router.websocket("/admin/logs/ws")
async def admin_logs_ws(websocket: WebSocket):
    """Стрим системного лога backend.log в реальном времени (роль admin).

    Авторизация ДО accept: cookie или Authorization. Query-токен убран - адрес
    сокета попадает в логи прокси и историю браузера. Проверяем не только
    подпись, но и активную сессию в БД (resolve_session): иначе logout и
    деактивация не действуют на открытом сокете, а роль из payload остаётся
    прежней после понижения в правах.
    """
    try:
        from src.api_pkg.routers.auth import resolve_session
        from src.api_pkg.ws_origin import reject_if_bad_origin

        # CSWSH: сторонний сайт не должен открывать сокет от имени админа.
        if reject_if_bad_origin(websocket, config.ALLOWED_ORIGINS):
            await websocket.close(code=4003, reason="Forbidden origin")
            return

        eff_token = (
            websocket.cookies.get("token", "")
            or websocket.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        )
        if not eff_token:
            await websocket.close(code=4001, reason="Unauthorized")
            return
        user = await resolve_session(eff_token)
        if user is None:
            await websocket.close(code=4001, reason="Unauthorized")
            return
        if user.get("r") != "admin":
            logger.warning("admin_logs_ws_forbidden", uid=user.get("uid"), role=user.get("r"))
            await websocket.close(code=4003, reason="Forbidden")
            return
    except Exception:
        await websocket.close(code=4001, reason="Auth failed")
        return

    await websocket.accept()

    # Начальный хвост файла
    try:
        from src.config import LOG_FILE

        if LOG_FILE.exists():
            tail = await asyncio.to_thread(_read_log_tail_lines, LOG_FILE, _TAIL_INITIAL_LINES)
            await websocket.send_json({"type": "init", "lines": tail})
        else:
            await websocket.send_json({"type": "init", "lines": []})
    except Exception as exc:
        await websocket.send_json({"type": "error", "error": str(exc)})

    stop = asyncio.Event()
    tail_task = asyncio.create_task(_tail_backend_log(websocket, stop))
    loop = asyncio.get_running_loop()
    started = loop.time()
    last_recheck = started
    try:
        while True:
            try:
                await asyncio.wait_for(
                    websocket.receive_text(), timeout=_WS_LOOP_INTERVAL_SEC
                )
            except TimeoutError:
                pass  # клиент жив, просто нет сообщений
            except Exception:
                break

            now = loop.time()
            if now - last_recheck >= _WS_SESSION_RECHECK_SEC:
                last_recheck = now
                session_user = await resolve_session(eff_token)
                if session_user is None or session_user.get("r") != "admin":
                    logger.warning(
                        "admin_logs_ws_session_invalid",
                        uid=(session_user or {}).get("uid"),
                    )
                    await websocket.close(code=4001, reason="Session invalid")
                    break
            if now - started >= _WS_MAX_LIFETIME_SEC:
                await websocket.close(code=1000, reason="Socket lifetime exceeded")
                break
    finally:
        stop.set()
        tail_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tail_task


# ---------- Scheduler (background collector + nightly gap, admin-only) ----------


class SchedulerSettingsPatch(BaseModel):
    collector_enabled: bool | None = None
    collect_interval_hours: int | None = None
    daily_gap_enabled: bool | None = None


@router.get("/admin/scheduler/status", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("30/minute")
async def scheduler_status(request: Request):
    """Состояние планировщика фоновых задач."""
    from src.pipeline.background_collector import scheduler_status as _status
    return _status()


@router.put("/admin/scheduler/settings", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("10/minute")
async def scheduler_settings_update(request: Request, body: SchedulerSettingsPatch):
    """Тумблеры планировщика (persist в data/settings/scheduler.json)."""
    from src.pipeline.background_collector import save_scheduler_settings
    from src.pipeline.background_collector import scheduler_status as _status
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    save_scheduler_settings(patch)
    logger.info("scheduler_settings_updated", patch=patch)
    await audit_action(request, "admin.scheduler.settings", str(patch))
    return _status()


@router.post("/admin/scheduler/collect/now", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("5/minute")
async def scheduler_collect_now(request: Request):
    """Внеплановый сбор (force; 409 если планировщик занят)."""
    from src.pipeline.background_collector import _run_collect_once, is_scheduler_busy
    if is_scheduler_busy():
        raise HTTPException(status_code=409, detail="Планировщик уже выполняет задачу")
    asyncio.create_task(_run_collect_once(force=True))
    await audit_action(request, "admin.scheduler.collect")
    return {"status": "started", "job": "collect"}


@router.post("/admin/scheduler/gap/now", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("5/minute")
async def scheduler_gap_now(request: Request):
    """Внеплановый gap по всем профилям (409 если планировщик занят)."""
    from src.pipeline.background_collector import _run_scheduled_gap, is_scheduler_busy
    if is_scheduler_busy():
        raise HTTPException(status_code=409, detail="Планировщик уже выполняет задачу")
    asyncio.create_task(_run_scheduled_gap())
    await audit_action(request, "admin.scheduler.gap")
    return {"status": "started", "job": "gap"}

# ---------- Feature flags (admin-only): включение функций без рестарта ----------


class FeatureFlagPatch(BaseModel):
    enabled: bool


@router.get("/admin/features", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("60/minute")
async def admin_features(request: Request):
    """Состояние переключателей функций.

    Флаг 2FA здесь не отдаём: в этом модуле он был вторым источником правды
    (файл features.json), а реальный переключатель живёт в app_settings. Пока
    показывать его тут, оператор включит его и решит, что 2FA работает.
    """
    from src.api_pkg.feature_flags import FLAG_TOTP_2FA, all_flags

    return {"features": {k: v for k, v in all_flags().items() if k != FLAG_TOTP_2FA}}


@router.put("/admin/features/{flag}", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("20/minute")
async def admin_set_feature(flag: str, body: FeatureFlagPatch, request: Request):
    """Включить или выключить функцию на лету, без перезапуска сервиса."""
    from src.api_pkg.feature_flags import FLAG_TOTP_2FA, all_flags, set_flag

    if flag == FLAG_TOTP_2FA:
        # Раньше здесь был отдельный флаг 2FA, который login не читал: тумблер
        # переключался, а второй фактор не включался. Оставлен 409 с адресом
        # настоящего переключателя, чтобы никто не наступил на это снова.
        raise HTTPException(
            status_code=409,
            detail="2FA переключается через PUT /admin/security/2fa/settings",
        )
    try:
        set_flag(flag, body.enabled)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown feature '{flag}'") from None
    logger.info("admin_feature_toggled", flag=flag, enabled=body.enabled, uid=getattr(request.state, "user_uid", None))
    return {"features": {k: v for k, v in all_flags().items() if k != FLAG_TOTP_2FA}}



# ---------- Безопасность: мастер-переключатель 2FA ----------
# Флаг хранится в БД (app_settings), а не в .env: включение/выключение
# действует сразу на все процессы и переживает деплой.


class TwoFactorSettingsPatch(BaseModel):
    enabled: bool


async def _two_factor_payload() -> dict[str, Any]:
    from src.api_pkg.security_settings import get_two_factor_enabled
    from src.db import get_pool

    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    enabled = await get_two_factor_enabled(pool)
    enrolled = 0
    try:
        enrolled = int(
            await pool.fetchval("SELECT count(*) FROM admin_totp WHERE confirmed = true") or 0
        )
    except Exception:
        # Таблицы может не быть до применения миграции - это не ошибка экрана.
        enrolled = 0
    return {"enabled": enabled, "enrolled_admins": enrolled}


@router.get("/admin/security/2fa/settings", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("60/minute")
async def two_factor_settings(request: Request):
    """Текущее состояние мастер-переключателя 2FA."""
    return await _two_factor_payload()


@router.put("/admin/security/2fa/settings", dependencies=[Depends(require_any_role("admin"))])
@limiter.limit("10/minute")
async def two_factor_settings_update(request: Request, body: TwoFactorSettingsPatch):
    """Включить/выключить 2FA одним кликом (только админ).

    Выключение не требует кода: это аварийный тумблер.

    При ВКЛЮЧЕНИИ сессии остальных админов отзываются: иначе сессия, добытая
    до включения второго фактора, продолжает жить (до суток) и обходит новую
    защиту. Сессия того, кто включает, сохраняется, иначе он выкинул бы себя.
    Подключения пользователей при выключении не удаляются.
    """
    from src.api_pkg.routers.auth import _hash_token, _request_token, get_current_user
    from src.api_pkg.security_settings import set_two_factor_enabled
    from src.db import get_pool

    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    user = await get_current_user(request) or {}
    await set_two_factor_enabled(pool, body.enabled, user.get("uid"))
    if body.enabled:
        token_hash = _hash_token(_request_token(request))
        status = await pool.execute(
            """DELETE FROM sessions
                WHERE user_id IN (SELECT id FROM users WHERE role = 'admin')
                  AND token_hash <> $1""",
            token_hash,
        )
        revoked = int(status.split()[-1]) if status.startswith("DELETE") else 0
        logger.info("two_factor_enabled_sessions_revoked", count=revoked)
    await audit_action(request, "admin.2fa.toggle", f"enabled={body.enabled}")
    logger.info("two_factor_toggled", enabled=body.enabled, by=user.get("u"))
    return await _two_factor_payload()
