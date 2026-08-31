"""KRM Teacher: доступ преподавателей и РОП к KRM-данным своих направлений.

Права:
- admin  — все направления (полный доступ).
- rop    — только направления, привязанные через user_directions.
- teacher— направления из user_directions; если привязок нет — все (fallback,
           чтобы не потерять доступ для существующих преподавателей без привязки).

Роуты монтируются под /api/krm/teacher/*.
"""

import json
import re
from typing import Any

import structlog
from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.db import get_pool

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["krm_teacher"])
limiter = Limiter(key_func=get_remote_address)

_DIR_CODE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}(?:_\w+)?$")


class GapRequest(BaseModel):
    topic: str


# ---------- authorization helpers ----------


async def _current_user(request: Request) -> dict[str, Any]:
    from src.api_pkg.routers.auth import get_current_user

    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _list_all_dirs() -> list[dict]:
    """Список всех направлений из KRM-файлов (аналог teacher._list_krm_directions)."""
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


async def _user_directions(user: dict[str, Any]) -> list[str]:
    """Коды направлений, доступные пользователю.

    admin — все; rop/teacher — из user_directions; teacher без привязок — все.
    """
    role = user.get("r")
    if role == "admin":
        return [d["dir_code"] for d in _list_all_dirs()]

    pool = get_pool()
    if pool is None:
        return []
    rows = await pool.fetch(
        "SELECT dir_code FROM user_directions WHERE user_id=$1 ORDER BY dir_code",
        user.get("uid"),
    )
    codes = [r["dir_code"] for r in rows]

    # teacher с пустой привязкой не теряет доступ (fallback: все направления)
    if role == "teacher" and not codes:
        return [d["dir_code"] for d in _list_all_dirs()]

    return codes


async def _ensure_dir_access(user: dict[str, Any], dir_code: str) -> None:
    """403, если у пользователя нет доступа к dir_code."""
    _validate_dir_code(dir_code)
    allow = await _user_directions(user)
    if dir_code not in allow:
        raise HTTPException(
            status_code=403,
            detail=f"Доступ к направлению '{dir_code}' запрещён",
        )


def _validate_dir_code(dir_code: str) -> None:
    if not _DIR_CODE_RE.match(dir_code):
        raise HTTPException(status_code=400, detail="Invalid direction code format")


def _load_krm_direction(dir_code: str) -> dict:
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


@router.get("/krm/teacher/my-directions")
@limiter.limit("60/minute")
async def krm_teacher_my_directions(request: Request):
    """Направления, доступные текущему пользователю (с метаданными)."""
    user = await _current_user(request)
    allowed = await _user_directions(user)
    all_dirs = _list_all_dirs()
    return [d for d in all_dirs if d["dir_code"] in allowed]


@router.get("/krm/teacher/check-access")
@limiter.limit("60/minute")
async def krm_teacher_check_access(request: Request, dir_code: str = ""):
    """Проверка доступа к направлению. Возвращает {access: true/false, dir_code}."""
    user = await _current_user(request)
    try:
        _validate_dir_code(dir_code)
        allowed = await _user_directions(user)
        return {"access": dir_code in allowed, "dir_code": dir_code}
    except HTTPException:
        return {"access": False, "dir_code": dir_code}


@router.get("/krm/teacher/directions/{dir_code}")
@limiter.limit("60/minute")
async def krm_teacher_direction(request: Request, dir_code: str):
    """Полные данные направления (direction_name/profile/disciplines)."""
    user = await _current_user(request)
    await _ensure_dir_access(user, dir_code)
    return _load_krm_direction(dir_code)


@router.get("/krm/teacher/directions/{dir_code}/disciplines")
@limiter.limit("60/minute")
async def krm_teacher_disciplines(request: Request, dir_code: str):
    """Список дисциплин направления с числом компетенций и навыков."""
    user = await _current_user(request)
    await _ensure_dir_access(user, dir_code)
    d = _load_krm_direction(dir_code).get("disciplines", {}) or {}
    return [
        {
            "name": name,
            "competencies_count": len(info.get("competencies", [])),
            "skills_count": sum(len(s) for s in info.get("skills", {}).values()),
        }
        for name, info in sorted(d.items())
    ]


@router.get("/krm/teacher/directions/{dir_code}/disciplines/{discipline_name:path}")
@limiter.limit("60/minute")
async def krm_teacher_discipline_detail(request: Request, dir_code: str, discipline_name: str):
    """Детали дисциплины: компетенции и ЗУН (skills по каждой компетенции)."""
    user = await _current_user(request)
    await _ensure_dir_access(user, dir_code)
    d = _load_krm_direction(dir_code).get("disciplines", {}) or {}

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
        "dir_code": dir_code,
        "competencies": [
            {
                "code": comp,
                "skills": info.get("skills", {}).get(comp, []),
            }
            for comp in info.get("competencies", [])
        ],
    }


@router.post("/krm/teacher/directions/{dir_code}/gap")
@limiter.limit("20/minute")
async def krm_teacher_gap(request: Request, dir_code: str, payload: GapRequest = Body(...)):
    """Анализ разрыва компетенций по теме для направления преподавателя.

    Использует локальный AcademicGapAnalyzer (эмбеддинги) с фильтром доступа.
    """
    import asyncio

    user = await _current_user(request)
    await _ensure_dir_access(user, dir_code)

    from src.analyzers.academic_gap import AcademicGapAnalyzer

    try:
        result = await asyncio.to_thread(
            AcademicGapAnalyzer(dir_code=dir_code).analyze, payload.topic
        )
    except Exception as exc:
        logger.error("krm_teacher_gap_failed", dir_code=dir_code, error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Не удалось выполнить анализ разрыва",
        ) from None
    return {"dir_code": dir_code, **result}
