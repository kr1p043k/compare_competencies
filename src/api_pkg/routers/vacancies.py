"""Vacancies: list, detail, stats."""

import structlog
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.models.api_responses import (
    VacanciesResponse,
    VacancyDetailResponse,
    VacancyStatsResponse,
)

from src.api_pkg import deps

logger = structlog.get_logger("api")

router = APIRouter(tags=["vacancies"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/vacancies", response_model=VacanciesResponse)
@limiter.limit("60/minute")
async def get_vacancies(
    request: Request,
    limit: int = Query(50, ge=1, le=500, description="Количество вакансий"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    experience: str | None = Query(
        None, description="Фильтр по опыту: junior, middle, senior"
    ),
    search: str | None = Query(None, description="Поиск по названию"),
    months: int | None = Query(None, ge=1, le=24, description="Период в месяцах"),
    vacancies: list = Depends(deps.get_basic_vacancies),
):
    filtered = vacancies.copy()
    if experience:
        exp_lower = experience.lower()
        filtered = [
            v
            for v in filtered
            if (
                isinstance(v.get("experience"), dict)
                and exp_lower in v["experience"].get("id", "").lower()
            )
            or (
                isinstance(v.get("experience"), str)
                and exp_lower in v["experience"].lower()
            )
            or exp_lower in v.get("name", "").lower()
        ]
    if search:
        search_lower = search.lower()
        filtered = [
            v
            for v in filtered
            if search_lower in v.get("name", "").lower()
        ]
    if months:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months * 30)
        filtered = [
            v
            for v in filtered
            if v.get("published_at") and isinstance(v["published_at"], str)
               and datetime.fromisoformat(v["published_at"].replace("Z", "+00:00").replace("+0300", "+03:00").replace("+0400", "+04:00")) >= cutoff
        ]
    total = len(filtered)
    items = filtered[offset : offset + limit]
    formatted_items = []
    for vac in items:
        skills = (
            vac.get("extracted_skills", [])[:10] if "extracted_skills" in vac else []
        )
        exp = "middle"
        if "experience" in vac:
            exp_obj = vac["experience"]
            if isinstance(exp_obj, dict):
                exp_id = exp_obj.get("id", "").lower()
                if "junior" in exp_id or "less1" in exp_id or "no_experience" in exp_id:
                    exp = "junior"
                elif "senior" in exp_id or "morethan10" in exp_id:
                    exp = "senior"
        salary_from = None
        salary_to = None
        salary_currency = "RUR"
        if "salary" in vac and vac["salary"]:
            sal = vac["salary"]
            salary_from = sal.get("from")
            salary_to = sal.get("to")
            salary_currency = sal.get("currency", "RUR")
        employer_name = "Не указано"
        employer_logo = None
        if "employer" in vac and vac["employer"]:
            emp = vac["employer"]
            employer_name = emp.get("name", "Не указано")
            if "logo_urls" in emp and emp["logo_urls"]:
                employer_logo = emp["logo_urls"].get("240") or emp["logo_urls"].get(
                    "90"
                )
        is_spam = vac.get("is_spam", False)
        spam_reason = vac.get("spam_reason", "")
        formatted_items.append(
            {
                "id": vac.get("id"),
                "name": vac.get("name", "Без названия"),
                "experience": exp,
                "salary_from": salary_from,
                "salary_to": salary_to,
                "salary_currency": salary_currency,
                "employer_name": employer_name,
                "employer_logo": employer_logo,
                "is_spam": is_spam,
                "spam_reason": spam_reason,
                "area": vac.get("area", {}).get("name", "Не указано")
                if isinstance(vac.get("area"), dict)
                else "Не указано",
                "published_at": vac.get("published_at"),
                "alternate_url": vac.get("alternate_url"),
                "skills": skills,
                "snippet": vac.get("snippet") or {},
            }
        )
    return {
        "items": formatted_items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total,
    }


@router.get("/vacancies/info")
async def get_vacancies_info():
    import os, json
    from datetime import datetime
    from src import config
    from src.api_pkg import deps
    from src.db import get_pool

    f = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    bf = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
    raw = f if f.exists() else bf
    info = {"count": 0, "file_modified": None, "date_range": None, "load_error": deps.vacancy_load_error}

    pool = get_pool()
    last = await pool.fetchrow(
        "SELECT completed_at, stats FROM pipeline_runs "
        "WHERE action = 'full-cycle' AND status = 'completed' "
        "ORDER BY completed_at DESC LIMIT 1"
    )
    try:
        total = await pool.fetchval(
            "SELECT COUNT(*) FROM vacancies WHERE parsed_skills IS NOT NULL"
        )
        info["total_vacancies"] = total or 0
    except Exception as exc:
        logger.warning("vacancies_info_db_unavailable", error=str(exc))
        info["total_vacancies"] = 0
        info["db_available"] = False

    info["last_updated"] = str(last["completed_at"]) if last else None
    info["last_pipeline_stats"] = last["stats"] if last else None

    if raw.exists():
        mtime = os.path.getmtime(raw)
        info["file_modified"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        try:
            import asyncio
            data = await asyncio.to_thread(lambda: json.loads(raw.read_bytes()))
            info["count"] = len(data)
            dates = sorted(set(v.get("published_at", "")[:10] for v in data if v.get("published_at")))
            if dates:
                info["date_range"] = {"from": dates[0], "to": dates[-1]}
        except Exception:
            pass
    return info


@router.get("/vacancies/{vacancy_id}", response_model=VacancyDetailResponse)
@limiter.limit("60/minute")
async def get_vacancy_detail(
    request: Request,
    vacancy_id: str,
    vacancies: list = Depends(deps.get_basic_vacancies),
):
    for vac in vacancies:
        if vac.get("id") == vacancy_id:
            skills = (
                vac.get("extracted_skills", []) if "extracted_skills" in vac else []
            )
            return {
                "id": vac.get("id"),
                "name": vac.get("name"),
                "description": vac.get("description", ""),
                "experience": vac.get("experience") or "",
                "salary": vac.get("salary"),
                "employer": vac.get("employer"),
                "area": vac.get("area"),
                "published_at": vac.get("published_at"),
                "alternate_url": vac.get("alternate_url"),
                "skills": skills,
                "schedule": vac.get("schedule"),
                "employment": vac.get("employment"),
                "key_skills": vac.get("key_skills", []),
                "snippet": vac.get("snippet"),
            }
    raise HTTPException(status_code=404, detail="Вакансия не найдена")


@router.get("/vacancies/stats/summary", response_model=VacancyStatsResponse)
@limiter.limit("30/minute")
async def get_vacancies_stats(
    request: Request,
    vacancies: list = Depends(deps.get_basic_vacancies),
):
    total = len(vacancies)
    junior = 0
    middle = 0
    senior = 0
    salaries = []
    for vac in vacancies:
        exp = "middle"
        if "experience" in vac:
            exp_obj = vac["experience"]
            if isinstance(exp_obj, dict):
                exp_id = exp_obj.get("id", "").lower()
                if "junior" in exp_id or "less1" in exp_id:
                    exp = "junior"
                elif "senior" in exp_id or "morethan10" in exp_id:
                    exp = "senior"
        if exp == "junior":
            junior += 1
        elif exp == "senior":
            senior += 1
        else:
            middle += 1
        if "salary" in vac and vac["salary"]:
            sal = vac["salary"]
            if sal.get("from"):
                salaries.append(sal["from"])
            if sal.get("to"):
                salaries.append(sal["to"])
    avg_salary = sum(salaries) / len(salaries) if salaries else 0
    return {
        "total": total,
        "by_experience": {"junior": junior, "middle": middle, "senior": senior},
        "salary": {
            "average": round(avg_salary, 0),
            "min": min(salaries) if salaries else 0,
            "max": max(salaries) if salaries else 0,
            "count": len(salaries),
        },
    }
