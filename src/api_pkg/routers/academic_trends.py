"""Прокси к Course Trends API ЮФУ (статьи и рекомендации по программам).

Новый сервис ЮФУ отдаёт предрассчитанные данные по программе (`pid`):
тренды/статьи по курсам, покрытие, рекомендации. Авторизация - тот же
hub-JWT, что и у старого academic-api, поэтому берём его из сессии тем же
способом (`_get_sso_token`).

Ответы сервиса не описаны моделями, поэтому наружу отдаём JSON как есть,
добавляя только `source` и `pid`, чтобы фронт мог показать происхождение.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from src.api_pkg.routers.academic import _get_sso_token
from src.api_pkg.routers.auth import require_any_role
from src.infrastructure.course_trends import CourseTrendsClient, CourseTrendsError
from src.infrastructure.course_trends_normalize import extract_articles

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["academic-trends"])

# Статьи/рекомендации по программе считаются заранее, но ответ может быть
# объёмным; держим таймаут выше обычного.
TRENDS_TIMEOUT = 90.0

_ROLES = ["admin", "teacher", "rop"]


async def _fetch(request: Request, path: str) -> Any:
    """Достаёт hub-JWT из сессии и запрашивает Course Trends API."""
    sso_token, _token_hash = await _get_sso_token(request)
    client = CourseTrendsClient(sso_token, timeout=TRENDS_TIMEOUT)
    try:
        return await asyncio.to_thread(client._get, path)
    except CourseTrendsError as exc:
        if exc.status_code == 401:
            raise HTTPException(
                status_code=419,
                detail="Доступ к сервису ЮФУ истёк (токен действует 1 час). "
                       "Войдите через hub.sfedu.ru",
            ) from None
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail="Программа не найдена в сервисе ЮФУ") from None
        logger.warning("academic_trends_failed", path=path, error=str(exc),
                       status=exc.status_code)
        raise HTTPException(status_code=502, detail=str(exc)) from None


@router.get("/academic/programs", dependencies=[Depends(require_any_role(*_ROLES))])
async def list_programs(request: Request) -> Any:
    """Список программ сервиса ЮФУ: pid и метки (курсов, среднее покрытие)."""
    return await _fetch(request, "/programs")


@router.get("/academic/glossary", dependencies=[Depends(require_any_role(*_ROLES))])
async def glossary(request: Request) -> Any:
    """Глоссарий терминов сервиса (что означает каждая метка)."""
    return await _fetch(request, "/meta/glossary")


@router.get("/academic/programs/{pid}/summary",
            dependencies=[Depends(require_any_role(*_ROLES))])
async def program_summary(pid: str, request: Request) -> Any:
    """Краткая сводка по программе для виджета."""
    return await _fetch(request, f"/programs/{pid}/summary")


@router.get("/academic/programs/{pid}/course-trends",
            dependencies=[Depends(require_any_role(*_ROLES))])
async def course_trends(pid: str, request: Request) -> Any:
    """Тренды и статьи по курсам программы."""
    return await _fetch(request, f"/programs/{pid}/course-trends")


@router.get("/academic/programs/{pid}/course-coverage",
            dependencies=[Depends(require_any_role(*_ROLES))])
async def course_coverage(pid: str, request: Request) -> Any:
    """Покрытие компетенций по курсам программы."""
    return await _fetch(request, f"/programs/{pid}/course-coverage")


@router.get("/academic/programs/{pid}/recommendations",
            dependencies=[Depends(require_any_role(*_ROLES))])
async def recommendations(pid: str, request: Request) -> Any:
    """Предрассчитанные рекомендации по программе."""
    return await _fetch(request, f"/programs/{pid}/recommendations")


@router.get("/academic/programs/{pid}", dependencies=[Depends(require_any_role(*_ROLES))])
async def program_bundle(pid: str, request: Request) -> Any:
    """Всё по программе одним ответом (summary + тренды + покрытие + рекомендации)."""
    return await _fetch(request, f"/programs/{pid}")


@router.get("/academic/programs/{pid}/articles",
            dependencies=[Depends(require_any_role(*_ROLES))])
async def program_articles(
    pid: str,
    request: Request,
    limit: int = 500,
    offset: int = 0,
    query: str | None = None,
) -> dict[str, Any]:
    """Статьи программы в каноническом виде.

    Сервис ЮФУ не публикует схемы ответов, поэтому сырой JSON проходит через
    `extract_articles`: он находит публикации по смыслу полей, а не по
    фиксированной модели. Наружу отдаётся один стабильный формат, чтобы фронт
    не зависел от переименований на стороне сервиса.

    `query` фильтрует по заголовку, аннотации и авторам (регистр не важен).
    """
    raw = await _fetch(request, f"/programs/{pid}/course-trends")
    articles = extract_articles(raw)

    if query:
        needle = query.strip().lower()
        articles = [
            a for a in articles
            if needle in a["title"].lower()
            or needle in (a["abstract"] or "").lower()
            or any(needle in author.lower() for author in a["authors"])
        ]

    articles.sort(key=lambda a: (a["year"] or 0, a["citations"] or 0), reverse=True)
    limit = max(1, min(limit, 2000))
    offset = max(0, offset)
    return {
        "pid": pid,
        "total": len(articles),
        "limit": limit,
        "offset": offset,
        "items": articles[offset:offset + limit],
        "source": "academic-api-dev (Course Trends)",
    }
