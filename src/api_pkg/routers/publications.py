"""Публикации хаба ЮФУ (hub.sfedu.ru) — прокси-эндпоинты.

Источник данных — публичный API хаба ``/api/publications/``. Список открыт,
карточка ``/api/assets/{id}/`` использует ``HUB_API_TOKEN``; без токена (или
если хаб его отклонил) запись ищется перебором страниц с начала списка.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from pydantic import BaseModel, Field

from src.api_pkg.rate_limit import limiter
from src.api_pkg.routers.auth import get_current_user, require_authenticated, user_error_detail
from src.infrastructure.hub_publications import (
    HubAuthRequiredError,
    HubNotFoundError,
    HubUnavailableError,
    get_client,
)
from src.models.api_responses import (
    PublicationListResponse,
    PublicationOut,
    PublicationSearchResponse,
    PublicationTypesResponse,
)

router = APIRouter(tags=["publications"])
logger = structlog.get_logger(__name__)

_NOT_FOUND_MSG = "Публикация не найдена в каталоге хаба"
_UNAVAILABLE_MSG = "Сервис хаба временно недоступен. Попробуйте позже"
_FORBIDDEN_MSG = "Хаб отклонил запрос. Попробуйте позже"


@router.get("/publications", response_model=PublicationListResponse)
@limiter.limit("60/minute")
async def list_publications(
    request: Request,
    response: Response,
    page: int = Query(1, ge=1, description="Номер страницы (с 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    type_id: int | None = Query(None, ge=1, description="Фильтр по типу публикации"),
    creator_id: int | None = Query(None, ge=1, description="Фильтр по id автора хаба"),
) -> PublicationListResponse:
    """Свежие публикации хаба ЮФУ."""
    response.headers["Cache-Control"] = "public, max-age=60"
    try:
        return await get_client().list_publications(
            page=page,
            page_size=page_size,
            type_id=type_id,
            creator_id=creator_id,
        )
    except HubNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=await user_error_detail(request, str(exc), _NOT_FOUND_MSG),
        ) from None
    except HubUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=await user_error_detail(request, str(exc), _UNAVAILABLE_MSG),
        ) from None
    except HubAuthRequiredError as exc:
        raise HTTPException(
            status_code=502,
            detail=await user_error_detail(request, str(exc), _FORBIDDEN_MSG),
        ) from None


@router.get("/publications/search", response_model=PublicationSearchResponse)
@limiter.limit("30/minute")
async def search_publications(
    request: Request,
    response: Response,
    q: str = Query(..., min_length=2, max_length=200, description="Подстрока в названии/авторах/подписи"),
    max_pages: int = Query(10, ge=1, le=10, description="Сколько последних страниц просматривать"),
    limit: int = Query(20, ge=1, le=200, description="Размер выдачи"),
    offset: int = Query(0, ge=0, description="Смещение в выдаче"),
    type_id: int | None = Query(None, ge=1, description="Фильтр по типу публикации"),
    creator_id: int | None = Query(None, ge=1, description="Фильтр по id автора хаба"),
) -> PublicationSearchResponse:
    """Поиск по публикациям хаба: серверного search у хаба нет, сканируем последние страницы."""
    response.headers["Cache-Control"] = "no-store"
    try:
        result = await get_client().search_publications(
            q,
            max_pages=max_pages,
            type_id=type_id,
            creator_id=creator_id,
        )
    except HubUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=await user_error_detail(request, str(exc), _UNAVAILABLE_MSG),
        ) from None
    except HubAuthRequiredError as exc:
        raise HTTPException(
            status_code=502,
            detail=await user_error_detail(request, str(exc), _FORBIDDEN_MSG),
        ) from None
    matches = result["matches"]
    return {
        "query": q,
        "matches": matches[offset : offset + limit],
        "total": len(matches),
        "scanned_pages": result["scanned_pages"],
        "truncated": result["truncated"],
        "total_scanned": result["total_scanned"],
    }


# ---------------------------------------------------------------------------
# Глубокий поиск (background job). Хаб не имеет полнотекстового индекса, а
# обычный /publications/search сканирует максимум _MAX_SCAN_PAGES (10) страниц.
# Глубокий поиск читает до 500 страниц в фоне и отдаёт прогресс, чтобы
# пользователь не висел на синхронном запросе.
#
# Храним задания в памяти процесса. Это приемлемо, пока приложение работает
# ОДНИМ воркером (наш деплой): не нужен Redis/Celery, а переживать рестарт
# заданию не обязательно - пользователь просто запустит его заново. При
# нескольких воркерах это сломается: (1) задание, созданное на воркере A,
# не видно на воркере B (GET/DELETE вернут 404); (2) лимит «одно задание на
# пользователя» и MAX_JOBS считаются независимо на каждом воркере; (3) отмену
# и rate-limit не увидят соседние воркеры. Для настоящего мультиворкового
# деплоя нужен внешний брокер задач (Celery/ARQ) и общий кэш.
# ---------------------------------------------------------------------------

_MAX_JOBS = 20
_JOB_TTL_S = 30 * 60
_MAX_DEEP_PAGES = 500


@dataclass
class _DeepSearchJob:
    job_id: str
    uid: str
    query: str
    status: str  # "running" | "done" | "error" | "cancelled"
    created_at: float
    updated_at: float
    scanned_pages: int = 0
    total_pages: int = 0
    match_count: int = 0
    matches: list[dict[str, Any]] = field(default_factory=list)
    total_scanned: int = 0
    elapsed_ms: int = 0
    truncated: bool = False
    error: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


#: job_id -> задание. Общий словарь процесса (см. комментарий выше).
_jobs: dict[str, _DeepSearchJob] = {}


def _now() -> float:
    return time.time()


def _uid_of(user: dict[str, Any]) -> str:
    return str(user.get("uid") or user.get("u") or "")


def _prune_expired(now: float | None = None) -> None:
    """Удаляет задания, не обновлявшиеся дольше `_JOB_TTL_S`."""
    now = _now() if now is None else now
    # Выполняющиеся задачи не трогаем: ими владеет asyncio.task, и удаление
    # записи сделало бы отмену невозможной. Если все слоты заняты живыми
    # задачами, _register_job вернёт False (503) - это честный предел.
    for job_id in [jid for jid, job in _jobs.items()
                   if job.status != "running" and now - job.updated_at > _JOB_TTL_S]:
        _jobs.pop(job_id, None)


def _register_job(job: _DeepSearchJob) -> bool:
    """Кладёт задание в словарь, вытесняя самую старую ЗАВЕРШЁННУЮ запись.

    :returns: ``False``, если в словаре уже `_MAX_JOBS` активных заданий и
        вытеснять нечего (новое задание не зарегистрировано).
    """
    _prune_expired()
    if len(_jobs) >= _MAX_JOBS:
        finished = sorted(
            (j.updated_at, jid) for jid, j in _jobs.items() if j.status != "running"
        )
        if not finished:
            return False
        _jobs.pop(finished[0][1], None)
    _jobs[job.job_id] = job
    return True


async def _run_deep_search(
    job_id: str,
    query: str,
    type_id: int | None,
    creator_id: int | None,
    max_pages: int,
) -> None:
    """Фоновая задача: гонит `deep_search` и фиксирует результат в словаре."""
    job = _jobs.get(job_id)
    if job is None:
        return

    async def on_progress(scanned_pages: int, total_pages: int, match_count: int) -> bool:
        current = _jobs.get(job_id)
        if current is None:
            return False
        current.scanned_pages = scanned_pages
        current.total_pages = total_pages
        current.match_count = match_count
        current.updated_at = _now()
        return not current.cancel_event.is_set()

    try:
        result = await get_client().deep_search(
            query,
            type_id=type_id,
            creator_id=creator_id,
            max_pages=max_pages,
            on_progress=on_progress,
        )
    except HubUnavailableError:
        job = _jobs.get(job_id)
        if job is not None and job.status == "running":
            job.status = "error"
            job.error = _UNAVAILABLE_MSG
            job.updated_at = _now()
        return
    except HubAuthRequiredError:
        job = _jobs.get(job_id)
        if job is not None and job.status == "running":
            job.status = "error"
            job.error = _FORBIDDEN_MSG
            job.updated_at = _now()
        return
    except Exception:  # noqa: BLE001 - фоновая задача не должна умирать молча
        logger.exception("deep_search_job_failed", job_id=job_id)
        job = _jobs.get(job_id)
        if job is not None and job.status == "running":
            job.status = "error"
            job.error = "Глубокий поиск завершился ошибкой"
            job.updated_at = _now()
        return

    job = _jobs.get(job_id)
    if job is None:
        return
    job.matches = result["matches"]
    job.total_scanned = result["total_scanned"]
    job.scanned_pages = result["scanned_pages"]
    job.total_pages = result["total_pages"]
    job.elapsed_ms = result["elapsed_ms"]
    job.truncated = result["truncated"]
    job.match_count = len(result["matches"])
    # Если пришёл DELETE, статус уже "cancelled": не перезаписываем обратно.
    if job.status == "running":
        job.status = "done" if not job.cancel_event.is_set() else "cancelled"
    job.updated_at = _now()


class DeepSearchRequest(BaseModel):
    """Тело запроса на запуск глубокого поиска."""

    q: str = Field(..., min_length=2, max_length=200, description="Подстрока для поиска")
    max_pages: int = Field(
        _MAX_DEEP_PAGES, ge=1, le=_MAX_DEEP_PAGES, description="Сколько страниц корпуса просмотреть"
    )
    type_id: int | None = Field(None, ge=1, description="Фильтр по типу публикации")
    creator_id: int | None = Field(None, ge=1, description="Фильтр по id автора хаба")


class DeepSearchStartedOut(BaseModel):
    """Ответ на POST/DELETE: id задания и его статус."""

    job_id: str
    status: str


class DeepSearchProgressOut(BaseModel):
    """Прогресс задания: страницы и процент."""

    scanned_pages: int
    total_pages: int
    pct: float


class DeepSearchStatusOut(BaseModel):
    """Состояние задания глубокого поиска."""

    job_id: str
    status: str
    progress: DeepSearchProgressOut
    matches: list[PublicationOut]
    total: int
    total_scanned: int = 0
    truncated: bool = False
    elapsed_ms: int = 0
    error: str | None = None


@router.post(
    "/publications/deep-search",
    response_model=DeepSearchStartedOut,
    status_code=202,
)
@limiter.limit("2/minute")
async def start_deep_search(
    request: Request,
    q: str = Query(..., min_length=2, max_length=200),
    max_pages: int = Query(_MAX_DEEP_PAGES, ge=1, le=_MAX_DEEP_PAGES),
    type_id: int | None = Query(None, ge=1),
    creator_id: int | None = Query(None, ge=1),
    user: dict[str, Any] = Depends(require_authenticated()),
) -> DeepSearchStartedOut:
    """Запускает фоновый глубокий поиск, возвращает id сразу (202).

    На обычный /publications/search тут не влияем: это явная, тяжёлая операция,
    ограниченная 2 запусками в минуту и одним активным заданием на пользователя.
    """
    uid = _uid_of(user)
    _prune_expired()

    # Одно задание на пользователя: если активное уже есть - вернуть его, а не плодить второе.
    for existing in _jobs.values():
        if existing.uid == uid and existing.status == "running":
            return DeepSearchStartedOut(job_id=existing.job_id, status="running")

    now = _now()
    job = _DeepSearchJob(
        job_id=secrets.token_urlsafe(16),
        uid=uid,
        query=q,
        status="running",
        created_at=now,
        updated_at=now,
    )
    if not _register_job(job):
        raise HTTPException(
            status_code=503,
            detail="Слишком много активных заданий глубокого поиска. Попробуйте позже.",
        )
    job.task = asyncio.create_task(
        _run_deep_search(job.job_id, q, type_id, creator_id, max_pages)
    )
    logger.info(
        "deep_search_started",
        job_id=job.job_id,
        uid=uid,
        query=q,
        max_pages=max_pages,
    )
    return DeepSearchStartedOut(job_id=job.job_id, status="running")


@router.get(
    "/publications/deep-search/{job_id}",
    response_model=DeepSearchStatusOut,
)
async def get_deep_search(
    request: Request,
    job_id: str = Path(..., description="id задания глубокого поиска"),
    user: dict[str, Any] = Depends(require_authenticated()),
) -> dict[str, Any]:
    """Состояние задания: прогресс, число совпадений, а по завершении - результат."""
    _prune_expired()
    job = _jobs.get(job_id)
    uid = _uid_of(user)
    if job is None or (job.uid != uid and user.get("r") != "admin"):
        raise HTTPException(status_code=404, detail="Задание не найдено")
    running = job.status == "running"
    pct = round(job.scanned_pages / job.total_pages * 100, 1) if job.total_pages else 0.0
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": {
            "scanned_pages": job.scanned_pages,
            "total_pages": job.total_pages,
            "pct": pct,
        },
        # Пока идёт скан полный список не шлём (может быть десятки тысяч записей
        # каждые 1.5 с), только счётчик total; полный список - по завершении.
        "matches": [] if running else job.matches,
        "total": job.match_count,
        "total_scanned": job.total_scanned,
        "truncated": job.truncated,
        "elapsed_ms": job.elapsed_ms,
        "error": job.error,
    }


@router.delete(
    "/publications/deep-search/{job_id}",
    response_model=DeepSearchStartedOut,
)
async def cancel_deep_search(
    request: Request,
    job_id: str = Path(..., description="id задания глубокого поиска"),
    user: dict[str, Any] = Depends(require_authenticated()),
) -> DeepSearchStartedOut:
    """Отменяет активное задание: выставляет флаг, скан остановится на чанке."""
    _prune_expired()
    job = _jobs.get(job_id)
    uid = _uid_of(user)
    if job is None or (job.uid != uid and user.get("r") != "admin"):
        raise HTTPException(status_code=404, detail="Задание не найдено")
    if job.status == "running":
        job.cancel_event.set()
        job.status = "cancelled"
        job.updated_at = _now()
    return DeepSearchStartedOut(job_id=job.job_id, status=job.status)


@router.get("/publications/types", response_model=PublicationTypesResponse)
@limiter.limit("30/minute")
async def list_publication_types(
    request: Request,
    response: Response,
    sample_pages: int = Query(
        3,
        ge=1,
        le=10,
        deprecated=True,
        description="Не используется: счётчики теперь точные, параметр оставлен для совместимости",
    ),
    refresh: bool = Query(False, description="Пересчитать счётчики, минуя суточный кэш"),
) -> PublicationTypesResponse:
    """Точные счётчики публикаций по типам.

    Считаются запросами ``page_size=1`` по известным id типов и кэшируются на
    сутки, поэтому сумма ``count`` равна общему числу записей хаба.
    """
    response.headers["Cache-Control"] = "public, max-age=3600"
    try:
        return await get_client().list_types(sample_pages=sample_pages, refresh=refresh)
    except HubUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=await user_error_detail(request, str(exc), _UNAVAILABLE_MSG),
        ) from None
    except HubAuthRequiredError as exc:
        raise HTTPException(
            status_code=502,
            detail=await user_error_detail(request, str(exc), _FORBIDDEN_MSG),
        ) from None


@router.get("/publications/{publication_id}", response_model=PublicationOut)
@limiter.limit("60/minute")
async def get_publication(
    request: Request,
    response: Response,
    publication_id: int = Path(..., ge=1, description="id публикации в хабе"),
    max_scan_pages: int = Query(10, ge=1, le=10, description="Сколько страниц перебрать без токена"),
) -> PublicationOut:
    """Карточка публикации хаба по id.

    Токен хаба используется только для авторизованных пользователей: он общий
    для организации, а анонимный скан ограничен ``max_scan_pages``.
    """
    response.headers["Cache-Control"] = "no-store"
    user = await get_current_user(request)
    try:
        return await get_client().get_publication(
            publication_id,
            max_scan_pages=max_scan_pages,
            use_token=user is not None,
        )
    except HubNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=await user_error_detail(request, str(exc), _NOT_FOUND_MSG),
        ) from None
    except HubUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=await user_error_detail(request, str(exc), _UNAVAILABLE_MSG),
        ) from None
    except HubAuthRequiredError as exc:
        raise HTTPException(
            status_code=502,
            detail=await user_error_detail(request, str(exc), _FORBIDDEN_MSG),
        ) from None
