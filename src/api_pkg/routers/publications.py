"""Публикации хаба ЮФУ (hub.sfedu.ru) — прокси-эндпоинты.

Источник данных — публичный API хаба ``/api/publications/``. Список открыт,
карточка ``/api/assets/{id}/`` использует ``HUB_API_TOKEN``; без токена (или
если хаб его отклонил) запись ищется перебором страниц с начала списка.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api_pkg.routers.auth import get_current_user, user_error_detail
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
limiter = Limiter(key_func=get_remote_address)

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
