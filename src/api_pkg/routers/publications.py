"""Публикации хаба ЮФУ (hub.sfedu.ru) — прокси-эндпоинты.

Источник данных — публичный API хаба ``/api/publications/``. Список открыт,
карточка ``/api/assets/{id}/`` требует ``HUB_API_TOKEN``; без токена запись
ищется перебором последних страниц.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.infrastructure.hub_publications import (
    HubAuthRequiredError,
    HubNotFoundError,
    HubPublicationsClient,
    HubUnavailableError,
    get_client,
)

logger = structlog.get_logger("api")

router = APIRouter(tags=["publications"])
limiter = Limiter(key_func=get_remote_address)


class PublicationFile(BaseModel):
    url: str
    filename: str | None = None
    size: int | None = None
    content_type: str | None = None


class PublicationOut(BaseModel):
    id: int
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    type_id: int | None = None
    type_name: str | None = None
    signature: str | None = None
    doi: str | None = None
    isbn: str | None = None
    track_id: str | None = None
    created_at: str | None = None
    year: int | None = None
    academic_subject: list[str] = Field(default_factory=list)
    education_program: list[str] = Field(default_factory=list)
    files: list[PublicationFile] = Field(default_factory=list)
    url: str | None = None
    creator_username: str | None = None


class PublicationListResponse(BaseModel):
    items: list[PublicationOut]
    total: int
    page: int
    page_size: int
    num_pages: int
    source: str = "hub.sfedu.ru"


class PublicationSearchResponse(BaseModel):
    query: str
    matches: list[PublicationOut]
    total: int
    scanned_pages: int
    truncated: bool
    source: str = "hub.sfedu.ru"


class PublicationType(BaseModel):
    id: int
    name: str | None = None
    count: int


class PublicationTypesResponse(BaseModel):
    types: list[PublicationType]
    scanned_pages: int
    sample_size: int
    source: str = "hub.sfedu.ru"


def _client() -> HubPublicationsClient:
    return get_client()


@router.get("/publications", response_model=PublicationListResponse)
@limiter.limit("60/minute")
async def list_publications(
    request: Request,
    page: int = Query(1, ge=1, description="Номер страницы (с 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    type_id: int | None = Query(None, ge=1, description="Фильтр по типу публикации"),
    creator_id: int | None = Query(None, ge=1, description="Фильтр по id автора хаба"),
):
    """Свежие публикации хаба ЮФУ."""
    try:
        return await _client().list_publications(
            page=page,
            page_size=page_size,
            type_id=type_id,
            creator_id=creator_id,
        )
    except HubNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except HubUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except HubAuthRequiredError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None


@router.get("/publications/search", response_model=PublicationSearchResponse)
@limiter.limit("20/minute")
async def search_publications(
    request: Request,
    q: str = Query(..., min_length=2, max_length=200, description="Подстрока в названии/авторах/подписи"),
    max_pages: int = Query(10, ge=1, le=50, description="Сколько последних страниц просматривать"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    type_id: int | None = Query(None, ge=1),
    creator_id: int | None = Query(None, ge=1),
):
    """Поиск по публикациям хаба (скан последних страниц, API хаба не умеет search)."""
    try:
        result = await _client().search_publications(
            q,
            max_pages=max_pages,
            type_id=type_id,
            creator_id=creator_id,
        )
    except HubUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except HubAuthRequiredError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    matches = result["matches"]
    return {
        "query": q,
        "matches": matches[offset : offset + limit],
        "total": len(matches),
        "scanned_pages": result["scanned_pages"],
        "truncated": result["truncated"],
    }


@router.get("/publications/types", response_model=PublicationTypesResponse)
@limiter.limit("30/minute")
async def list_publication_types(
    request: Request,
    sample_pages: int = Query(3, ge=1, le=20, description="Сколько последних страниц просматривать"),
):
    """Типы публикаций, встреченные на последних страницах хаба."""
    try:
        return await _client().list_types(sample_pages=sample_pages)
    except HubUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except HubAuthRequiredError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None


@router.get("/publications/{publication_id}", response_model=PublicationOut)
@limiter.limit("60/minute")
async def get_publication(
    request: Request,
    publication_id: int,
    max_scan_pages: int = Query(10, ge=1, le=50, description="Сколько страниц перебрать без токена"),
):
    """Карточка публикации хаба по id."""
    try:
        return await _client().get_publication(publication_id, max_scan_pages=max_scan_pages)
    except HubNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except HubUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except HubAuthRequiredError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
