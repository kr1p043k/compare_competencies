"""Клиент публичного API публикаций хаба ЮФУ (hub.sfedu.ru).

Хаб отдаёт список публикаций без авторизации:
    GET https://hub.sfedu.ru/api/publications/?page=1&page_size=20
Фильтры: ``type=<id>``, ``creator=<id>``. Пагинация — заголовки
``Results-Count``, ``Page-Size``, ``Current-Page``, ``Num-Pages``.

Карточка ``/api/assets/{id}/`` требует токен (``Authorization: Token ...``),
поэтому при его отсутствии запись ищется перебором последних страниц списка.
"""

from __future__ import annotations

import math
import re
from typing import Any

import httpx
import structlog

from src import config

logger = structlog.get_logger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
)
_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100
_YEAR_RE = re.compile(r"(19|20)\d{2}")


class HubPublicationsError(RuntimeError):
    """Базовая ошибка клиента хаба."""


class HubUnavailableError(HubPublicationsError):
    """Хаб недоступен или ответил ошибкой."""


class HubNotFoundError(HubPublicationsError):
    """Публикация (или страница) не найдена."""


class HubAuthRequiredError(HubPublicationsError):
    """Для запроса нужен токен хаба."""


def _first_text(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _absolute_url(base: str, url: Any) -> str | None:
    if not url:
        return None
    text = str(url)
    if text.startswith(("http://", "https://")):
        return text
    return f"{base.rstrip('/')}/{text.lstrip('/')}"


def _author_names(raw: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for creator in raw.get("publication_creators") or []:
        if not isinstance(creator, dict):
            continue
        full = " ".join(
            part for part in (creator.get("last_name"), creator.get("first_name")) if part
        ).strip()
        if full:
            names.append(full)
    if names:
        return names
    return _text_list(payload.get("author"))


def _year(raw: dict[str, Any], payload: dict[str, Any]) -> int | None:
    educational_year = _first_text(payload.get("educational_year"))
    if educational_year:
        match = _YEAR_RE.search(educational_year)
        if match:
            return int(match.group(0))
    created = raw.get("creation_datetime")
    if created:
        match = _YEAR_RE.search(str(created))
        if match:
            return int(match.group(0))
    return None


def _files(base: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in ("fulltext", "review_fulltext"):
        for item in payload.get(field) or []:
            if not isinstance(item, dict):
                continue
            url = _absolute_url(base, item.get("url"))
            if not url:
                continue
            result.append(
                {
                    "url": url,
                    "filename": item.get("filename"),
                    "size": item.get("size"),
                    "content_type": item.get("content_type"),
                }
            )
    return result


def normalize_publication(raw: dict[str, Any], base: str | None = None) -> dict[str, Any]:
    """Приводит запись хаба к плоской схеме нашего API."""
    base = base or config.HUB_API_BASE
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    publication_type = raw.get("type") if isinstance(raw.get("type"), dict) else {}
    creator = raw.get("creator") if isinstance(raw.get("creator"), dict) else {}
    authors = _author_names(raw, payload)
    return {
        "id": int(raw.get("id") or 0),
        "title": _first_text(payload.get("title")) or _first_text(raw.get("signature_string")),
        "authors": authors,
        "type_id": publication_type.get("id"),
        "type_name": publication_type.get("type_name"),
        "signature": raw.get("signature_string"),
        "doi": raw.get("doi"),
        "isbn": raw.get("isbn"),
        "track_id": raw.get("track_id"),
        "created_at": raw.get("creation_datetime"),
        "year": _year(raw, payload),
        "academic_subject": _text_list(payload.get("academic_subject")),
        "education_program": _text_list(payload.get("education_program")),
        "files": _files(base, payload),
        "url": _absolute_url(base, raw.get("url")),
        "creator_username": creator.get("username"),
    }


class HubPublicationsClient:
    """Тонкая обёртка над hub.sfedu.ru/api/publications."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = (base_url or config.HUB_API_BASE).rstrip("/")
        if token is None and config.HUB_API_TOKEN is not None:
            token = config.HUB_API_TOKEN.get_secret_value()
        self.token = token or None
        self.timeout = timeout or config.HUB_API_TIMEOUT_S
        self.transport = transport

    def _headers(self, auth: bool = False) -> dict[str, str]:
        headers = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}
        if auth:
            if not self.token:
                raise HubAuthRequiredError("Для карточки публикации нужен HUB_API_TOKEN")
            token = self.token
            if not token.lower().startswith("token "):
                token = f"Token {token}"
            headers["Authorization"] = token
        return headers

    async def _get(self, path: str, params: dict[str, Any], auth: bool = False) -> httpx.Response:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                transport=self.transport,
                headers=self._headers(auth),
            ) as client:
                return await client.get(f"{self.base_url}{path}", params=params)
        except httpx.HTTPError as exc:
            logger.warning("hub_publications_unreachable", path=path, error=str(exc))
            raise HubUnavailableError("hub.sfedu.ru недоступен") from None

    async def list_publications(
        self,
        page: int = 1,
        page_size: int = _DEFAULT_PAGE_SIZE,
        type_id: int | None = None,
        creator_id: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "page": max(1, page),
            "page_size": max(1, min(page_size, _MAX_PAGE_SIZE)),
        }
        if type_id is not None:
            params["type"] = type_id
        if creator_id is not None:
            params["creator"] = creator_id

        resp = await self._get("/api/publications/", params)
        if resp.status_code == 404:
            raise HubNotFoundError(f"Страница {page} вне диапазона")
        if resp.status_code in (401, 403):
            raise HubAuthRequiredError("hub.sfedu.ru отклонил запрос")
        if resp.status_code != 200:
            logger.warning(
                "hub_publications_error",
                status=resp.status_code,
                body=resp.text[:300],
            )
            raise HubUnavailableError(f"hub.sfedu.ru ответил {resp.status_code}")
        try:
            raw_items = resp.json()
        except ValueError:
            raise HubUnavailableError("hub.sfedu.ru вернул не-JSON") from None
        if not isinstance(raw_items, list):
            raise HubUnavailableError("Неожиданный формат ответа hub.sfedu.ru")

        return {
            "items": [normalize_publication(item, self.base_url) for item in raw_items if isinstance(item, dict)],
            "total": _int_header(resp, "Results-Count", len(raw_items)),
            "page": _int_header(resp, "Current-Page", page),
            "page_size": _int_header(resp, "Page-Size", page_size),
            "num_pages": _int_header(resp, "Num-Pages", 0),
        }

    async def search_publications(
        self,
        query: str,
        max_pages: int = 10,
        type_id: int | None = None,
        creator_id: int | None = None,
    ) -> dict[str, Any]:
        needle = query.casefold().strip()
        matches: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        for page in range(1, max(1, max_pages) + 1):
            try:
                batch = await self.list_publications(
                    page=page,
                    page_size=_MAX_PAGE_SIZE,
                    type_id=type_id,
                    creator_id=creator_id,
                )
            except HubNotFoundError:
                break
            scanned += 1
            items = batch["items"]
            if not items:
                if not batch["num_pages"] or page >= batch["num_pages"]:
                    break
                continue
            for item in items:
                haystack = " ".join(
                    [
                        item.get("title") or "",
                        " ".join(item.get("authors") or []),
                        item.get("signature") or "",
                        " ".join(item.get("academic_subject") or []),
                    ]
                ).casefold()
                if needle in haystack:
                    matches.append(item)
            num_pages = batch["num_pages"]
            if num_pages and page >= num_pages:
                break
            if page == max_pages and num_pages > max_pages:
                truncated = True
        return {"matches": matches, "scanned_pages": scanned, "truncated": truncated}

    async def get_publication(self, publication_id: int, max_scan_pages: int = 10) -> dict[str, Any]:
        if self.token:
            resp = await self._get(f"/api/assets/{publication_id}/", {}, auth=True)
            if resp.status_code == 404:
                raise HubNotFoundError(f"Публикация {publication_id} не найдена")
            if resp.status_code in (401, 403):
                raise HubAuthRequiredError("Токен хаба отклонён")
            if resp.status_code != 200:
                raise HubUnavailableError(f"hub.sfedu.ru ответил {resp.status_code}")
            raw = resp.json()
            if isinstance(raw, dict):
                return normalize_publication(raw, self.base_url)

        for page in range(1, max(1, max_scan_pages) + 1):
            try:
                batch = await self.list_publications(page=page, page_size=_MAX_PAGE_SIZE)
            except HubNotFoundError:
                break
            for item in batch["items"]:
                if item["id"] == publication_id:
                    return item
            if not batch["items"]:
                if not batch["num_pages"] or page >= batch["num_pages"]:
                    break
                continue
            if batch["num_pages"] and page >= batch["num_pages"]:
                break
        raise HubNotFoundError(f"Публикация {publication_id} не найдена")

    async def list_types(self, sample_pages: int = 3) -> dict[str, Any]:
        counts: dict[int, dict[str, Any]] = {}
        scanned = 0
        sample_size = 0
        for page in range(1, max(1, sample_pages) + 1):
            try:
                batch = await self.list_publications(page=page, page_size=_MAX_PAGE_SIZE)
            except HubNotFoundError:
                break
            scanned += 1
            sample_size += len(batch["items"])
            for item in batch["items"]:
                type_id = item.get("type_id")
                if type_id is None:
                    continue
                entry = counts.setdefault(int(type_id), {"id": int(type_id), "name": item.get("type_name"), "count": 0})
                entry["count"] += 1
                if not entry["name"]:
                    entry["name"] = item.get("type_name")
            if not batch["items"]:
                if not batch["num_pages"] or page >= batch["num_pages"]:
                    break
                continue
            if batch["num_pages"] and page >= batch["num_pages"]:
                break
        types = sorted(counts.values(), key=lambda entry: (-entry["count"], entry["id"]))
        return {"types": types, "scanned_pages": scanned, "sample_size": sample_size}


def _int_header(resp: httpx.Response, name: str, default: int) -> int:
    value = resp.headers.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def pages_for(total: int, page_size: int) -> int:
    if page_size <= 0:
        return 0
    return math.ceil(total / page_size)


_client: HubPublicationsClient | None = None


def get_client() -> HubPublicationsClient:
    global _client
    if _client is None:
        _client = HubPublicationsClient()
    return _client


def reset_client() -> None:
    global _client
    _client = None
