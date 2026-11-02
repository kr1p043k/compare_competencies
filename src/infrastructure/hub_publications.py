"""Клиент публичного API публикаций хаба ЮФУ (hub.sfedu.ru).

Хаб отдаёт список публикаций без авторизации:
    GET https://hub.sfedu.ru/api/publications/?page=1&page_size=20
Фильтры: ``type=<id>``, ``creator=<id>``. Пагинация — заголовки
``Results-Count``, ``Page-Size``, ``Current-Page``, ``Num-Pages``.

Карточка ``/api/assets/{id}/`` требует токен (``Authorization: Token ...``),
поэтому без токена (или если хаб его отклонил) запись ищется перебором
страниц с начала списка — там самые свежие публикации.
Серверного поиска нет: ``search`` сканирует последние страницы и матчит локально.
"""

from __future__ import annotations

import asyncio
import re
import threading
from typing import Any
from urllib.parse import urlsplit

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
_MAX_SCAN_PAGES = 10
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_SCAN_CONCURRENCY = 5
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_ALLOWED_FILE_HOSTS = {"hub.sfedu.ru", "hub.lib.sfedu.ru"}


class HubPublicationsError(RuntimeError):
    """Базовая ошибка клиента хаба."""


class HubUnavailableError(HubPublicationsError):
    """Хаб недоступен или ответил ошибкой."""


class HubNotFoundError(HubPublicationsError):
    """Публикация (или страница) не найдена."""


class HubAuthRequiredError(HubPublicationsError):
    """Для запроса нужен токен хаба или хаб отклонил его."""


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
    """Абсолютный URL только для доверенных хостов хаба, иначе только относительный путь."""
    if not url:
        return None
    text = str(url).strip()
    if text.startswith(("http://", "https://")):
        host = urlsplit(text).hostname or ""
        if host not in _ALLOWED_FILE_HOSTS:
            return None
        return text
    if not text.startswith("/"):
        return None
    return f"{base.rstrip('/')}/{text.lstrip('/')}"


def _payload_of(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("payload")
    return payload if isinstance(payload, dict) else {}


def _dict_of(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


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
    return names or _text_list(payload.get("author"))


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


def _publication_id(raw: dict[str, Any]) -> int:
    value = raw.get("id")
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def normalize_publication(raw: dict[str, Any], base: str | None = None) -> dict[str, Any]:
    """Приводит запись хаба к плоской схеме нашего API."""
    base = base or config.HUB_API_BASE
    payload = _payload_of(raw)
    publication_type = _dict_of(raw, "type")
    creator = _dict_of(raw, "creator")
    return {
        "id": _publication_id(raw),
        "title": _first_text(payload.get("title")) or _first_text(raw.get("signature_string")),
        "authors": _author_names(raw, payload),
        "type_id": publication_type.get("id"),
        "type_name": publication_type.get("type_name"),
        "signature": raw.get("signature_string"),
        "doi": _first_text(raw.get("doi")),
        "isbn": _first_text(raw.get("isbn")),
        "track_id": _first_text(raw.get("track_id")),
        "created_at": raw.get("creation_datetime"),
        "year": _year(raw, payload),
        "academic_subject": _text_list(payload.get("academic_subject")),
        "education_program": _text_list(payload.get("education_program")),
        "files": _files(base, payload),
        "url": _absolute_url(base, raw.get("url")),
        "creator_username": creator.get("username"),
    }


def _matches(raw: dict[str, Any], needle: str) -> bool:
    payload = _payload_of(raw)
    haystack = " ".join(
        [
            _first_text(payload.get("title")) or "",
            " ".join(_text_list(payload.get("author"))),
            " ".join(_text_list(payload.get("academic_subject"))),
            str(raw.get("signature_string") or ""),
        ]
    )
    return needle in haystack.casefold()


def _int_header(resp: httpx.Response, name: str, default: int) -> int:
    value = resp.headers.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _exhausted(meta: dict[str, Any], page: int) -> bool:
    num_pages = meta["num_pages"]
    return bool(num_pages) and page >= num_pages


class HubPublicationsClient:
    """Тонкая обёртка над hub.sfedu.ru/api/publications с общим пулом соединений."""

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
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    def _headers(self, auth: bool = False) -> dict[str, str]:
        headers = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}
        if auth and self.token:
            token = self.token
            if not token.lower().startswith("token "):
                token = f"Token {token}"
            headers["Authorization"] = token
        return headers

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                transport=self._transport,
                headers={"User-Agent": _BROWSER_UA, "Accept": "application/json"},
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
            self._closed = False
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._closed:
            await self._client.aclose()
            self._closed = True

    async def _get(self, path: str, params: dict[str, Any], auth: bool = False) -> httpx.Response:
        headers = self._headers(auth)
        url = f"{self.base_url}{path}"
        try:
            async with self._http().stream("GET", url, params=params, headers=headers) as response:
                declared = response.headers.get("Content-Length", "")
                if declared.isdigit() and int(declared) > _MAX_RESPONSE_BYTES:
                    logger.warning("hub_publications_too_large", path=path, declared=int(declared))
                    raise HubUnavailableError("hub.sfedu.ru вернул слишком большой ответ")
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_RESPONSE_BYTES:
                        logger.warning("hub_publications_too_large", path=path, bytes=total)
                        raise HubUnavailableError("hub.sfedu.ru вернул слишком большой ответ")
                    chunks.append(chunk)
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=b"".join(chunks),
                    request=response.request,
                )
        except httpx.HTTPError as exc:
            logger.warning("hub_publications_unreachable", path=path, error=type(exc).__name__)
            raise HubUnavailableError("hub.sfedu.ru недоступен") from None

    @staticmethod
    def _decode(resp: httpx.Response) -> list[dict[str, Any]]:
        try:
            data = resp.json()
        except ValueError:
            raise HubUnavailableError("hub.sfedu.ru вернул не-JSON") from None
        if not isinstance(data, list):
            raise HubUnavailableError("Неожиданный формат ответа hub.sfedu.ru")
        return [item for item in data if isinstance(item, dict)]

    def _check_status(
        self,
        resp: httpx.Response,
        page: int | None = None,
        not_found_msg: str | None = None,
    ) -> None:
        if resp.status_code == 404:
            raise HubNotFoundError(not_found_msg or f"Страница {page} вне диапазона")
        if resp.status_code in (401, 403):
            raise HubAuthRequiredError("hub.sfedu.ru отклонил запрос")
        if resp.status_code != 200:
            logger.warning("hub_publications_error", status=resp.status_code, bytes=len(resp.content))
            raise HubUnavailableError(f"hub.sfedu.ru ответил {resp.status_code}")

    async def _fetch_raw(
        self,
        page: int,
        page_size: int,
        type_id: int | None = None,
        creator_id: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        params: dict[str, Any] = {
            "page": max(1, page),
            "page_size": max(1, min(page_size, _MAX_PAGE_SIZE)),
        }
        if type_id is not None:
            params["type"] = type_id
        if creator_id is not None:
            params["creator"] = creator_id

        resp = await self._get("/api/publications/", params)
        self._check_status(resp, page)
        meta = {
            "total": _int_header(resp, "Results-Count", 0),
            "page": _int_header(resp, "Current-Page", page),
            "page_size": _int_header(resp, "Page-Size", params["page_size"]),
            "num_pages": _int_header(resp, "Num-Pages", 0),
        }
        return meta, self._decode(resp)

    async def _scan(
        self,
        pages: int,
        type_id: int | None = None,
        creator_id: int | None = None,
    ) -> tuple[list[list[dict[str, Any]]], dict[str, Any] | None, int]:
        """Читает страницы 1..N (первая — последовательно, чтобы узнать Num-Pages).

        Возвращает (страницы, meta последней прочитанной, число прочитанных).
        """
        try:
            first_meta, first_items = await self._fetch_raw(1, _MAX_PAGE_SIZE, type_id, creator_id)
        except HubNotFoundError:
            return [], None, 0

        pages_data: list[list[dict[str, Any]]] = [first_items]
        last_meta = first_meta
        total_pages = first_meta["num_pages"]
        target = pages if not total_pages else min(pages, total_pages)
        if not first_items or _exhausted(first_meta, 1) or target <= 1:
            return pages_data, last_meta, len(pages_data)

        semaphore = asyncio.Semaphore(_SCAN_CONCURRENCY)

        async def fetch(page: int) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
            async with semaphore:
                try:
                    return await self._fetch_raw(page, _MAX_PAGE_SIZE, type_id, creator_id)
                except HubNotFoundError:
                    return None

        results = await asyncio.gather(*(fetch(page) for page in range(2, target + 1)))
        for page_result in results:
            if page_result is None:
                break
            meta, items = page_result
            last_meta = meta
            pages_data.append(items)
            if not items or _exhausted(meta, meta["page"]):
                break
        return pages_data, last_meta, len(pages_data)

    async def list_publications(
        self,
        page: int = 1,
        page_size: int = _DEFAULT_PAGE_SIZE,
        type_id: int | None = None,
        creator_id: int | None = None,
    ) -> dict[str, Any]:
        meta, items = await self._fetch_raw(page, page_size, type_id, creator_id)
        return {
            "items": [normalize_publication(item, self.base_url) for item in items],
            "total": meta["total"],
            "page": meta["page"],
            "page_size": meta["page_size"],
            "num_pages": meta["num_pages"],
        }

    async def search_publications(
        self,
        query: str,
        max_pages: int = 10,
        type_id: int | None = None,
        creator_id: int | None = None,
    ) -> dict[str, Any]:
        needle = query.casefold().strip()
        pages = max(1, min(max_pages, _MAX_SCAN_PAGES))
        pages_data, last_meta, scanned = await self._scan(pages, type_id, creator_id)

        matches: list[dict[str, Any]] = []
        for items in pages_data:
            for raw in items:
                if _matches(raw, needle):
                    matches.append(normalize_publication(raw, self.base_url))

        num_pages = last_meta["num_pages"] if last_meta else 0
        truncated = scanned >= pages and (num_pages == 0 or num_pages > scanned)
        return {
            "matches": matches,
            "scanned_pages": scanned,
            "truncated": truncated,
            "total_scanned": sum(len(items) for items in pages_data),
        }

    async def get_publication(
        self,
        publication_id: int,
        max_scan_pages: int = 10,
        use_token: bool = True,
    ) -> dict[str, Any]:
        """Карточка по id. ``use_token=False`` — только сканирование списка.

        Токен хаба — общий ресурс организации, поэтому анонимные запросы
        в него не ходят.
        """
        if self.token and use_token:
            resp = await self._get(f"/api/assets/{publication_id}/", {}, auth=True)
            if resp.status_code in (401, 403):
                logger.warning("hub_publications_token_rejected", status=resp.status_code)
                self.token = None
            else:
                self._check_status(resp, not_found_msg=f"Публикация {publication_id} не найдена")
                try:
                    raw = resp.json()
                except ValueError:
                    raise HubUnavailableError("hub.sfedu.ru вернул не-JSON") from None
                if isinstance(raw, dict) and _publication_id(raw) == publication_id:
                    return normalize_publication(raw, self.base_url)

        pages = max(1, min(max_scan_pages, _MAX_SCAN_PAGES))
        pages_data, _, _ = await self._scan(pages)
        for items in pages_data:
            for raw in items:
                if _publication_id(raw) == publication_id:
                    return normalize_publication(raw, self.base_url)
        raise HubNotFoundError(f"Публикация {publication_id} не найдена")

    async def list_types(self, sample_pages: int = 3) -> dict[str, Any]:
        pages = max(1, min(sample_pages, _MAX_SCAN_PAGES))
        pages_data, _, scanned = await self._scan(pages)

        counts: dict[int, dict[str, Any]] = {}
        sample_size = 0
        for items in pages_data:
            sample_size += len(items)
            for raw in items:
                publication_type = _dict_of(raw, "type")
                type_id = publication_type.get("id")
                if type_id is None:
                    continue
                try:
                    type_id_int = int(type_id)
                except (TypeError, ValueError):
                    continue
                entry = counts.setdefault(
                    type_id_int,
                    {"id": type_id_int, "name": publication_type.get("type_name"), "count": 0},
                )
                entry["count"] += 1
                if not entry["name"]:
                    entry["name"] = publication_type.get("type_name")
        types = sorted(counts.values(), key=lambda entry: (-entry["count"], entry["id"]))
        return {"types": types, "scanned_pages": scanned, "sample_size": sample_size}


_client: HubPublicationsClient | None = None
_client_lock = threading.Lock()


def get_client() -> HubPublicationsClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = HubPublicationsClient()
    return _client


async def close_client() -> None:
    """Закрывает общий клиент (вызывается на shutdown приложения)."""
    global _client
    with _client_lock:
        client = _client
        _client = None
    if client is not None:
        await client.aclose()
