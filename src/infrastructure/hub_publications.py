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
import contextlib
import copy
import html
import re
import threading
import time
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
# Сколько страниц максимум просматривает поиск. Корпус - 133 902 записи
# (26 781 страница по 5), полнотекстового поиска у hub API нет: искать можно
# только последовательным чтением. 50 страниц по 100 записей = 5 000 записей
# (~4%), и это честный предел интерактивного запроса. Сколько реально
# просмотрено, ответ сообщает в total_scanned/truncated.
_MAX_SCAN_PAGES = 10
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_SCAN_CONCURRENCY = 5
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_ALLOWED_FILE_HOSTS = {"hub.sfedu.ru", "hub.lib.sfedu.ru"}

# id типов публикаций, реально встречающихся в хабе (проверено обходом 1..70,
# 2026-11-03; сумма счётчиков совпала с Results-Count = 133 902).
# Нужны как стартовый набор для точных счётчиков: у хаба нет эндпоинта,
# который перечисляет только используемые типы.
_TYPE_SEED_IDS: tuple[int, ...] = (34, 35, 36, 37, 38, 39, 45, 46, 48, 49, 50, 51, 56, 59)
_TYPE_COUNT_TTL_S = 24 * 60 * 60
_TYPE_PROBE_CONCURRENCY = 8
_NEGATIVE_CACHE_TTL_S = 300.0


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


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str | None:
    """Снимает HTML-сущности и теги: хаб отдаёт ``&quot;`` и разметку в abstract."""
    text = _first_text(value)
    if not text:
        return None
    text = html.unescape(_HTML_TAG_RE.sub(" ", text))
    return _WS_RE.sub(" ", text).strip() or None


def _int_or_none(value: Any) -> int | None:
    text = _first_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def web_url(publication_id: int, base: str | None = None) -> str:
    """Человеческая страница записи в хабе (требует входа в hub.sfedu.ru)."""
    base = (base or config.HUB_API_BASE).rstrip("/")
    return f"{base}/dashboard/asset/{publication_id}/"


def normalize_publication(raw: dict[str, Any], base: str | None = None) -> dict[str, Any]:
    """Приводит запись хаба к плоской схеме нашего API."""
    base = base or config.HUB_API_BASE
    payload = _payload_of(raw)
    publication_type = _dict_of(raw, "type")
    creator = _dict_of(raw, "creator")
    publication_id = _publication_id(raw)
    return {
        "id": publication_id,
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
        "abstract": clean_text(payload.get("abstract")),
        "keywords": _text_list(payload.get("keyword")),
        "journal": _first_text(payload.get("periodical_title")),
        "publisher": _first_text(payload.get("publisher")),
        "language": _first_text(payload.get("language")),
        "publishing_year": _int_or_none(payload.get("publishing_year")),
        "pages": _first_text(payload.get("page_range")),
        "files": _files(base, payload),
        "url": _absolute_url(base, raw.get("url")),
        "web_url": web_url(publication_id, base) if publication_id else None,
        "creator_username": creator.get("username"),
    }


def _matches(raw: dict[str, Any], needle: str) -> bool:
    payload = _payload_of(raw)
    haystack = " ".join(
        [
            _first_text(payload.get("title")) or "",
            " ".join(_text_list(payload.get("author"))),
            " ".join(_text_list(payload.get("academic_subject"))),
            " ".join(_text_list(payload.get("keyword"))),
            _first_text(payload.get("periodical_title")) or "",
            _first_text(payload.get("abstract")) or "",
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
    """Тонкая обёртка над hub.sfedu.ru/api/publications с общим пулом соединений.

    Список и типы кэшируются на ``cache_ttl`` секунд: хаб большой и общий,
    а страницы списка меняются медленно.
    """

    _MAX_CACHE_ENTRIES = 256

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        cache_ttl: float | None = None,
    ):
        self.base_url = (base_url or config.HUB_API_BASE).rstrip("/")
        if token is None and config.HUB_API_TOKEN is not None:
            token = config.HUB_API_TOKEN.get_secret_value()
        self.token = token or None
        self.timeout = timeout or config.HUB_API_TIMEOUT_S
        self.cache_ttl = config.HUB_CACHE_TTL_S if cache_ttl is None else cache_ttl
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._loop: Any = None
        self._closed = False
        self._cache: dict[tuple[Any, ...], tuple[float, dict[str, Any], list[dict[str, Any]]]] = {}
        self._types_cache: tuple[float, dict[str, Any]] | None = None
        self._types_lock = asyncio.Lock()
        self._missing: dict[int, float] = {}

    def _cache_get(self, key: tuple[Any, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        if self.cache_ttl <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, meta, items = entry
        if expires_at <= time.monotonic():
            self._cache.pop(key, None)
            return None
        return meta, items

    def _cache_put(self, key: tuple[Any, ...], meta: dict[str, Any], items: list[dict[str, Any]]) -> None:
        if self.cache_ttl <= 0:
            return
        if len(self._cache) >= self._MAX_CACHE_ENTRIES:
            oldest = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
            self._cache.pop(oldest, None)
        self._cache[key] = (time.monotonic() + self.cache_ttl, meta, items)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._types_cache = None
        self._missing.clear()

    def _missing_get(self, publication_id: int) -> bool:
        """Негативный кэш: id, который хаб только что не нашёл, не сканируем снова."""
        if self.cache_ttl <= 0:
            return False
        expires_at = self._missing.get(publication_id)
        if expires_at is None:
            return False
        if expires_at <= time.monotonic():
            self._missing.pop(publication_id, None)
            return False
        return True

    def _missing_put(self, publication_id: int) -> None:
        if self.cache_ttl <= 0:
            return
        if len(self._missing) >= self._MAX_CACHE_ENTRIES:
            self._missing.clear()
        self._missing[publication_id] = time.monotonic() + _NEGATIVE_CACHE_TTL_S

    def _headers(self, auth: bool = False) -> dict[str, str]:
        headers = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}
        if auth and self.token:
            token = self.token
            if not token.lower().startswith("token "):
                token = f"Token {token}"
            headers["Authorization"] = token
        return headers

    def _http(self) -> httpx.AsyncClient:
        """Общий клиент, привязанный к текущему event loop.

        httpx.AsyncClient нельзя переиспользовать между разными loop'ами
        (TestClient, перезапуск lifespan): пул соединений остаётся в закрытом
        loop и падает на ``aclose``. При смене loop создаём новый клиент.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._client is not None and self._loop is not loop:
            self._client = None
            self._closed = True
        if self._client is None or self._closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                transport=self._transport,
                headers={"User-Agent": _BROWSER_UA, "Accept": "application/json"},
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
            self._loop = loop
            self._closed = False
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._closed:
            # Пул мог принадлежать уже закрытому loop — просто отпускаем ссылку.
            with contextlib.suppress(RuntimeError):
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

        cache_key = ("list", params["page"], params["page_size"], type_id, creator_id)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        resp = await self._get("/api/publications/", params)
        self._check_status(resp, page)
        meta = {
            "total": _int_header(resp, "Results-Count", 0),
            # Отличаем «0 записей» от «хаб не прислал заголовок»: для точных
            # счётчиков типов это разница между нулём и неизвестным.
            "total_known": "Results-Count" in resp.headers,
            "page": _int_header(resp, "Current-Page", page),
            "page_size": _int_header(resp, "Page-Size", params["page_size"]),
            "num_pages": _int_header(resp, "Num-Pages", 0),
        }
        items = self._decode(resp)
        self._cache_put(cache_key, meta, items)
        return meta, items

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
        if self._missing_get(publication_id):
            raise HubNotFoundError(f"Публикация {publication_id} не найдена")

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
        pages_data, last_meta, scanned = await self._scan(pages)
        for items in pages_data:
            for raw in items:
                if _publication_id(raw) == publication_id:
                    return normalize_publication(raw, self.base_url)
        # Кэшируем «не найдено» только если просмотрели весь доступный диапазон:
        # при обрезанном скане запись может лежать глубже, и 300 секунд лжи
        # дороже, чем повторная проверка.
        num_pages = last_meta["num_pages"] if last_meta else 0
        truncated = scanned >= pages and (num_pages == 0 or num_pages > scanned)
        if not truncated:
            self._missing_put(publication_id)
        raise HubNotFoundError(f"Публикация {publication_id} не найдена")

    async def _asset_type_names(self) -> dict[int, str]:
        """Справочник типов, который отдаёт сам хаб (частичный, без «Научная статья»)."""
        try:
            resp = await self._get("/api/asset_types/", {})
        except HubUnavailableError:
            return {}
        if resp.status_code != 200:
            return {}
        try:
            data = resp.json()
        except ValueError:
            return {}
        if not isinstance(data, list):
            return {}
        names: dict[int, str] = {}
        for item in data:
            if not isinstance(item, dict) or item.get("id") is None:
                continue
            try:
                names[int(item["id"])] = str(item.get("type_name") or "")
            except (TypeError, ValueError):
                continue
        return names

    async def _count_for_type(self, type_id: int) -> tuple[int | None, str | None]:
        """Точное число записей типа: ``page_size=1`` + заголовок ``Results-Count``.

        ``None`` вместо числа, если хаб не прислал заголовок: молчаливый ноль
        выкинул бы тип из выдачи, а ``exact`` остался бы ``True``.
        """
        try:
            meta, items = await self._fetch_raw(1, 1, type_id, None)
        except HubNotFoundError:
            return 0, None
        name: str | None = None
        if items:
            publication_type = _dict_of(items[0], "type")
            if _publication_id({"id": publication_type.get("id")}) == type_id:
                name = publication_type.get("type_name")
        if not meta.get("total_known", False):
            return None, name
        return meta["total"], name

    async def list_types(self, sample_pages: int = 3, refresh: bool = False) -> dict[str, Any]:
        """Точные счётчики по типам публикаций.

        Списку типов хаба доверять нельзя: ``/api/asset_types/`` перечисляет
        типы файлов (20 записей, «Научной статьи» там нет), а параметр ``type=``
        принимает только один id. Поэтому опрашиваем известные id по одному с
        ``page_size=1`` и берём ``Results-Count``: 14 параллельных запросов
        дают точные числа вместо выборки по последним страницам.
        """
        now = time.monotonic()
        if not refresh and self._types_cache is not None and self._types_cache[0] > now:
            return copy.deepcopy(self._types_cache[1])

        async with self._types_lock:
            now = time.monotonic()
            if not refresh and self._types_cache is not None and self._types_cache[0] > now:
                return copy.deepcopy(self._types_cache[1])

            names = await self._asset_type_names()
            type_ids = sorted(set(_TYPE_SEED_IDS) | set(names))
            semaphore = asyncio.Semaphore(_TYPE_PROBE_CONCURRENCY)

            async def probe(type_id: int) -> tuple[int, int | None, str | None]:
                async with semaphore:
                    try:
                        count, name = await self._count_for_type(type_id)
                    except HubUnavailableError:
                        return type_id, None, None
                    return type_id, count, name

            results = await asyncio.gather(*(probe(type_id) for type_id in type_ids))
            types: list[dict[str, Any]] = []
            failed = 0
            for type_id, count, name in results:
                if count is None:
                    failed += 1
                    continue
                if count == 0:
                    continue
                types.append({"id": type_id, "name": name or names.get(type_id), "count": count})
            types.sort(key=lambda entry: (-entry["count"], entry["id"]))
            total = sum(entry["count"] for entry in types)
            payload: dict[str, Any] = {
                "types": types,
                "total": total,
                "exact": failed == 0,
                "failed_probes": failed,
            }
            if failed == 0:
                self._types_cache = (time.monotonic() + _TYPE_COUNT_TTL_S, payload)
            return copy.deepcopy(payload)


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
