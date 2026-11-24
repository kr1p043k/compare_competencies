"""Загрузка публикаций hub.sfedu.ru в RAG-корпус.

Два режима:
  * ``mode="list"`` — последние N страниц каталога (дешёво, широкий охват);
  * ``mode="search"`` — deep search по списку запросов (профессии/навыки),
    точнее по теме, но дороже: каждый запрос сканирует несколько страниц.

Сеть изолирована здесь: ``publications_to_documents`` — чистая функция
(тестируется без httpx), ``load_publications_documents`` делает запросы.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import structlog

from src.rag.chunking.publication import chunk_publication
from src.rag.types import Document

logger = structlog.get_logger(__name__)

#: Сколько страниц листать в режиме list по умолчанию.
DEFAULT_LIST_PAGES = 10
#: Потолок запросов в режиме search: hub-API не любит шквал.
DEFAULT_MAX_QUERIES = 20


def publications_to_documents(
    publications: Iterable[dict[str, Any]],
    *,
    max_tokens: int = 700,
    min_tokens: int = 40,
) -> list[Document]:
    """Нормализованные публикации -> документы (дедуп по id, порядок сохранён)."""
    documents: list[Document] = []
    seen: set[str] = set()
    for publication in publications:
        doc = chunk_publication(publication, max_tokens=max_tokens, min_tokens=min_tokens)
        if doc is None:
            continue
        if doc.external_id in seen:
            continue
        seen.add(doc.external_id)
        documents.append(doc)
    return documents


async def _fetch_by_list(client: Any, pages: int) -> list[dict[str, Any]]:
    """Листать каталог: страницы 1..pages."""
    collected: list[dict[str, Any]] = []
    for page in range(1, max(1, pages) + 1):
        try:
            payload = await client.list_publications(page=page)
        except Exception as exc:
            logger.warning("rag_publications_list_failed", page=page, error=str(exc))
            break
        items = payload.get("items") or []
        if not items:
            break
        collected.extend(items)
        num_pages = int(payload.get("num_pages") or 0)
        if num_pages and page >= num_pages:
            break
    return collected


async def _fetch_by_search(
    client: Any, queries: Iterable[str], max_pages: int
) -> list[dict[str, Any]]:
    """Deep search по запросам; ошибки одного запроса не роняют остальные."""
    collected: list[dict[str, Any]] = []
    for query in queries:
        text = str(query or "").strip()
        if not text:
            continue
        try:
            payload = await client.search_publications(text, max_pages=max_pages)
        except Exception as exc:
            logger.warning("rag_publications_search_failed", query=text, error=str(exc))
            continue
        matches = payload.get("matches") or []
        logger.info("rag_publications_search", query=text, matches=len(matches))
        collected.extend(matches)
    return collected


async def load_publications_documents(
    *,
    mode: str = "list",
    pages: int = DEFAULT_LIST_PAGES,
    queries: Iterable[str] | None = None,
    search_pages: int = 3,
    max_tokens: int = 700,
    min_tokens: int = 40,
    client: Any = None,
) -> list[Document]:
    """Скачать публикации и превратить их в документы. Сетевые ошибки — лог, не падение."""
    owns_client = client is None
    if client is None:
        from src.infrastructure.hub_publications import get_client

        client = get_client()

    try:
        if mode == "search":
            query_list = [str(q) for q in (queries or []) if str(q).strip()][:DEFAULT_MAX_QUERIES]
            if not query_list:
                logger.warning("rag_publications_no_queries")
                return []
            raw = await _fetch_by_search(client, query_list, max(1, search_pages))
        else:
            raw = await _fetch_by_list(client, pages)
    finally:
        if owns_client:
            try:
                await client.aclose()
            except Exception:
                logger.info("rag_publications_client_close_failed")

    documents = publications_to_documents(raw, max_tokens=max_tokens, min_tokens=min_tokens)
    logger.info(
        "rag_publications_loaded",
        mode=mode,
        fetched=len(raw),
        documents=len(documents),
    )
    return documents
