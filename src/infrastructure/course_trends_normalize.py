"""Нормализация ответов Course Trends API в список статей.

Сервис ЮФУ не публикует схемы ответов (`"schema": {}` в OpenAPI), поэтому
структура полей заранее неизвестна и может меняться. Вместо жёсткого разбора
здесь работает поиск по смыслу: обходим JSON и собираем объекты, похожие на
публикацию (есть заголовок + хотя бы один признак статьи: авторы, год,
аннотация, DOI, ссылка).

Такой адаптер переживает переименование полей на стороне сервиса: `title` ->
`name` -> `heading` не ломает выдачу, как это случилось бы с фиксированной
моделью.
"""

from __future__ import annotations

import re
from typing import Any

TITLE_KEYS = ("title", "name", "heading", "название", "заголовок", "paper_title")
ABSTRACT_KEYS = ("abstract", "summary", "description", "аннотация", "abstract_text")
YEAR_KEYS = ("year", "publication_year", "published", "date", "год")
AUTHORS_KEYS = ("authors", "author", "авторы", "author_names")
URL_KEYS = ("url", "link", "doi_url", "landing_page_url", "href", "ссылка")
DOI_KEYS = ("doi", "DOI")
CITATIONS_KEYS = ("citations", "cited_by_count", "citation_count", "цитирования")
SOURCE_KEYS = ("source", "journal", "venue", "publisher", "издание")
COURSE_KEYS = ("course", "course_name", "discipline", "disciplines", "курс", "дисциплина")

MAX_DEPTH = 8
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def _first_str(payload: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def _as_year(payload: dict) -> int | None:
    raw = _first_str(payload, YEAR_KEYS)
    if not raw:
        return None
    match = re.search(r"(19|20)\d{2}", raw)
    return int(match.group(0)) if match else None


def _as_authors(payload: dict) -> list[str]:
    for key in AUTHORS_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
                elif isinstance(item, dict):
                    name = _first_str(item, ("name", "display_name", "full_name", "author"))
                    if name:
                        out.append(name)
            if out:
                return out
        if isinstance(value, str) and value.strip():
            return [p.strip() for p in re.split(r"[;,]", value) if p.strip()]
    return []


def _as_int(payload: dict, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _as_url(payload: dict) -> str:
    url = _first_str(payload, URL_KEYS)
    if not url:
        # Ссылка часто лежит во вложенном объекте (OpenAlex-стиль:
        # primary_location.landing_page_url, best_oa_location.pdf_url).
        for container in ("primary_location", "best_oa_location", "location",
                          "source", "links"):
            nested = payload.get(container)
            if isinstance(nested, dict):
                url = _first_str(nested, URL_KEYS + ("pdf_url", "html_url"))
                if url:
                    break
    if url:
        return url
    doi = _first_str(payload, DOI_KEYS).strip()
    if doi and _DOI_RE.match(doi):
        return f"https://doi.org/{doi}"
    if doi.startswith("10."):
        return f"https://doi.org/{doi}"
    return ""


def _course_hint(payload: dict) -> str:
    for key in COURSE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict):
                name = _first_str(first, ("name", "title", "course"))
                if name:
                    return name
    return ""


def looks_like_article(payload: dict) -> bool:
    """Заголовок обязателен; плюс хотя бы один признак публикации."""
    title = _first_str(payload, TITLE_KEYS)
    if not title or len(title) < 3:
        return False
    signals = (
        _as_authors(payload),
        _as_year(payload),
        _first_str(payload, ABSTRACT_KEYS),
        _first_str(payload, DOI_KEYS),
        _as_url(payload),
        _as_int(payload, CITATIONS_KEYS),
    )
    return any(signals)


def normalize_article(payload: dict, *, course: str | None = None) -> dict[str, Any]:
    """Канонический вид статьи, независимый от исходных имён полей."""
    doi = _first_str(payload, DOI_KEYS) or None
    if doi and not _DOI_RE.match(doi):
        doi = None
    return {
        "title": _first_str(payload, TITLE_KEYS),
        "abstract": _first_str(payload, ABSTRACT_KEYS),
        "year": _as_year(payload),
        "authors": _as_authors(payload),
        "url": _as_url(payload),
        "doi": doi,
        "citations": _as_int(payload, CITATIONS_KEYS),
        "source": _first_str(payload, SOURCE_KEYS),
        "course": course or _course_hint(payload),
    }


def extract_articles(payload: Any, *, course: str | None = None) -> list[dict[str, Any]]:
    """Рекурсивно собирает статьи из произвольного JSON-ответа сервиса.

    Дедупликация: приоритет у записи с DOI, иначе по нормализованному
    заголовку. Порядок сохраняется (сервис уже отсортировал по релевантности).
    """
    found: list[dict[str, Any]] = []

    def walk(node: Any, current_course: str | None, depth: int) -> None:
        if depth > MAX_DEPTH:
            return
        if isinstance(node, dict):
            hint = _course_hint(node) or current_course
            if looks_like_article(node):
                found.append(normalize_article(node, course=hint))
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value, hint, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, current_course, depth + 1)

    walk(payload, course, 0)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for article in found:
        key = (article["doi"] or "").lower() or re.sub(
            r"\s+", " ", article["title"].lower()
        ).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(article)
    return unique
