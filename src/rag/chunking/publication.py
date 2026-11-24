"""Чанкер публикаций hub.sfedu.ru (научные работы сотрудников).

Публикации — второй рыночный/научный источник после вакансий: по ним видно,
какие навыки реально используются в исследованиях кафедры, и это опора для
ответов «чем подкреплена дисциплина».

Особенности:
  * аннотация и ключевые слова — отдельные чанки (они короткие и плотные);
  * метаданные (авторы, журнал, год, DOI) идут в ``meta`` и в заголовок
    секции, а не в текст, чтобы не засорять эмбеддинг;
  * публикация без аннотации всё равно индексируется: заголовок + ключевые
    слова + тип — этого достаточно для поиска «кто занимается X», а
    ``meta.partial`` честно говорит, что текста нет.
"""

from __future__ import annotations

from typing import Any

from src.rag.chunking.base import build_chunks
from src.rag.types import Document, normalize_entity_key

#: Минимум символов, ниже — карточка без смысла (пустой заголовок и т.п.).
MIN_DOC_CHARS = 40


def _str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _year_of(publication: dict[str, Any]) -> int | None:
    for key in ("year", "publishing_year"):
        value = publication.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def chunk_publication(
    publication: dict[str, Any],
    *,
    max_tokens: int = 700,
    min_tokens: int = 40,
) -> Document | None:
    """Построить Document из нормализованной публикации (hub_publications)."""
    if not isinstance(publication, dict):
        return None
    publication_id = publication.get("id")
    if publication_id is None:
        return None
    title = str(publication.get("title") or "").strip()
    if not title:
        return None

    abstract = str(publication.get("abstract") or "").strip()
    keywords = _str_list(publication.get("keywords"))
    subjects = _str_list(publication.get("academic_subject"))
    programs = _str_list(publication.get("education_program"))
    authors = _str_list(publication.get("authors"))
    type_name = str(publication.get("type_name") or "").strip()
    journal = str(publication.get("journal") or "").strip()

    sections: list[tuple[str, str]] = []
    if abstract:
        sections.append((f"{title} / Аннотация", abstract))
    if keywords:
        # Ключевые слова повторяют название: чанк самодостаточен для FTS и
        # не проваливается под min_tokens (3-4 слова — это ~12 токенов).
        sections.append((f"{title} / Ключевые слова", f"{title}. Ключевые слова: {', '.join(keywords)}."))
    if subjects:
        sections.append(
            (f"{title} / Предметная область", f"Предметная область: {', '.join(subjects)}.")
        )
    if programs:
        sections.append(
            (
                f"{title} / Образовательные программы",
                f"Образовательные программы: {', '.join(programs)}.",
            )
        )

    if not abstract:
        # Текста нет: карточка из метаданных, чтобы публикацию можно было найти
        # по названию/автору/типу (``meta.partial`` честно об этом сообщает).
        parts = [f"Публикация: {title}."]
        if type_name:
            parts.append(f"Тип: {type_name}.")
        if authors:
            parts.append("Авторы: " + ", ".join(authors) + ".")
        if journal:
            parts.append(f"Издание: {journal}.")
        sections.append((f"{title} / Карточка", " ".join(parts)))

    total = sum(len(body) for _, body in sections)
    if total < MIN_DOC_CHARS:
        return None

    entities: list[tuple[str, str]] = []
    for keyword in keywords:
        key = normalize_entity_key(keyword)
        if key:
            entities.append(("skill", key))
    for author in authors[:20]:
        key = normalize_entity_key(author)
        if key:
            entities.append(("author", key))

    meta = {
        "kind": "publication",
        "publication_id": publication_id,
        "type_name": type_name,
        "journal": journal,
        "doi": publication.get("doi"),
        "authors": authors[:20],
        "keywords": keywords[:40],
        "partial": not abstract,
    }
    chunks = build_chunks(
        sections,
        max_tokens=max_tokens,
        min_tokens=min(min_tokens, 20),
        entities=entities,
        meta=meta,
    )
    if not chunks:
        return None

    url = publication.get("web_url") or publication.get("url")
    return Document(
        source="publications",
        external_id=str(publication_id),
        title=title,
        chunks=chunks,
        url=str(url) if url else None,
        acl_scope="internal",
        year=_year_of(publication),
        meta=meta,
    )
