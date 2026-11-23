"""Чанкер вакансий hh.ru (таблица vacancies или hh_vacancies_*.json).

Важно: в search-выгрузке `description` часто пуст — тогда индексируем
snippet_requirement / snippet_responsibility и честно помечаем
``meta.partial = True``, чтобы в цитате было видно «фрагмент вакансии».
"""

from __future__ import annotations

from typing import Any

from src.rag.chunking.base import build_chunks
from src.rag.types import Document, normalize_entity_key

MIN_DOC_CHARS = 40

#: Сниппеты вакансий короткие по своей природе — не отбрасываем их порогом
#: min_tokens, иначе половина рынка выпадет из индекса.
_SNIPPET_MIN_TOKENS = 8


def _as_list(value: Any) -> list[str]:
    """Список навыков из БД (строки) или из hh API ([{"name": ...}])."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = (
                str(item.get("name") or "").strip()
                if isinstance(item, dict)
                else str(item).strip()
            )
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    return []


def _field(vacancy: Any, name: str, default: Any = None) -> Any:
    if isinstance(vacancy, dict):
        return vacancy.get(name, default)
    return getattr(vacancy, name, default)


def _snippet_field(vacancy: Any, key: str) -> str:
    """hh search отдаёт snippet.requirement / snippet.responsibility, БД — плоские колонки."""
    snippet = _field(vacancy, "snippet")
    if isinstance(snippet, dict):
        value = str(snippet.get(key) or "").strip()
        if value:
            return value
    return str(_field(vacancy, f"snippet_{key}") or "").strip()


def chunk_vacancy(vacancy: Any, *, max_tokens: int = 700, min_tokens: int = 40) -> Document | None:
    """Построить Document из вакансии (dict из hh API или ORM-объекта)."""
    hh_id = _field(vacancy, "hh_id") or _field(vacancy, "id")
    if hh_id is None:
        return None
    name = str(_field(vacancy, "name") or "").strip() or "Вакансия"
    description = str(_field(vacancy, "description") or "").strip()
    requirement = _snippet_field(vacancy, "requirement")
    responsibility = _snippet_field(vacancy, "responsibility")
    skills = _as_list(_field(vacancy, "key_skills")) or _as_list(_field(vacancy, "parsed_skills"))

    sections: list[tuple[str, str]] = []
    if requirement:
        sections.append((f"{name} / Требования", requirement))
    if responsibility:
        sections.append((f"{name} / Обязанности", responsibility))
    if description:
        sections.append((f"{name} / Описание", description))
    if skills:
        sections.append((f"{name} / Навыки", ", ".join(skills)))

    total = sum(len(body) for _, body in sections)
    if total < MIN_DOC_CHARS:
        return None

    partial = not description
    entities: list[tuple[str, str]] = []
    for skill in skills:
        key = normalize_entity_key(skill)
        if key:
            entities.append(("skill", key))

    meta = {
        "employer": _field(vacancy, "employer_name"),
        "area": _field(vacancy, "area_name"),
        "experience": _field(vacancy, "experience"),
        "partial": partial,
        "skills": skills[:40],
    }
    chunks = build_chunks(
        sections,
        max_tokens=max_tokens,
        min_tokens=min(min_tokens, _SNIPPET_MIN_TOKENS),
        entities=entities,
        meta=meta,
    )
    if not chunks:
        return None

    published_at = _field(vacancy, "published_at")
    return Document(
        source="vacancies",
        external_id=str(hh_id),
        title=name,
        chunks=chunks,
        url=_field(vacancy, "alternate_url"),
        acl_scope="internal",
        published_at=published_at if hasattr(published_at, "year") else None,
        meta=meta,
    )
