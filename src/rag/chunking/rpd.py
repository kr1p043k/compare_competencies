"""Чанкер РПД/аннотаций (data/reference/sfu_annotations_all.json).

Документ уже разбит по специальностям и содержит полный текст рабочей
программы. Режем по смысловым разделам: цели, задачи, компетенции,
содержание, фонд оценочных средств. Это даёт цитаты вида
«РПД / Математический анализ / Компетенции / УК-3».
"""

from __future__ import annotations

import re

from src.rag.chunking.base import build_chunks
from src.rag.types import Chunk, Document, normalize_entity_key

#: Заголовки разделов РПД. Порядок важен: первый матч задаёт границу.
_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Цели",
        re.compile(
            r"^\s*(?:\d\.?\s*)?(?:Цели|Цель|Целью изучения)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    ("Задачи", re.compile(r"^\s*(?:\d\.?\s*)?Задачи\b", re.IGNORECASE | re.MULTILINE)),
    ("Компетенции", re.compile(r"^\s*(?:3\.?\s*)?(?:Компетенц|Планируемые результаты)", re.IGNORECASE | re.MULTILINE)),
    ("Структура", re.compile(r"^\s*(?:4\.?\s*)?(?:Структура|Объем|Объём)\b", re.IGNORECASE | re.MULTILINE)),
    ("Содержание", re.compile(r"^\s*(?:5\.?\s*)?(?:Содержание)\b", re.IGNORECASE | re.MULTILINE)),
    ("ФОС", re.compile(r"^\s*(?:6\.?\s*)?(?:Оценочные|Фонд оценочных)", re.IGNORECASE | re.MULTILINE)),
    (
        "Литература",
        re.compile(
            r"^\s*(?:7\.?\s*)?(?:Учебно-методическое|Литература|Перечень)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    ("МТО", re.compile(r"^\s*(?:8\.?\s*)?(?:Материально-техническое|МТО)", re.IGNORECASE | re.MULTILINE)),
)

#: Строки-разделители, которые не несут смысла.
_NOISE_RE = re.compile(
    r"^(?:МИНИСТЕРСТВО|ФЕДЕРАЛЬНОЕ|ЮЖНЫЙ ФЕДЕРАЛЬНЫЙ|Институт|Кафедра|"
    r"УТВЕРЖДАЮ|РАБОЧАЯ ПРОГРАММА|АННОТАЦИЯ|1\.\s*Цели)",
    re.IGNORECASE,
)

#: Минимальная длина текста документа: короче — OCR-обрывок, не индексируем.
MIN_DOC_CHARS = 400


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Разбить текст РПД на (заголовок, содержимое) по известным разделам."""
    marks: list[tuple[int, str]] = []
    for name, pattern in _SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            marks.append((match.start(), name))
    if not marks:
        return [("Текст", text)]
    marks.sort(key=lambda item: item[0])
    # Дубликаты заголовков (например, «Компетенции» дважды) — оставляем первый.
    seen: set[str] = set()
    unique: list[tuple[int, str]] = []
    for pos, name in marks:
        if name in seen:
            continue
        seen.add(name)
        unique.append((pos, name))
    sections: list[tuple[str, str]] = []
    if unique[0][0] > 0:
        preamble = text[: unique[0][0]].strip()
        if preamble:
            sections.append(("Преамбула", preamble))
    for index, (pos, name) in enumerate(unique):
        end = unique[index + 1][0] if index + 1 < len(unique) else len(text)
        body = text[pos:end].strip()
        if body:
            sections.append((name, body))
    return sections


def _doc_entities(doc: dict) -> list[tuple[str, str]]:
    entities: list[tuple[str, str]] = []
    name = str(doc.get("name") or "").strip()
    if name:
        entities.append(("discipline", normalize_entity_key(name)))
    for code in doc.get("competencies") or []:
        key = normalize_entity_key(str(code))
        if key:
            entities.append(("competency", key))
    return entities


def chunk_rpd_document(
    doc: dict,
    *,
    direction: str | None = None,
    year: int | None = None,
    max_tokens: int = 700,
    min_tokens: int = 40,
) -> Document | None:
    """Построить Document из записи sfu_annotations_all.json.

    Возвращает None, если текста слишком мало (OCR-обрывок) или он пуст.
    """
    text = str(doc.get("text") or "")
    if len(text) < MIN_DOC_CHARS:
        return None
    name = str(doc.get("name") or "").strip() or "Без названия"
    file_name = str(doc.get("file") or "").strip()
    external_id = f"{direction or 'na'}:{file_name or name}"

    sections: list[tuple[str, str]] = []
    for section, body in _split_sections(text):
        if _NOISE_RE.match(body.strip()[:200]):
            body = _NOISE_RE.sub("", body, count=1).strip()
        if body:
            sections.append((f"{name} / {section}", body))

    entities = _doc_entities(doc)
    chunks: list[Chunk] = build_chunks(
        sections,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        entities=entities,
        meta={"file": file_name, "direction": direction, "year": year},
    )
    if not chunks:
        return None
    return Document(
        source="rpd",
        external_id=external_id,
        title=name,
        chunks=chunks,
        acl_scope="internal",
        direction=direction,
        year=year,
        meta={
            "file": file_name,
            "competencies": list(doc.get("competencies") or []),
            "text_len": len(text),
        },
    )


def iter_rpd_documents(
    payload: dict,
    *,
    direction: str | None = None,
    year: int | None = None,
    max_tokens: int = 700,
    min_tokens: int = 40,
):
    """Обойти все документы из sfu_annotations_all.json: {направление: [doc, ...]}."""
    for dir_code, docs in (payload or {}).items():
        if not isinstance(docs, list):
            continue
        effective_direction = direction or str(dir_code)
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            if doc.get("ok") is False:
                continue
            chunked = chunk_rpd_document(
                doc,
                direction=effective_direction,
                year=year,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
            )
            if chunked is not None:
                yield chunked
