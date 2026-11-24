"""Типы подсистемы RAG: документ, чанк, результат retrieval, контекст.

Никакой логики — только dataclass'ы, чтобы chunkers/retriever/context
общались между собой без словарей «на честном слове».
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def content_hash(text: str) -> str:
    """sha256 от нормализованного текста — ключ идемпотентности индекса."""
    normalized = " ".join((text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_entity_key(value: str) -> str:
    """Нормализация ключа сущности: нижний регистр, схлопнутые пробелы."""
    return " ".join((value or "").lower().split())


@dataclass(slots=True)
class Chunk:
    """Один фрагмент документа, который уходит в индекс."""

    text: str
    ord: int = 0
    section_path: str = ""
    token_count: int = 0
    entities: list[tuple[str, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = estimate_tokens(self.text)


@dataclass(slots=True)
class Document:
    """Единица ингестии: источник + внешний id + чанки."""

    source: str
    external_id: str
    title: str
    chunks: list[Chunk] = field(default_factory=list)
    url: str | None = None
    lang: str = "ru"
    acl_scope: str = "internal"
    direction: str | None = None
    year: int | None = None
    discipline_id: str | None = None
    published_at: datetime | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text_for_hash(self) -> str:
        return "\n".join(chunk.text for chunk in self.chunks)

    @property
    def content_hash(self) -> str:
        return content_hash(self.text_for_hash)


@dataclass(slots=True)
class RetrievalHit:
    """Кандидат из retrieval (до сборки контекста)."""

    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    entity_score: float = 0.0
    section_path: str = ""
    url: str | None = None
    ord: int = 0
    external_id: str | None = None
    title_score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        return content_hash(self.text)


@dataclass(slots=True)
class Citation:
    """Ссылка [n] в ответе LLM."""

    n: int
    source: str
    title: str
    url: str | None = None
    section: str = ""
    quote: str = ""
    document_id: str = ""
    chunk_id: str = ""

    def as_block(self) -> dict[str, Any]:
        return {
            "type": "citation",
            "n": self.n,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "section": self.section,
            "quote": self.quote,
        }


@dataclass(slots=True)
class RagQuery:
    """Проанализированный запрос."""

    raw: str
    normalized: str = ""
    lemmas: list[str] = field(default_factory=list)
    intent: str = ""
    profile: str = ""
    entities: dict[str, list[str]] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RagContext:
    """Готовый к вставке в промпт блок доказательств."""

    evidence: list[RetrievalHit] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    prompt_block: str = ""
    degraded: str | None = None
    top_score: float = 0.0
    token_count: int = 0

    def citation_blocks(self) -> list[dict[str, Any]]:
        return [citation.as_block() for citation in self.citations]

    def __bool__(self) -> bool:
        return bool(self.evidence)


def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов: для русского ~3.2 символа на токен."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.2))
