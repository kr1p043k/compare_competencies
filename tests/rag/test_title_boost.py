"""Lexical title boost: заголовок, совпавший с запросом, важнее косинуса."""

from __future__ import annotations

import pytest

from src.rag.retriever import (
    HybridRetriever,
    apply_title_boost,
    hit_confidence,
    title_coverage,
)
from src.rag.settings import RagSettings
from src.rag.types import RagQuery, RetrievalHit


def _hit(
    title: str,
    *,
    score: float = 0.5,
    dense: float = 0.0,
    text: str = "тело",
    doc: str | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"c-{title}-{score}",
        document_id=doc or f"d-{title}",
        source="rpd",
        title=title,
        text=text,
        score=score,
        dense_score=dense,
    )


def test_title_coverage_exact_match() -> None:
    assert title_coverage("Методы компьютерного моделирования", "методы компьютерного моделирования") == 1.0


def test_title_coverage_handles_inflection() -> None:
    # «моделирования» vs «моделирование», «методы» vs «метод»
    assert title_coverage("Метод компьютерного моделирования", "методы моделирования") == 1.0


def test_title_coverage_partial() -> None:
    # «по» (2 символа) отбрасывается, «junior» в заголовке отсутствует
    coverage = title_coverage(
        "Инженер по информационной безопасности", "инженер по информационной безопасности junior"
    )
    assert coverage == pytest.approx(3 / 4)


def test_title_coverage_ignores_short_terms_and_empty() -> None:
    # «и» (1 символ) не должен считаться термином
    assert title_coverage("Теория чисел", "теория чисел и практика") == pytest.approx(2 / 3)
    assert title_coverage("", "теория чисел") == 0.0
    assert title_coverage("Теория чисел", "") == 0.0
    assert title_coverage("Теория чисел", "и в на") == 0.0


def test_apply_title_boost_reorders_and_marks_hit() -> None:
    weak_but_matching = _hit("Теория чисел", score=0.7)
    strong_but_other = _hit("Численные методы", score=0.9)
    boosted = apply_title_boost([strong_but_other, weak_but_matching], "теория чисел")

    assert boosted[0] is weak_but_matching
    assert weak_but_matching.title_score == 1.0
    assert weak_but_matching.score == pytest.approx(1.05)
    assert strong_but_other.title_score == 0.0
    assert strong_but_other.score == 0.9


def test_apply_title_boost_skips_low_coverage() -> None:
    hit = _hit("Теория вероятностей и математическая статистика", score=0.5)
    apply_title_boost([hit], "теория чисел")
    assert hit.title_score == 0.0
    assert hit.score == 0.5


def test_apply_title_boost_noop_without_query() -> None:
    hit = _hit("Теория чисел", score=0.5)
    assert apply_title_boost([hit], "") == [hit]
    assert hit.score == 0.5


def test_apply_title_boost_can_be_disabled() -> None:
    hit = _hit("Теория чисел", score=0.5)
    apply_title_boost([hit], "теория чисел", boost=0.0)
    assert hit.title_score == 0.0
    assert hit.score == 0.5


def test_hit_confidence_uses_title_score() -> None:
    hit = _hit("Теория чисел", dense=0.1)
    hit.title_score = 1.0
    assert hit_confidence(hit) == pytest.approx(0.8)


def test_hit_confidence_without_title_unchanged() -> None:
    assert hit_confidence(_hit("Другое", dense=0.2)) == pytest.approx(0.2)


class _FakeStore:
    def __init__(self, dense_hits: list[RetrievalHit]) -> None:
        self._dense_hits = dense_hits

    async def dense_search(self, vector, **kwargs):
        return self._dense_hits

    async def sparse_search(self, text, **kwargs):
        return []

    async def entity_search(self, pairs, **kwargs):
        return []


class _FakeEmbedder:
    available = True

    async def encode_one(self, text: str):
        return [0.1, 0.2]


async def test_retrieve_applies_title_boost_end_to_end() -> None:
    matching = _hit("Теория чисел", score=0.0, dense=0.3, text="делимость целых чисел")
    other = _hit("Численные методы", score=0.0, dense=0.6, text="метод конечных разностей")
    settings = RagSettings(enabled=True, dense_k=10, sparse_k=10, entity_k=10, top_n=2)
    retriever = HybridRetriever(
        store=_FakeStore([other, matching]), embedder=_FakeEmbedder(), settings=settings
    )

    hits, degraded = await retriever.retrieve(RagQuery(raw="теория чисел", normalized="теория чисел"))

    assert degraded is None
    assert hits[0].title == "Теория чисел"
    assert hits[0].title_score == 1.0


async def test_retrieve_without_title_match_keeps_semantic_order() -> None:
    first = _hit("Численные методы", score=0.0, dense=0.6, text="метод конечных разностей")
    second = _hit("Теория вероятностей", score=0.0, dense=0.4, text="случайные величины и события")
    settings = RagSettings(enabled=True, dense_k=10, sparse_k=10, entity_k=10, top_n=2)
    retriever = HybridRetriever(
        store=_FakeStore([first, second]), embedder=_FakeEmbedder(), settings=settings
    )

    hits, _ = await retriever.retrieve(RagQuery(raw="дифференцирование", normalized="дифференцирование"))

    assert [hit.title for hit in hits] == ["Численные методы", "Теория вероятностей"]
