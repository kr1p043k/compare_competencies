"""Офлайн-оценка retrieval по golden-set: recall@k, MRR, nDCG@k.

Метрики считаются по уже полученным hits, поэтому модуль чистый: ни БД, ни
модели. CLI (``src.cli.rag_eval``) гоняет реальный ``RagService.search`` и
складывает результаты сюда; тесты проверяют метрики на синтетике.

Релевантность определяется по ключам golden-элемента: ``external_id``
документа, заголовок или ``source:title``. Любое совпадение делает hit
релевантным - корпус живёт в трёх источниках, и один и тот же заголовок
может встретиться в РПД и в карточке KRM.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.rag.types import RetrievalHit


def normalize_key(value: str) -> str:
    """Ключ сравнения: регистр, лишние пробелы, ё/е."""
    return " ".join((value or "").casefold().replace("ё", "е").split())


@dataclass(slots=True)
class GoldenItem:
    """Один вопрос golden-set и документы, которые обязаны попасть в top-k."""

    query: str
    expect: list[str] = field(default_factory=list)
    profile: str | None = None
    sources: list[str] | None = None
    direction: str | None = None
    year: int | None = None
    note: str = ""

    def keys(self) -> set[str]:
        return {normalize_key(key) for key in self.expect if key and key.strip()}


def hit_keys(hit: RetrievalHit) -> set[str]:
    """Ключи, по которым hit может совпасть с golden-элементом."""
    keys: set[str] = set()
    external_id = getattr(hit, "external_id", None)
    if external_id:
        keys.add(normalize_key(str(external_id)))
    if hit.title:
        keys.add(normalize_key(hit.title))
        keys.add(normalize_key(f"{hit.source}:{hit.title}"))
    return keys


def matched_keys(hits: Sequence[RetrievalHit], golden: GoldenItem) -> set[str]:
    """Какие ожидаемые ключи реально закрыты выданными hits."""
    expected = golden.keys()
    found: set[str] = set()
    for hit in hits:
        found |= hit_keys(hit) & expected
    return found


def is_relevant(hit: RetrievalHit, golden: GoldenItem) -> bool:
    return bool(hit_keys(hit) & golden.keys())


def recall_at_k(hits: Sequence[RetrievalHit], golden: GoldenItem, k: int) -> float:
    expected = golden.keys()
    if not expected or k <= 0:
        return 0.0
    return len(matched_keys(hits[:k], golden)) / len(expected)


def reciprocal_rank(hits: Sequence[RetrievalHit], golden: GoldenItem, k: int | None = None) -> float:
    """1/ранг первого релевантного hit; 0, если в пределах k его нет."""
    limit = len(hits) if k is None else min(k, len(hits))
    for index in range(limit):
        if is_relevant(hits[index], golden):
            return 1.0 / (index + 1)
    return 0.0


def dcg_at_k(hits: Sequence[RetrievalHit], golden: GoldenItem, k: int) -> float:
    """DCG с бинарным gain: вклад hit = 1/log2(rank+1).

    Ключ учитывается один раз: несколько чанков одного документа (или разные
    источники с тем же заголовком) не должны раздувать DCG выше идеального.
    """
    expected = golden.keys()
    seen: set[str] = set()
    total = 0.0
    for index, hit in enumerate(hits[: max(k, 0)]):
        fresh = (hit_keys(hit) & expected) - seen
        if fresh:
            seen |= fresh
            total += 1.0 / math.log2(index + 2)
    return total


def ndcg_at_k(hits: Sequence[RetrievalHit], golden: GoldenItem, k: int) -> float:
    """nDCG@k: DCG, нормированный на идеальный порядок ожидаемых ключей."""
    expected = golden.keys()
    if not expected or k <= 0:
        return 0.0
    ideal_hits = min(len(expected), k)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    if ideal <= 0:
        return 0.0
    return dcg_at_k(hits, golden, k) / ideal


def evaluate_item(
    hits: Sequence[RetrievalHit], golden: GoldenItem, *, k: int = 5
) -> dict[str, Any]:
    """Метрики одного вопроса плюс диагностика (что нашлось, что нет)."""
    expected = golden.keys()
    found = matched_keys(hits, golden)
    return {
        "query": golden.query,
        "profile": golden.profile or "",
        "expected": len(expected),
        "recall_at_k": round(recall_at_k(hits, golden, k), 4),
        "mrr": round(reciprocal_rank(hits, golden, k), 4),
        "ndcg_at_k": round(ndcg_at_k(hits, golden, k), 4),
        "hits": len(hits),
        "missing": sorted(expected - found),
        "top": [
            {
                "source": hit.source,
                "title": hit.title,
                "external_id": hit.external_id,
                "score": round(hit.score, 4),
            }
            for hit in hits[:k]
        ],
    }


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Средние метрики по набору; вопросы без expect в средние не входят."""
    scored = [row for row in rows if row.get("expected")]
    skipped = [row for row in rows if not row.get("expected")]
    if not scored:
        return {
            "queries": 0,
            "skipped": len(skipped),
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "empty_result_queries": 0,
        }
    return {
        "queries": len(scored),
        "skipped": len(skipped),
        "recall_at_k": round(sum(row["recall_at_k"] for row in scored) / len(scored), 4),
        "mrr": round(sum(row["mrr"] for row in scored) / len(scored), 4),
        "ndcg_at_k": round(sum(row["ndcg_at_k"] for row in scored) / len(scored), 4),
        "empty_result_queries": sum(1 for row in scored if not row["hits"]),
    }


def parse_golden(payload: Any) -> list[GoldenItem]:
    """Разобрать golden-set: dict с ``items`` или голый список.

    Ошибка формата - ``ValueError`` с номером элемента: молча пропускать
    битый вопрос нельзя, иначе recall посчитается по неполному набору.
    """
    if isinstance(payload, dict):
        raw_items = payload.get("items")
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raise ValueError("golden-set должен быть объектом с 'items' или списком")

    if not isinstance(raw_items, list):
        raise ValueError("golden-set: 'items' должен быть списком")

    items: list[GoldenItem] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"golden-set[{index}]: элемент должен быть объектом")
        query = raw.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"golden-set[{index}]: нужен непустой 'query'")
        expect_raw = raw.get("expect", [])
        if isinstance(expect_raw, str):
            expect_raw = [expect_raw]
        if not isinstance(expect_raw, list) or any(
            not isinstance(key, str) for key in expect_raw
        ):
            raise ValueError(f"golden-set[{index}]: 'expect' должен быть списком строк")
        sources = raw.get("sources")
        if sources is not None and (
            not isinstance(sources, list) or any(not isinstance(src, str) for src in sources)
        ):
            raise ValueError(f"golden-set[{index}]: 'sources' должен быть списком строк")
        year = raw.get("year")
        if year is not None and not isinstance(year, int):
            raise ValueError(f"golden-set[{index}]: 'year' должен быть числом")
        items.append(
            GoldenItem(
                query=query.strip(),
                expect=[key.strip() for key in expect_raw if key.strip()],
                profile=raw.get("profile") or None,
                sources=sources,
                direction=raw.get("direction") or None,
                year=year,
                note=str(raw.get("note") or ""),
            )
        )
    return items


def load_golden(path: str | Path) -> list[GoldenItem]:
    with Path(path).open(encoding="utf-8") as handle:
        return parse_golden(json.load(handle))


def golden_template() -> dict[str, Any]:
    """Схема файла: используется в ``rag-eval --template``."""
    return {
        "version": 1,
        "k": 5,
        "items": [
            {
                "query": "чем покрыть машинное обучение в РПД",
                "expect": ["09.03.02:Машинное обучение", "machine learning"],
                "profile": "curriculum_design",
                "sources": ["rpd", "krm"],
                "direction": "09.03.02",
                "year": 2025,
                "note": "ключи: external_id | title | source:title",
            }
        ],
    }
