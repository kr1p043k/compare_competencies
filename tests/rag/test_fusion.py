"""Тесты fusion/отбора: RRF, дедуп, агрегация по документу, MMR, confidence."""

from __future__ import annotations

from src.rag.retriever import (
    aggregate_by_document,
    dedupe,
    hit_confidence,
    mmr_select,
    rrf_fuse,
)
from src.rag.types import RetrievalHit


def _hit(chunk_id: str, text: str, *, doc: str = "d1", dense: float = 0.0,
         sparse: float = 0.0, entity: float = 0.0) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=doc,
        source="rpd",
        title="РПД",
        text=text,
        dense_score=dense,
        sparse_score=sparse,
        entity_score=entity,
    )


def test_rrf_prefers_hit_found_by_both_branches():
    dense = [_hit("a", "текст про команду", dense=0.9), _hit("b", "другой", dense=0.8)]
    sparse = [_hit("b", "другой", sparse=0.5), _hit("c", "третий", sparse=0.4)]
    fused = rrf_fuse({"dense": dense, "sparse": sparse})
    assert fused[0].chunk_id == "b"
    assert all(0.0 <= hit.score <= 1.0 for hit in fused)


def test_rrf_respects_branch_weights():
    dense = [_hit("a", "a", dense=0.9)]
    sparse = [_hit("b", "b", sparse=0.9)]
    fused = rrf_fuse({"dense": dense, "sparse": sparse}, weights={"dense": 3.0, "sparse": 1.0})
    assert fused[0].chunk_id == "a"


def test_rrf_empty_branches():
    assert rrf_fuse({"dense": [], "sparse": []}) == []


def test_dedupe_keeps_best_score_for_same_text():
    first = _hit("a", "одинаковый   текст")
    second = _hit("b", "одинаковый текст")
    first.score = 0.4
    second.score = 0.9
    result = dedupe([first, second])
    assert len(result) == 1
    assert result[0].chunk_id == "b"


def test_aggregate_by_document_limits_chunks_per_doc():
    hits = [_hit(f"c{i}", f"текст {i}", doc="same") for i in range(5)]
    result = aggregate_by_document(hits, max_per_document=2)
    assert len(result) == 2


def test_mmr_diversifies_near_duplicates():
    hits = [
        _hit("a", "python sql pandas numpy", doc="d1"),
        _hit("b", "python sql pandas numpy", doc="d2"),
        _hit("c", "работа в команде коммуникация", doc="d3"),
    ]
    for index, hit in enumerate(hits):
        hit.score = 1.0 - index * 0.01
    selected = mmr_select(hits, top_n=2, lambda_=0.5)
    texts = {hit.chunk_id for hit in selected}
    assert "c" in texts


def test_hit_confidence_prefers_dense_then_entity_then_sparse():
    assert hit_confidence(_hit("a", "x", dense=0.77)) == 0.77
    assert hit_confidence(_hit("b", "x", entity=3.0)) == 0.8
    assert 0.0 < hit_confidence(_hit("c", "x", sparse=0.05)) <= 1.0
    assert hit_confidence(_hit("d", "x")) == 0.0
