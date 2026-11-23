"""Тесты оркестрации индексации на фейковом store/embedder (без БД и torch)."""

from __future__ import annotations

import pytest

from src.rag.indexer import index_documents
from src.rag.types import Chunk, Document


class FakeEmbedder:
    version = "fake-model_st0"

    def __init__(self, fail_on: str | None = None):
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    async def encode_async(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    def __init__(self):
        self.generation = 1
        self.docs: dict[str, Document] = {}
        self.hashes: dict[str, str] = {}
        self.runs: list[dict] = []
        self.tombstoned: list[set[str]] = []
        self.active = 1
        self.fail_external: set[str] = set()

    async def active_generation(self, source):
        return self.active

    async def next_generation(self, source, spec_version):
        self.generation = self.active + 1
        return self.generation

    async def ensure_source(self, source, spec_version):
        return self.active

    async def existing_hashes(self, source, generation):
        return dict(self.hashes)

    async def tombstone_missing(self, source, generation, keep_external_ids):
        removed = [key for key in self.hashes if key not in keep_external_ids]
        for key in removed:
            self.hashes.pop(key, None)
            self.docs.pop(key, None)
        self.tombstoned.append(set(keep_external_ids))
        return len(removed)

    async def start_run(self, source, generation):
        run = {"source": source, "generation": generation, "status": "running"}
        self.runs.append(run)
        return f"run-{len(self.runs)}"

    async def finish_run(self, run_id, *, status, error=None, **counters):
        self.runs[-1].update({"status": status, "error": error, **counters})

    async def set_active_generation(self, source, generation):
        self.active = generation

    async def upsert_document(self, doc, *, generation, model_version, embeddings):
        if doc.external_id in self.fail_external:
            raise RuntimeError("boom")
        assert len(embeddings) == len(doc.chunks)
        self.docs[doc.external_id] = doc
        self.hashes[doc.external_id] = doc.content_hash
        return f"uuid-{doc.external_id}", len(doc.chunks)


def _doc(external_id: str, text: str = "Компетенция УК-3 формирует командную работу.") -> Document:
    return Document(
        source="rpd",
        external_id=external_id,
        title=f"Документ {external_id}",
        chunks=[Chunk(text=text, ord=0)],
    )


@pytest.mark.asyncio
async def test_first_index_upserts_all_documents():
    store, embedder = FakeStore(), FakeEmbedder()
    stats = await index_documents(
        [_doc("a"), _doc("b")], source="rpd", store=store, embedder=embedder
    )
    assert stats.docs_upserted == 2
    assert stats.chunks_written == 2
    assert stats.docs_skipped == 0
    assert store.runs[-1]["status"] == "ok"
    assert len(embedder.calls) == 1  # один батч на все чанки


@pytest.mark.asyncio
async def test_second_run_skips_unchanged_documents():
    store, embedder = FakeStore(), FakeEmbedder()
    docs = [_doc("a"), _doc("b")]
    await index_documents(docs, source="rpd", store=store, embedder=embedder)
    stats = await index_documents(docs, source="rpd", store=store, embedder=embedder)
    assert stats.docs_skipped == 2
    assert stats.docs_upserted == 0
    assert len(embedder.calls) == 1


@pytest.mark.asyncio
async def test_changed_document_is_reindexed():
    store, embedder = FakeStore(), FakeEmbedder()
    await index_documents([_doc("a")], source="rpd", store=store, embedder=embedder)
    stats = await index_documents(
        [_doc("a", "Совсем другой текст про анализ данных.")],
        source="rpd",
        store=store,
        embedder=embedder,
    )
    assert stats.docs_upserted == 1
    assert stats.docs_skipped == 0


@pytest.mark.asyncio
async def test_removed_document_is_tombstoned():
    store, embedder = FakeStore(), FakeEmbedder()
    await index_documents([_doc("a"), _doc("b")], source="rpd", store=store, embedder=embedder)
    stats = await index_documents([_doc("a")], source="rpd", store=store, embedder=embedder)
    assert stats.docs_removed == 1
    assert "b" not in store.hashes


@pytest.mark.asyncio
async def test_full_reindex_switches_generation():
    store, embedder = FakeStore(), FakeEmbedder()
    await index_documents([_doc("a")], source="rpd", store=store, embedder=embedder)
    stats = await index_documents(
        [_doc("a")], source="rpd", store=store, embedder=embedder, full_reindex=True
    )
    assert stats.generation == 2
    assert store.active == 2
    assert stats.docs_upserted == 1


@pytest.mark.asyncio
async def test_dry_run_does_not_touch_store():
    store, embedder = FakeStore(), FakeEmbedder()
    stats = await index_documents(
        [_doc("a"), _doc("b")], source="rpd", store=store, embedder=embedder, dry_run=True
    )
    assert stats.dry_run is True
    assert stats.chunks_written == 2
    assert store.docs == {}
    assert store.runs == []
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_document_failure_marks_run_partial():
    store, embedder = FakeStore(), FakeEmbedder()
    store.fail_external = {"b"}
    stats = await index_documents(
        [_doc("a"), _doc("b")], source="rpd", store=store, embedder=embedder
    )
    assert stats.docs_upserted == 1
    assert stats.errors
    assert store.runs[-1]["status"] == "partial"


@pytest.mark.asyncio
async def test_empty_input_returns_zero_stats():
    store, embedder = FakeStore(), FakeEmbedder()
    stats = await index_documents([], source="rpd", store=store, embedder=embedder)
    assert stats.docs_seen == 0
    assert store.runs == []
