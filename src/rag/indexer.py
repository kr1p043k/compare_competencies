"""Оркестрация индексации: документы -> чанки -> эмбеддинги -> upsert.

Два режима:
  * инкрементальный (по умолчанию) — пишет в активное поколение, пропускает
    документы с неизменившимся content_hash, помечает удалённые tombstone'ом;
  * полный реиндекс — новое поколение, после успеха атомарное переключение
    (blue-green): поиск никогда не видит полупустой индекс.

Эмбеддинги считаются одним батчем по всем изменившимся чанкам: так torch
прогревается один раз, а не на каждый документ.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import structlog

from src.rag.embedder import Embedder
from src.rag.store import RagStore
from src.rag.types import Document

logger = structlog.get_logger(__name__)

#: Версия правил чанкования. Меняешь chunker — поднимай, иначе инкремент
#: пропустит документы с тем же текстом, но другим разбиением.
SPEC_VERSION = "1"


@dataclass(slots=True)
class IndexStats:
    source: str = ""
    generation: int = 0
    docs_seen: int = 0
    docs_upserted: int = 0
    docs_skipped: int = 0
    docs_removed: int = 0
    chunks_written: int = 0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "generation": self.generation,
            "docs_seen": self.docs_seen,
            "docs_upserted": self.docs_upserted,
            "docs_skipped": self.docs_skipped,
            "docs_removed": self.docs_removed,
            "chunks_written": self.chunks_written,
            "dry_run": self.dry_run,
            "errors": self.errors[:10],
        }


async def index_documents(
    docs: Iterable[Document],
    *,
    source: str,
    store: RagStore | None = None,
    embedder: Embedder | None = None,
    spec_version: str = SPEC_VERSION,
    full_reindex: bool = False,
    dry_run: bool = False,
) -> IndexStats:
    """Проиндексировать документы источника. Никогда не бросает на одном битом документе."""
    store = store or RagStore()
    prepared = [doc for doc in docs if doc.chunks]
    stats = IndexStats(source=source, docs_seen=len(prepared), dry_run=dry_run)
    if not prepared:
        logger.info("rag_index_no_documents", source=source)
        return stats

    if dry_run:
        stats.chunks_written = sum(len(doc.chunks) for doc in prepared)
        stats.docs_upserted = len(prepared)
        try:
            stats.generation = await store.active_generation(source) or 0
        except Exception:
            # dry-run не должен падать без БД: он нужен и для оценки корпуса.
            logger.debug("indexer_suppressed")
            stats.generation = 0
        return stats

    if embedder is None:
        from src.rag.embedder import get_embedder

        embedder = get_embedder()

    if full_reindex:
        generation = await store.next_generation(source, spec_version)
        existing: dict[str, str] = {}
    else:
        generation = await store.ensure_source(source, spec_version)
        existing = await store.existing_hashes(source, generation)
    stats.generation = generation

    changed: list[Document] = []
    for doc in prepared:
        if existing.get(doc.external_id) == doc.content_hash:
            stats.docs_skipped += 1
        else:
            changed.append(doc)

    if not changed:
        if not full_reindex:
            stats.docs_removed = await store.tombstone_missing(
                source, generation, {doc.external_id for doc in prepared}
            )
        else:
            await store.set_active_generation(source, generation)
        logger.info("rag_index_nothing_changed", source=source, generation=generation)
        return stats

    run_id = await store.start_run(source, generation)
    try:
        flat_texts: list[str] = []
        for doc in changed:
            flat_texts.extend(chunk.text for chunk in doc.chunks)
        vectors = await embedder.encode_async(flat_texts)
        if len(vectors) != len(flat_texts):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(flat_texts)} chunks"
            )

        cursor = 0
        for doc in changed:
            count = len(doc.chunks)
            doc_vectors = vectors[cursor : cursor + count]
            cursor += count
            try:
                _, written = await store.upsert_document(
                    doc,
                    generation=generation,
                    model_version=embedder.version,
                    embeddings=doc_vectors,
                )
                stats.docs_upserted += 1
                stats.chunks_written += written
            except Exception as exc:
                stats.errors.append(f"{doc.external_id}: {exc}")
                logger.warning(
                    "rag_index_document_failed",
                    source=source,
                    external_id=doc.external_id,
                    error=str(exc),
                )

        if not full_reindex:
            stats.docs_removed = await store.tombstone_missing(
                source, generation, {doc.external_id for doc in prepared}
            )
        else:
            await store.set_active_generation(source, generation)

        status = "ok" if not stats.errors else "partial"
        await store.finish_run(
            run_id,
            status=status,
            docs_seen=stats.docs_seen,
            docs_upserted=stats.docs_upserted,
            docs_skipped=stats.docs_skipped,
            docs_removed=stats.docs_removed,
            chunks_written=stats.chunks_written,
        )
    except Exception as exc:
        await store.finish_run(run_id, status="failed", error=str(exc)[:2000])
        logger.exception("rag_index_failed", source=source)
        raise

    logger.info("rag_index_done", **stats.as_dict())
    _observe_index(stats)
    return stats


def _observe_index(stats: IndexStats) -> None:
    """Метрики ингестии: документы по исходам и записанные чанки."""
    try:
        from src.monitoring.metrics import rag_index_chunks, rag_index_documents

        rag_index_documents.labels(source=stats.source, status="upserted").inc(stats.docs_upserted)
        rag_index_documents.labels(source=stats.source, status="skipped").inc(stats.docs_skipped)
        rag_index_documents.labels(source=stats.source, status="removed").inc(stats.docs_removed)
        if stats.errors:
            rag_index_documents.labels(source=stats.source, status="failed").inc(len(stats.errors))
        rag_index_chunks.labels(source=stats.source).inc(stats.chunks_written)
    except Exception:
        logger.debug("indexer_suppressed")
        pass
