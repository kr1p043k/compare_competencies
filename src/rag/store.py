"""Доступ к таблицам rag_*: upsert, поколения, три ветки поиска.

Сознательно без ORM-моделей: таблицы живут только в alembic-ревизии, а
запросы здесь — чистый SQL (как в ``src/services/llm_cache.py``). Так проще
держать HNSW-операторы, ts_rank_cd и фильтры ACL в одном месте.

Ключевое решение: поиск всегда фильтрует ``d.generation = s.active_generation``
через join на ``rag_sources``. Это снимает необходимость передавать поколения
по каждому источнику и делает blue-green переключение атомарным.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import async_session_factory
from src.rag.embedder import format_vector
from src.rag.types import Document, RetrievalHit

logger = structlog.get_logger(__name__)

_HIT_COLUMNS = (
    "c.id AS chunk_id, c.document_id, c.text, c.section_path, c.ord, c.meta AS chunk_meta, "
    "d.source, d.title, d.url, d.external_id"
)


def _json(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, default=str)


class RagStore:
    """Тонкий слой над rag_documents / rag_chunks / rag_chunk_entities."""

    def __init__(self, session: AsyncSession | None = None) -> None:
        self._session = session

    # ------------------------------------------------------------- session
    @asynccontextmanager
    async def _ctx(self):
        """Чужая сессия — не коммитим; своя — commit/rollback/close."""
        if self._session is not None:
            yield self._session
            return
        session = async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # ------------------------------------------------------------ sources
    async def ensure_source(self, source: str, spec_version: str, acl_scope: str = "internal") -> int:
        async with self._ctx() as session:
            row = (
                await session.execute(
                    text("SELECT active_generation FROM rag_sources WHERE source = :s"),
                    {"s": source},
                )
            ).first()
            if row is not None:
                await session.execute(
                    text(
                        "UPDATE rag_sources SET spec_version = :v, acl_scope = :a, "
                        "updated_at = NOW() WHERE source = :s"
                    ),
                    {"s": source, "v": spec_version, "a": acl_scope},
                )
                return int(row[0])
            await session.execute(
                text(
                    "INSERT INTO rag_sources (source, spec_version, acl_scope, active_generation) "
                    "VALUES (:s, :v, :a, 1)"
                ),
                {"s": source, "v": spec_version, "a": acl_scope},
            )
            return 1

    async def active_generation(self, source: str) -> int | None:
        async with self._ctx() as session:
            row = (
                await session.execute(
                    text("SELECT active_generation FROM rag_sources WHERE source = :s"),
                    {"s": source},
                )
            ).first()
        return int(row[0]) if row is not None else None

    async def next_generation(self, source: str, spec_version: str) -> int:
        """Новое поколение для полного реиндекса (blue-green)."""
        async with self._ctx() as session:
            row = (
                await session.execute(
                    text("SELECT active_generation FROM rag_sources WHERE source = :s"),
                    {"s": source},
                )
            ).first()
            generation = int(row[0]) + 1 if row is not None else 1
            if row is None:
                await session.execute(
                    text(
                        "INSERT INTO rag_sources (source, spec_version, active_generation) "
                        "VALUES (:s, :v, :g)"
                    ),
                    {"s": source, "v": spec_version, "g": generation},
                )
            else:
                await session.execute(
                    text("UPDATE rag_sources SET spec_version = :v, updated_at = NOW() WHERE source = :s"),
                    {"s": source, "v": spec_version},
                )
        return generation

    async def set_active_generation(self, source: str, generation: int) -> None:
        async with self._ctx() as session:
            await session.execute(
                text("UPDATE rag_sources SET active_generation = :g, updated_at = NOW() WHERE source = :s"),
                {"s": source, "g": generation},
            )

    # ---------------------------------------------------------- ingest runs
    async def start_run(self, source: str, generation: int) -> str:
        async with self._ctx() as session:
            row = (
                await session.execute(
                    text(
                        "INSERT INTO rag_ingest_runs (source, generation) "
                        "VALUES (:s, :g) RETURNING id"
                    ),
                    {"s": source, "g": generation},
                )
            ).first()
        return str(row[0])

    async def finish_run(
        self, run_id: str, *, status: str, error: str | None = None, **counters: int
    ) -> None:
        allowed = {"docs_seen", "docs_upserted", "docs_skipped", "docs_removed", "chunks_written"}
        sets = ["finished_at = NOW()", "status = :status", "error = :error"]
        params: dict[str, Any] = {"id": run_id, "status": status, "error": error}
        for key, value in counters.items():
            if key in allowed:
                sets.append(f"{key} = :{key}")
                params[key] = int(value)
        async with self._ctx() as session:
            await session.execute(
                text(f"UPDATE rag_ingest_runs SET {', '.join(sets)} WHERE id = :id"),
                params,
            )

    # ------------------------------------------------------------ documents
    async def existing_hashes(self, source: str, generation: int) -> dict[str, str]:
        async with self._ctx() as session:
            rows = await session.execute(
                text(
                    "SELECT external_id, content_hash FROM rag_documents "
                    "WHERE source = :s AND generation = :g AND valid_to IS NULL"
                ),
                {"s": source, "g": generation},
            )
            return {str(row[0]): str(row[1]) for row in rows}

    async def upsert_document(
        self,
        doc: Document,
        *,
        generation: int,
        model_version: str,
        embeddings: list[list[float]],
    ) -> tuple[str, int]:
        """Записать документ и чанки. Возвращает (document_id, число чанков)."""
        if len(embeddings) != len(doc.chunks):
            raise ValueError(f"embeddings/chunks mismatch: {len(embeddings)} != {len(doc.chunks)}")
        published_at = doc.published_at if isinstance(doc.published_at, datetime) else None
        async with self._ctx() as session:
            row = (
                await session.execute(
                    text(
                        """
                        INSERT INTO rag_documents
                            (source, external_id, generation, title, url, lang, content_hash,
                             model_version, acl_scope, direction, year, discipline_id,
                             published_at, valid_from, valid_to, meta)
                        VALUES
                            (:source, :external_id, :generation, :title, :url, :lang, :content_hash,
                             :model_version, :acl_scope, :direction, :year, :discipline_id,
                             :published_at, NOW(), NULL, CAST(:meta AS jsonb))
                        ON CONFLICT (source, external_id, generation) DO UPDATE SET
                            title = EXCLUDED.title,
                            url = EXCLUDED.url,
                            content_hash = EXCLUDED.content_hash,
                            model_version = EXCLUDED.model_version,
                            acl_scope = EXCLUDED.acl_scope,
                            direction = EXCLUDED.direction,
                            year = EXCLUDED.year,
                            discipline_id = EXCLUDED.discipline_id,
                            published_at = EXCLUDED.published_at,
                            valid_to = NULL,
                            meta = EXCLUDED.meta
                        RETURNING id
                        """
                    ),
                    {
                        "source": doc.source,
                        "external_id": doc.external_id,
                        "generation": generation,
                        "title": doc.title[:500],
                        "url": doc.url,
                        "lang": doc.lang,
                        "content_hash": doc.content_hash,
                        "model_version": model_version,
                        "acl_scope": doc.acl_scope,
                        "direction": doc.direction,
                        "year": doc.year,
                        "discipline_id": doc.discipline_id,
                        "published_at": published_at,
                        "meta": _json(doc.meta),
                    },
                )
            ).first()
            document_id = str(row[0])
            await session.execute(
                text("DELETE FROM rag_chunks WHERE document_id = CAST(:d AS uuid)"),
                {"d": document_id},
            )
            for chunk, vector in zip(doc.chunks, embeddings, strict=True):
                chunk_row = (
                    await session.execute(
                        text(
                            """
                            INSERT INTO rag_chunks
                                (document_id, generation, ord, section_path, text,
                                 token_count, embedding, tsv, meta)
                            VALUES
                                (CAST(:document_id AS uuid), :generation, :ord, :section_path, :text,
                                 :token_count, CAST(:embedding AS vector),
                                 to_tsvector('russian', :text), CAST(:meta AS jsonb))
                            RETURNING id
                            """
                        ),
                        {
                            "document_id": document_id,
                            "generation": generation,
                            "ord": chunk.ord,
                            "section_path": chunk.section_path[:500],
                            "text": chunk.text,
                            "token_count": chunk.token_count,
                            "embedding": format_vector(vector),
                            "meta": _json(chunk.meta),
                        },
                    )
                ).first()
                chunk_id = str(chunk_row[0])
                for entity_type, entity_key in chunk.entities:
                    await session.execute(
                        text(
                            "INSERT INTO rag_chunk_entities (chunk_id, entity_type, entity_key) "
                            "VALUES (CAST(:c AS uuid), :t, :k) ON CONFLICT DO NOTHING"
                        ),
                        {"c": chunk_id, "t": entity_type[:40], "k": entity_key[:200]},
                    )
        return document_id, len(doc.chunks)

    async def tombstone_missing(
        self, source: str, generation: int, keep_external_ids: set[str]
    ) -> int:
        """Пометить удалённые документы (valid_to) вместо DELETE."""
        async with self._ctx() as session:
            rows = await session.execute(
                text(
                    "SELECT id, external_id FROM rag_documents "
                    "WHERE source = :s AND generation = :g AND valid_to IS NULL"
                ),
                {"s": source, "g": generation},
            )
            stale = [str(row[0]) for row in rows if str(row[1]) not in keep_external_ids]
            if not stale:
                return 0
            placeholders = ", ".join(f":id{index}" for index in range(len(stale)))
            params = {f"id{index}": value for index, value in enumerate(stale)}
            await session.execute(
                text(
                    "UPDATE rag_documents SET valid_to = NOW() "
                    f"WHERE id IN ({placeholders})"
                ),
                params,
            )
        return len(stale)

    # -------------------------------------------------------------- search
    def _base_filters(
        self,
        *,
        scopes: list[str],
        sources: list[str] | None,
        direction: str | None,
        year: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Общие фильтры поиска.

        Списки разворачиваются в явные :scope0, :src0, ... — так поведение
        одинаково на asyncpg/psycopg2 и не зависит от expanding bindparam.
        """
        params: dict[str, Any] = {}
        scope_placeholders = []
        for index, scope in enumerate(scopes or ["internal"]):
            key = f"scope{index}"
            scope_placeholders.append(f":{key}")
            params[key] = scope
        clauses = [
            "d.valid_to IS NULL",
            f"d.acl_scope IN ({', '.join(scope_placeholders)})",
            "d.generation = s.active_generation",
        ]
        if sources:
            source_placeholders = []
            for index, source in enumerate(sources):
                key = f"src{index}"
                source_placeholders.append(f":{key}")
                params[key] = source
            clauses.append(f"d.source IN ({', '.join(source_placeholders)})")
        if direction:
            clauses.append("d.direction = :direction")
            params["direction"] = direction
        if year is not None:
            clauses.append("d.year = :year")
            params["year"] = int(year)
        return " AND ".join(clauses), params

    @staticmethod
    def _row_to_hit(row: Any, score: float, branch: str) -> RetrievalHit:
        mapping = row._mapping
        hit = RetrievalHit(
            chunk_id=str(mapping["chunk_id"]),
            document_id=str(mapping["document_id"]),
            source=str(mapping["source"]),
            title=str(mapping["title"]),
            text=str(mapping["text"]),
            section_path=str(mapping["section_path"] or ""),
            url=mapping["url"],
            ord=int(mapping["ord"] or 0),
            external_id=str(mapping["external_id"]) if mapping["external_id"] else None,
            meta=dict(mapping["chunk_meta"] or {}),
            score=float(score or 0.0),
        )
        if branch == "dense":
            hit.dense_score = float(score or 0.0)
        elif branch == "sparse":
            hit.sparse_score = float(score or 0.0)
        else:
            hit.entity_score = float(score or 0.0)
        return hit

    async def dense_search(
        self,
        query_vector: list[float],
        *,
        scopes: list[str],
        sources: list[str] | None = None,
        direction: str | None = None,
        year: int | None = None,
        limit: int = 50,
    ) -> list[RetrievalHit]:
        if not query_vector:
            return []
        where, params = self._base_filters(
            scopes=scopes, sources=sources, direction=direction, year=year
        )
        sql = (
            f"SELECT {_HIT_COLUMNS}, 1 - (c.embedding <=> CAST(:q AS vector)) AS score "
            "FROM rag_chunks c "
            "JOIN rag_documents d ON d.id = c.document_id "
            "JOIN rag_sources s ON s.source = d.source "
            f"WHERE {where} AND c.embedding IS NOT NULL "
            "ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :limit"
        )
        params.update({"q": format_vector(query_vector), "limit": int(limit)})
        stmt = text(sql)
        async with self._ctx() as session:
            rows = await session.execute(stmt, params)
            return [self._row_to_hit(row, row._mapping["score"], "dense") for row in rows]

    async def sparse_search(
        self,
        query_text: str,
        *,
        scopes: list[str],
        sources: list[str] | None = None,
        direction: str | None = None,
        year: int | None = None,
        limit: int = 50,
    ) -> list[RetrievalHit]:
        if not (query_text or "").strip():
            return []
        where, params = self._base_filters(
            scopes=scopes, sources=sources, direction=direction, year=year
        )
        sql = (
            f"SELECT {_HIT_COLUMNS}, "
            "ts_rank_cd(c.tsv, plainto_tsquery('russian', :q)) AS score "
            "FROM rag_chunks c "
            "JOIN rag_documents d ON d.id = c.document_id "
            "JOIN rag_sources s ON s.source = d.source "
            f"WHERE {where} AND c.tsv @@ plainto_tsquery('russian', :q) "
            "ORDER BY score DESC LIMIT :limit"
        )
        params.update({"q": query_text, "limit": int(limit)})
        stmt = text(sql)
        async with self._ctx() as session:
            rows = await session.execute(stmt, params)
            return [self._row_to_hit(row, row._mapping["score"], "sparse") for row in rows]

    async def entity_search(
        self,
        entities: list[tuple[str, str]],
        *,
        scopes: list[str],
        sources: list[str] | None = None,
        direction: str | None = None,
        year: int | None = None,
        limit: int = 20,
    ) -> list[RetrievalHit]:
        if not entities:
            return []
        where, params = self._base_filters(
            scopes=scopes, sources=sources, direction=direction, year=year
        )
        pairs: list[str] = []
        for index, (entity_type, entity_key) in enumerate(entities[:40]):
            pairs.append(f"(:et{index}, :ek{index})")
            params[f"et{index}"] = entity_type
            params[f"ek{index}"] = entity_key
        sql = (
            f"SELECT {_HIT_COLUMNS}, COUNT(*)::float AS score "
            "FROM rag_chunks c "
            "JOIN rag_documents d ON d.id = c.document_id "
            "JOIN rag_sources s ON s.source = d.source "
            "JOIN rag_chunk_entities e ON e.chunk_id = c.id "
            f"WHERE {where} AND (e.entity_type, e.entity_key) IN ({', '.join(pairs)}) "
            "GROUP BY c.id, c.document_id, c.text, c.section_path, c.ord, c.meta, "
            "d.source, d.title, d.url, d.external_id "
            "ORDER BY score DESC LIMIT :limit"
        )
        params["limit"] = int(limit)
        stmt = text(sql)
        async with self._ctx() as session:
            rows = await session.execute(stmt, params)
            return [self._row_to_hit(row, row._mapping["score"], "entity") for row in rows]

    # --------------------------------------------------------------- stats
    async def stats(self) -> dict[str, Any]:
        async with self._ctx() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT d.source, COUNT(DISTINCT d.id) AS docs, COUNT(c.id) AS chunks "
                        "FROM rag_documents d LEFT JOIN rag_chunks c ON c.document_id = d.id "
                        "WHERE d.valid_to IS NULL GROUP BY d.source ORDER BY d.source"
                    )
                )
            ).all()
            sources = (
                await session.execute(
                    text(
                        "SELECT source, active_generation, spec_version, updated_at "
                        "FROM rag_sources ORDER BY source"
                    )
                )
            ).all()
        return {
            "corpus": [
                {"source": str(row[0]), "documents": int(row[1]), "chunks": int(row[2])}
                for row in rows
            ],
            "sources": [
                {
                    "source": str(row[0]),
                    "active_generation": int(row[1]),
                    "spec_version": str(row[2]),
                    "updated_at": row[3].isoformat() if row[3] else None,
                }
                for row in sources
            ],
        }

    # ---------------------------------------------------------------- audit
    async def log_query(
        self,
        *,
        query_hash: str,
        query_len: int,
        intent: str,
        profile: str,
        filters: dict[str, Any],
        hits: int,
        top_score: float,
        latency_ms: int,
        user_id: str | None = None,
        role: str | None = None,
        llm_used: bool | None = None,
        citation_count: int = 0,
    ) -> int | None:
        try:
            async with self._ctx() as session:
                row = (
                    await session.execute(
                        text(
                            """
                            INSERT INTO rag_query_log
                                (user_id, role, profile, query_hash, query_len, intent, filters,
                                 hits, top_score, latency_ms, llm_used, citation_count)
                            VALUES
                                (:user_id, :role, :profile, :query_hash, :query_len, :intent,
                                 CAST(:filters AS jsonb), :hits, :top_score, :latency_ms,
                                 :llm_used, :citation_count)
                            RETURNING id
                            """
                        ),
                        {
                            "user_id": user_id,
                            "role": role,
                            "profile": profile,
                            "query_hash": query_hash,
                            "query_len": query_len,
                            "intent": intent,
                            "filters": _json(filters),
                            "hits": hits,
                            "top_score": top_score,
                            "latency_ms": latency_ms,
                            "llm_used": llm_used,
                            "citation_count": citation_count,
                        },
                    )
                ).first()
            return int(row[0]) if row is not None else None
        except Exception:
            logger.warning("rag_query_log_failed", exc_info=True)
            return None

    async def add_feedback(
        self, *, query_log_id: int | None, user_id: str | None, verdict: int, comment: str = ""
    ) -> None:
        try:
            async with self._ctx() as session:
                await session.execute(
                    text(
                        "INSERT INTO rag_feedback (query_log_id, user_id, verdict, comment) "
                        "VALUES (:q, :u, :v, :c)"
                    ),
                    {"q": query_log_id, "u": user_id, "v": int(verdict), "c": comment[:2000]},
                )
        except Exception:
            logger.warning("rag_feedback_failed", exc_info=True)
