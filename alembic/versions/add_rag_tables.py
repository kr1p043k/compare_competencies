"""Add RAG tables (documents/chunks/entities/runs/query log/feedback).

Revision ID: add_rag_tables
Revises: add_admin_totp_last_step, vacancy_source

Хранение корпуса для retrieval: pgvector (768) + Postgres FTS + entity-слой.
Поколения (generation) дают blue-green реиндекс: пока строится новое, старое
продолжает обслуживать запросы, переключение — одна строка в rag_sources.

Ревизия сводит две ветки графа (add_admin_totp_last_step, vacancy_source),
чтобы после неё остался один head.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "add_rag_tables"
down_revision: tuple[str, ...] | str = ("add_admin_totp_last_step", "vacancy_source")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_sources (
            source            TEXT PRIMARY KEY,
            spec_version      TEXT NOT NULL,
            acl_scope         TEXT NOT NULL DEFAULT 'internal',
            active_generation INTEGER NOT NULL DEFAULT 1,
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_documents (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source        TEXT NOT NULL,
            external_id   TEXT NOT NULL,
            generation    INTEGER NOT NULL DEFAULT 1,
            title         TEXT NOT NULL,
            url           TEXT,
            lang          TEXT NOT NULL DEFAULT 'ru',
            content_hash  CHAR(64) NOT NULL,
            model_version TEXT NOT NULL DEFAULT '',
            acl_scope     TEXT NOT NULL DEFAULT 'internal',
            direction     TEXT,
            year          INTEGER,
            discipline_id UUID,
            published_at  TIMESTAMPTZ,
            valid_from    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            valid_to      TIMESTAMPTZ,
            meta          JSONB NOT NULL DEFAULT '{}'::jsonb,
            UNIQUE (source, external_id, generation)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id  UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
            generation   INTEGER NOT NULL DEFAULT 1,
            ord          INTEGER NOT NULL DEFAULT 0,
            section_path TEXT,
            text         TEXT NOT NULL,
            token_count  INTEGER NOT NULL DEFAULT 0,
            embedding    VECTOR(768),
            tsv          TSVECTOR,
            meta         JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_chunk_entities (
            chunk_id    UUID NOT NULL REFERENCES rag_chunks(id) ON DELETE CASCADE,
            entity_type TEXT NOT NULL,
            entity_key  TEXT NOT NULL,
            PRIMARY KEY (chunk_id, entity_type, entity_key)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_ingest_runs (
            id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source         TEXT NOT NULL,
            generation     INTEGER NOT NULL,
            started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at    TIMESTAMPTZ,
            docs_seen      INTEGER NOT NULL DEFAULT 0,
            docs_upserted  INTEGER NOT NULL DEFAULT 0,
            docs_skipped   INTEGER NOT NULL DEFAULT 0,
            docs_removed   INTEGER NOT NULL DEFAULT 0,
            chunks_written INTEGER NOT NULL DEFAULT 0,
            status         TEXT NOT NULL DEFAULT 'running',
            error          TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_query_log (
            id             BIGSERIAL PRIMARY KEY,
            user_id        TEXT,
            role           TEXT,
            profile        TEXT,
            query_hash     TEXT,
            query_len      INTEGER,
            intent         TEXT,
            filters        JSONB NOT NULL DEFAULT '{}'::jsonb,
            hits           INTEGER NOT NULL DEFAULT 0,
            top_score      REAL,
            latency_ms     INTEGER,
            llm_used       BOOLEAN,
            citation_count INTEGER NOT NULL DEFAULT 0,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_feedback (
            id           BIGSERIAL PRIMARY KEY,
            query_log_id BIGINT REFERENCES rag_query_log(id) ON DELETE SET NULL,
            user_id      TEXT,
            verdict      SMALLINT NOT NULL,
            comment      TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_documents_source "
        "ON rag_documents (source, generation, acl_scope)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_documents_scope "
        "ON rag_documents (direction, year)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_chunks_generation ON rag_chunks (generation)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_chunks_document ON rag_chunks (document_id, ord)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_rag_chunks_tsv ON rag_chunks USING gin (tsv)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_chunk_entities_key "
        "ON rag_chunk_entities (entity_type, entity_key)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_query_log_created ON rag_query_log (created_at)"
    )
    # HNSW строится и на пустой таблице; при реиндексе индекс не пересоздаётся.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_chunks_embedding "
        "ON rag_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS rag_feedback")
    op.execute("DROP TABLE IF EXISTS rag_query_log")
    op.execute("DROP TABLE IF EXISTS rag_ingest_runs")
    op.execute("DROP TABLE IF EXISTS rag_chunk_entities")
    op.execute("DROP TABLE IF EXISTS rag_chunks")
    op.execute("DROP TABLE IF EXISTS rag_documents")
    op.execute("DROP TABLE IF EXISTS rag_sources")
