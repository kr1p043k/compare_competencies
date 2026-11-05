"""Backfill missing pgvector embeddings for skills, KSA entries and competencies.

The application matches skills semantically through
``paraphrase-multilingual-mpnet-base-v2`` (768 dims). Rows without an
embedding silently fall back to string matching, so a large NULL share makes
the semantic path dead weight.

Conventions (verified against existing rows by cosine == 1.0000):
  * skills.embedding       <- encode(name)
  * ksa_entries.embedding  <- encode(cleaned_text or original_text)
  * competencies.embedding <- encode(name)

Competency embeddings in the table were produced from a different source text
(cosine ~0.2-0.5 against the name), so they are recomputed wholesale to keep a
single vector space.

Usage:
    python scripts/backfill_embeddings.py --dry-run
    python scripts/backfill_embeddings.py
    python scripts/backfill_embeddings.py --only ksa --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analyzers.comparison.embedding_provider import EmbeddingProviderFactory  # noqa: E402
try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402

DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env

# key -> (table name, text expression, where clause)
TARGETS: dict[str, tuple[str, str, str]] = {
    "skills": ("skills", "name", "embedding is null"),
    "ksa": ("ksa_entries", "coalesce(nullif(cleaned_text, ''), original_text, '')", "embedding is null"),
    "competencies": ("competencies", "name", "true"),
}


def fetch_rows(conn, key: str) -> list[tuple]:
    table, text_expr, where = TARGETS[key]
    sql = f"select id, {text_expr} as body from {table} where {where} order by id"
    with conn.cursor() as cur:
        return cur.execute(sql).fetchall()


def backfill(conn, provider, key: str, batch_size: int, dry_run: bool) -> tuple[int, int]:
    table = TARGETS[key][0]
    rows = fetch_rows(conn, key)
    usable = [(rid, body) for rid, body in rows if body and body.strip()]
    skipped = len(rows) - len(usable)
    print(f"[{key}] rows={len(rows)} with_text={len(usable)} skipped_empty={skipped}")

    if dry_run or not usable:
        return len(usable), skipped

    update_sql = f"update {table} set embedding = %s::vector where id = %s"
    done = 0
    started = time.time()
    for start in range(0, len(usable), batch_size):
        chunk = usable[start : start + batch_size]
        vectors = provider.encode([body for _, body in chunk], show_progress_bar=False)
        with conn.cursor() as cur:
            for (row_id, _), vec in zip(chunk, vectors, strict=True):
                cur.execute(update_sql, ("[" + ",".join(f"{float(x):.6f}" for x in vec) + "]", row_id))
        conn.commit()
        done += len(chunk)
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0
        print(f"[{key}] {done}/{len(usable)} ({rate:.1f}/s)")

    return done, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default="")
    parser.add_argument("--only", choices=sorted(TARGETS), action="append")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)

    tables = args.only or ["skills", "ksa", "competencies"]
    provider = EmbeddingProviderFactory.get()
    dims = getattr(provider, "dimension", None)
    dims = dims() if callable(dims) else dims
    print(f"model={provider.model_version()} dims={dims}")

    totals: dict[str, tuple[int, int]] = {}
    with psycopg.connect(args.dsn, connect_timeout=15) as conn:
        for table in tables:
            totals[table] = backfill(conn, provider, table, args.batch_size, args.dry_run)

    print("\nsummary")
    for table, (done, skipped) in totals.items():
        print(f"  {table:14} embedded={done:5} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
