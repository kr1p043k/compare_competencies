"""CLI индексации RAG-корпуса.

Usage:
    python -m src.cli rag-index --source rpd [--reindex] [--dry-run]
    python -m src.cli rag-index --source all
    python -m src.cli rag-index --source vacancies --from-db
    python -m src.cli rag-index --stats

Источники:
  rpd         data/reference/sfu_annotations_all.json (полные тексты РПД)
  krm         krm_disciplines_*.json + skill_taxonomy.json (карточки)
  vacancies   таблица vacancies (--from-db) или data/raw/hh_vacancies_basic.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

import structlog

from src import config
from src.rag import index_documents
from src.rag.chunking import iter_krm_cards, iter_rpd_documents, iter_skill_cards
from src.rag.chunking.vacancy import chunk_vacancy
from src.rag.indexer import IndexStats
from src.rag.store import RagStore
from src.rag.types import Document

logger = structlog.get_logger(__name__)

SOURCES = ("rpd", "krm", "vacancies")


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_rpd_documents(direction: str | None = None, year: int | None = None) -> list[Document]:
    path = config.REFERENCE_DIR / "sfu_annotations_all.json"
    if not path.exists():
        logger.warning("rag_rpd_source_missing", path=str(path))
        return []
    payload = _read_json(path)
    return list(
        iter_rpd_documents(
            payload,
            direction=direction,
            year=year,
            max_tokens=config.RAG_MAX_CHUNK_TOKENS,
            min_tokens=config.RAG_MIN_CHUNK_TOKENS,
        )
    )


def load_krm_documents(direction: str | None = None) -> list[Document]:
    docs: list[Document] = []
    krm_path = config.KRM_DISCIPLINES_PATH
    if krm_path.exists():
        payload = _read_json(krm_path)
        docs.extend(iter_krm_cards(payload, direction=direction))
    else:
        logger.warning("rag_krm_source_missing", path=str(krm_path))
    taxonomy_path = config.SKILL_TAXONOMY_PATH
    if taxonomy_path.exists():
        docs.extend(iter_skill_cards(_read_json(taxonomy_path)))
    return docs


def load_vacancy_documents_from_json(path: Path | None = None) -> list[Document]:
    path = path or (config.DATA_RAW_DIR / "hh_vacancies_basic.json")
    if not path.exists():
        logger.warning("rag_vacancies_source_missing", path=str(path))
        return []
    payload = _read_json(path)
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    docs: list[Document] = []
    for vacancy in items:
        if not isinstance(vacancy, dict):
            continue
        doc = chunk_vacancy(
            vacancy,
            max_tokens=config.RAG_MAX_CHUNK_TOKENS,
            min_tokens=config.RAG_MIN_CHUNK_TOKENS,
        )
        if doc is not None:
            docs.append(doc)
    return docs


async def load_vacancy_documents_from_db(limit: int | None = None) -> list[Document]:
    from sqlalchemy import text

    from src.database import async_session_factory

    sql = (
        "SELECT hh_id, name, employer_name, area_name, experience, snippet_requirement, "
        "snippet_responsibility, description, key_skills, parsed_skills, published_at, alternate_url "
        "FROM vacancies WHERE description IS NOT NULL OR snippet_requirement IS NOT NULL "
        "ORDER BY published_at DESC NULLS LAST"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    docs: list[Document] = []
    async with async_session_factory() as session:
        rows = await session.execute(text(sql))
        for row in rows:
            vacancy = {
                "hh_id": row[0],
                "name": row[1],
                "employer_name": row[2],
                "area_name": row[3],
                "experience": row[4],
                "snippet_requirement": row[5],
                "snippet_responsibility": row[6],
                "description": row[7],
                "key_skills": row[8],
                "parsed_skills": row[9],
                "published_at": row[10],
                "alternate_url": row[11],
            }
            doc = chunk_vacancy(
                vacancy,
                max_tokens=config.RAG_MAX_CHUNK_TOKENS,
                min_tokens=config.RAG_MIN_CHUNK_TOKENS,
            )
            if doc is not None:
                docs.append(doc)
    return docs


async def index_source(
    source: str,
    *,
    reindex: bool = False,
    dry_run: bool = False,
    from_db: bool = False,
    direction: str | None = None,
    year: int | None = None,
    limit: int | None = None,
) -> list[IndexStats]:
    stats: list[IndexStats] = []
    if source in ("rpd", "all"):
        docs = load_rpd_documents(direction=direction, year=year)
        stats.append(
            await index_documents(
                docs, source="rpd", full_reindex=reindex, dry_run=dry_run
            )
        )
    if source in ("krm", "all"):
        docs = load_krm_documents(direction=direction)
        stats.append(
            await index_documents(
                docs, source="krm", full_reindex=reindex, dry_run=dry_run
            )
        )
    if source in ("vacancies", "all"):
        docs = (
            await load_vacancy_documents_from_db(limit=limit)
            if from_db
            else load_vacancy_documents_from_json()
        )
        if limit and not from_db:
            docs = docs[:limit]
        stats.append(
            await index_documents(
                docs, source="vacancies", full_reindex=reindex, dry_run=dry_run
            )
        )
    return stats


async def print_stats() -> None:
    data = await RagStore().stats()
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(args: Any) -> None:
    if getattr(args, "stats", False):
        asyncio.run(print_stats())
        return
    source = getattr(args, "source", "all")
    if source not in (*SOURCES, "all"):
        raise SystemExit(f"unknown source: {source}")
    results = asyncio.run(
        index_source(
            source,
            reindex=bool(getattr(args, "reindex", False)),
            dry_run=bool(getattr(args, "dry_run", False)),
            from_db=bool(getattr(args, "from_db", False)),
            direction=getattr(args, "direction", None),
            year=getattr(args, "year", None),
            limit=getattr(args, "limit", None),
        )
    )
    for item in results:
        print(json.dumps(item.as_dict(), ensure_ascii=False))


def build_parser(sub) -> None:
    parser = sub.add_parser("rag-index", help="Проиндексировать корпус RAG (РПД/КРМ/вакансии)")
    parser.add_argument("--source", default="all", choices=[*SOURCES, "all"])
    parser.add_argument("--reindex", action="store_true", help="Полный реиндекс в новое поколение")
    parser.add_argument("--dry-run", action="store_true", help="Только посчитать документы/чанки")
    parser.add_argument("--from-db", action="store_true", help="Вакансии из PostgreSQL, не из JSON")
    parser.add_argument("--direction", default=None, help="Код направления (09.03.02)")
    parser.add_argument("--year", type=int, default=None, help="Год набора")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число вакансий")
    parser.add_argument("--stats", action="store_true", help="Показать состав индекса и выйти")
    parser.set_defaults(func=main)
