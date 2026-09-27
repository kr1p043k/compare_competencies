"""Sync DB cache for LLM calls + skill embeddings (graceful degradation).

Every public function uses its own sync SQLAlchemy connection, so there are
no side effects on any caller transaction. All DB errors are logged and
degraded to None -- these helpers never raise.

Tables (see alembic revision add_llm_cache_tables):
  llm_cache(task, model, prompt_hash, prompt_text, response_text, ...)
  skill_embedding_cache(skill_key, model, embedding vector(768), ...)

prompt_hash = hex(sha256(prompt_text)).
Style matches src/services/llm_client.py (structlog) and src/database.py
(cached engine factory) plus Vector(768) dims from src/models/krm_models.py.
"""

import hashlib
from functools import lru_cache
from typing import Any

import structlog
from sqlalchemy import create_engine, text

from src.config import settings

logger = structlog.get_logger(__name__)


def _prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def _sync_database_url() -> str:
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    return url


@lru_cache(maxsize=1)
def _engine():
    return create_engine(
        _sync_database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def _extract_usage(usage: Any) -> tuple[int | None, int | None]:
    try:
        if usage is None:
            return None, None
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
        else:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
        pt = int(pt) if pt is not None else None
        ct = int(ct) if ct is not None else None
        return pt, ct
    except Exception:
        return None, None


def _format_vector(vector: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in vector) + "]"


def _parse_vector(raw: Any) -> list[float] | None:
    try:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            return [float(v) for v in raw]
        s = str(raw).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return None
        return [float(x) for x in s.split(",")]
    except Exception:
        return None


def get_cached(task: str, model: str, prompt_text: str) -> str | None:
    try:
        ph = _prompt_hash(prompt_text)
        with _engine().connect() as conn:
            row = conn.execute(
                text(
                    "SELECT response_text FROM llm_cache "
                    "WHERE task = :task AND model = :model AND prompt_hash = :ph"
                ),
                {"task": task, "model": model, "ph": ph},
            ).fetchone()
        if row is None:
            return None
        return row[0]
    except Exception as exc:
        logger.warning("llm_cache.get_cached_failed", error=str(exc), task=task, model=model)
        return None


def put_cached(
    task: str,
    model: str,
    prompt_text: str,
    response: str,
    usage: Any = None,
) -> None:
    try:
        ph = _prompt_hash(prompt_text)
        prompt_tokens, completion_tokens = _extract_usage(usage)
        with _engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO llm_cache "
                    "(task, model, prompt_hash, prompt_text, response_text, "
                    "prompt_tokens, completion_tokens) "
                    "VALUES (:task, :model, :ph, :pt, :rt, :pto, :cto) "
                    "ON CONFLICT (task, model, prompt_hash) DO NOTHING"
                ),
                {
                    "task": task,
                    "model": model,
                    "ph": ph,
                    "pt": prompt_text,
                    "rt": response,
                    "pto": prompt_tokens,
                    "cto": completion_tokens,
                },
            )
        return None
    except Exception as exc:
        logger.warning("llm_cache.put_cached_failed", error=str(exc), task=task, model=model)
        return None


def get_embedding(skill_key: str, model: str) -> list[float] | None:
    try:
        with _engine().connect() as conn:
            row = conn.execute(
                text(
                    "SELECT embedding::text AS emb FROM skill_embedding_cache "
                    "WHERE skill_key = :k AND model = :m"
                ),
                {"k": skill_key, "m": model},
            ).fetchone()
        if row is None:
            return None
        return _parse_vector(row[0])
    except Exception as exc:
        logger.warning(
            "llm_cache.get_embedding_failed", error=str(exc), skill_key=skill_key, model=model
        )
        return None


def put_embedding(skill_key: str, model: str, vector: list[float]) -> None:
    try:
        if not vector:
            return None
        emb = _format_vector(vector)
        with _engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO skill_embedding_cache "
                    "(skill_key, model, embedding, updated_at) "
                    "VALUES (:k, :m, CAST(:emb AS vector), NOW()) "
                    "ON CONFLICT (skill_key) DO UPDATE SET "
                    "model = EXCLUDED.model, "
                    "embedding = EXCLUDED.embedding, "
                    "updated_at = NOW()"
                ),
                {"k": skill_key, "m": model, "emb": emb},
            )
        return None
    except Exception as exc:
        logger.warning(
            "llm_cache.put_embedding_failed", error=str(exc), skill_key=skill_key, model=model
        )
        return None
