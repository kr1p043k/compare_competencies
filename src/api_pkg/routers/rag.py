"""RAG API: отладочный поиск, состав индекса, переиндексация, обратная связь.

Всё за фича-флагом: если ``RAG_ENABLED=false`` эндпоинты поиска отвечают
503 (индекс не обслуживается), но ``/rag/sources`` показывает состояние.
Переиндексация доступна только admin и выполняется синхронно в пуле потоков
(CLI всё равно основной путь для больших корпусов).
"""

from __future__ import annotations

import asyncio
import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from src import config
from src.api_pkg.request_logger import audit_action
from src.api_pkg.routers.auth import require_any_role, require_role, user_error_detail

logger = structlog.get_logger("api")
router = APIRouter(tags=["rag"])
limiter = Limiter(key_func=get_remote_address)

_TEACHER_ROLES = ("admin", "teacher", "rop")


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    profile: str | None = Field(default=None, max_length=64)
    sources: list[str] | None = None
    direction: str | None = Field(default=None, max_length=16)
    year: int | None = None
    top_n: int | None = Field(default=None, ge=1, le=30)


class RagFeedbackRequest(BaseModel):
    query_log_id: int | None = None
    verdict: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=2000)


class RagReindexRequest(BaseModel):
    source: str = Field(default="rpd", pattern="^(rpd|krm|vacancies|all)$")
    reindex: bool = True
    direction: str | None = None
    year: int | None = None
    limit: int | None = Field(default=None, ge=1, le=200000)


class RagCorpusItem(BaseModel):
    source: str
    documents: int
    chunks: int


class RagSourceItem(BaseModel):
    source: str
    active_generation: int
    spec_version: str
    updated_at: str | None = None


class RagSourcesResponse(BaseModel):
    corpus: list[RagCorpusItem]
    sources: list[RagSourceItem]


class RagSearchHit(BaseModel):
    n: int
    source: str
    title: str
    section: str
    url: str | None = None
    score: float
    dense: float
    sparse: float
    entity: float
    text: str


class RagSearchResponse(BaseModel):
    query: str
    intent: str
    profile: str
    entities: dict[str, list[str]]
    degraded: str | None = None
    top_score: float
    latency_ms: float
    log_id: int | None = None
    hits: list[RagSearchHit]
    prompt_block: str


class RagFeedbackResponse(BaseModel):
    ok: bool


class RagIndexStats(BaseModel):
    source: str
    generation: int
    docs_seen: int
    docs_upserted: int
    docs_skipped: int
    docs_removed: int
    chunks_written: int
    dry_run: bool
    errors: list[str]


class RagReindexResponse(BaseModel):
    stats: list[RagIndexStats]


def _rag_enabled() -> bool:
    return bool(getattr(config, "RAG_ENABLED", False))


@router.get("/rag/sources", response_model=RagSourcesResponse)
async def rag_sources(_: dict = Depends(require_any_role(*_TEACHER_ROLES))):
    """Состав корпуса и активные поколения индекса."""
    from src.rag.store import RagStore

    try:
        return await RagStore().stats()
    except Exception as exc:
        logger.error("rag_stats_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Индекс RAG недоступен") from exc


@router.post("/rag/search", response_model=RagSearchResponse)
@limiter.limit("30/minute")
async def rag_search(
    request: Request,
    req: RagSearchRequest,
    user: dict = Depends(require_any_role(*_TEACHER_ROLES)),
):
    """Отладочный поиск: те же ветки и пороги, что в чате, без LLM."""
    if not _rag_enabled():
        raise HTTPException(status_code=503, detail="RAG выключен (RAG_ENABLED=false)")
    from src.rag.service import get_service

    started = time.perf_counter()
    try:
        result = await get_service().search(
            req.query,
            role=str(user.get("r") or ""),
            profile=req.profile,
            sources=req.sources,
            direction=req.direction,
            year=req.year,
            top_n=req.top_n,
            user_id=user.get("uid"),
        )
    except Exception as exc:
        logger.error("rag_search_failed", error=str(exc))
        raise HTTPException(
            status_code=502,
            detail=await user_error_detail(request, str(exc), "Поиск по корпусу недоступен."),
        ) from None

    await audit_action(
        request,
        "rag.search",
        f"intent={result.query.intent} profile={result.query.profile} hits={len(result.context.evidence)}",
    )
    return {
        "query": result.query.raw,
        "intent": result.query.intent,
        "profile": result.query.profile,
        "entities": result.query.entities,
        "degraded": result.degraded,
        "top_score": round(result.context.top_score, 4),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "log_id": result.log_id,
        "hits": [
            {
                "n": index + 1,
                "source": hit.source,
                "title": hit.title,
                "section": hit.section_path,
                "url": hit.url,
                "score": round(hit.score, 4),
                "dense": round(hit.dense_score, 4),
                "sparse": round(hit.sparse_score, 4),
                "entity": round(hit.entity_score, 4),
                "text": hit.text[:800],
            }
            for index, hit in enumerate(result.context.evidence)
        ],
        "prompt_block": result.context.prompt_block,
    }


@router.post("/rag/feedback", response_model=RagFeedbackResponse)
@limiter.limit("30/minute")
async def rag_feedback(
    request: Request,
    req: RagFeedbackRequest,
    user: dict = Depends(require_any_role("admin", "teacher", "rop", "student")),
):
    """Оценка ответа (thumbs up/down) — сырьё для golden-set."""
    from src.rag.service import get_service

    try:
        await get_service().feedback(
            query_log_id=req.query_log_id,
            user_id=user.get("uid"),
            verdict=req.verdict,
            comment=req.comment,
        )
    except Exception as exc:
        logger.error("rag_feedback_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Не удалось сохранить оценку") from exc
    await audit_action(
        request,
        "rag.feedback",
        f"verdict={req.verdict} log={req.query_log_id or '-'}",
    )
    return {"ok": True}


@router.post("/rag/reindex", response_model=RagReindexResponse)
async def rag_reindex(
    request: Request,
    req: RagReindexRequest,
    _: dict = Depends(require_role("admin")),
):
    """Переиндексация источника. Тяжёлая операция: запускать вне пиков."""
    from src.cli.rag_index import index_source

    try:
        stats = await asyncio.wait_for(
            index_source(
                req.source,
                reindex=req.reindex,
                direction=req.direction,
                year=req.year,
                limit=req.limit,
            ),
            timeout=3600,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Переиндексация не уложилась в час") from exc
    except Exception as exc:
        logger.exception("rag_reindex_failed", source=req.source)
        raise HTTPException(
            status_code=500,
            detail=await user_error_detail(request, str(exc), "Переиндексация не удалась."),
        ) from None
    # Индекс обновился: старые цитаты в кэше больше не валидны.
    try:
        from src.cache_bus import invalidate
        from src.rag.service import clear_cache

        clear_cache()
        invalidate("rag")
    except Exception:
        logger.info("rag_cache_invalidate_unavailable")
    await audit_action(
        request,
        "rag.reindex",
        f"source={req.source} reindex={req.reindex} "
        f"docs={sum(item.docs_upserted for item in stats)}",
    )
    return {"stats": [item.as_dict() for item in stats]}
