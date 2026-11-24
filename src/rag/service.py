"""Фасад RAG для прикладного кода (llm_chat, роутеры, CLI).

Скрывает три шага (analyze -> retrieve -> assemble) и гарантирует, что
отсутствие индекса/модели не превращается в исключение у вызывающего.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import structlog

from src.rag import query as query_mod
from src.rag.context import assemble_context
from src.rag.embedder import Embedder
from src.rag.retriever import HybridRetriever
from src.rag.settings import AclScope, RagSettings, get_settings
from src.rag.store import RagStore
from src.rag.types import RagContext, RagQuery

logger = structlog.get_logger(__name__)

#: L1-кэш retrieval: один вопрос в минуту не должен трижды ходить в БД
#: (SSE + панель + повторный клик). Сбрасывается cache_bus-ом после реиндекса.
_CACHE: dict[str, tuple[float, RagResult]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 512
_cache_bus_registered = False


def _cache_key(
    normalized: str,
    profile: str,
    scope_level: str,
    sources: Iterable[str] | None,
    top_n: int | None,
    direction: Any,
    year: Any,
) -> str:
    payload = "|".join(
        [
            normalized,
            profile,
            scope_level,
            ",".join(sorted(sources)) if sources else "*",
            str(top_n or 0),
            str(direction or ""),
            str(year or ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clear_cache() -> int:
    """Сбросить L1-кэш retrieval (вызывается после переиндексации)."""
    with _CACHE_LOCK:
        size = len(_CACHE)
        _CACHE.clear()
    if size:
        logger.info("rag_cache_cleared", entries=size)
    return size


def _ensure_cache_bus() -> None:
    """Подписать сброс кэша на шину: pipeline инвалидирует после реиндекса."""
    global _cache_bus_registered
    if _cache_bus_registered:
        return
    _cache_bus_registered = True
    try:
        from src.cache_bus import register

        register("rag", clear_cache)
    except Exception:
        logger.info("rag_cache_bus_unavailable")


def _cache_get(key: str) -> RagResult | None:
    now = time.time()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        expires, value = entry
        if expires <= now:
            _CACHE.pop(key, None)
            return None
        return value


def _cache_put(key: str, value: RagResult, ttl_s: float) -> None:
    if ttl_s <= 0:
        return
    now = time.time()
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            for stale_key in [k for k, (exp, _) in _CACHE.items() if exp <= now]:
                _CACHE.pop(stale_key, None)
            while len(_CACHE) >= _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)), None)
        _CACHE[key] = (now + ttl_s, value)


@dataclass(slots=True)
class RagResult:
    query: RagQuery
    context: RagContext
    latency_ms: float = 0.0
    degraded: str | None = None
    log_id: int | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def prompt_block(self) -> str:
        return self.context.prompt_block

    def __bool__(self) -> bool:
        return bool(self.context)


@lru_cache(maxsize=1)
def skill_vocabulary() -> tuple[str, ...]:
    """Словарь навыков для entity-linking (it_skills.json). Пустой — не страшно."""
    try:
        from src.parsing.utils import load_it_skills

        payload = load_it_skills()
        names: list[str] = []
        if isinstance(payload, dict):
            names = [str(key) for key in payload]
        elif isinstance(payload, (set, frozenset, list, tuple)):
            names = [str(item) for item in payload]
        return tuple(name for name in names if name)
    except Exception:
        logger.info("rag_skill_vocab_unavailable")
        return ()


class RagService:
    def __init__(
        self,
        *,
        store: RagStore | None = None,
        embedder: Embedder | None = None,
        settings: RagSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store or RagStore()
        self._retriever = HybridRetriever(
            store=self._store, embedder=embedder, settings=self._settings
        )

    @property
    def settings(self) -> RagSettings:
        return self._settings

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    async def search(
        self,
        text: str,
        *,
        role: str | None = None,
        profile: str | None = None,
        sources: Iterable[str] | None = None,
        direction: str | None = None,
        year: int | None = None,
        top_n: int | None = None,
        budget_tokens: int | None = None,
        log: bool = True,
        user_id: str | None = None,
    ) -> RagResult:
        """Полный цикл retrieval. Исключений не бросает."""
        started = time.perf_counter()
        try:
            analyzed = query_mod.analyze(
                text,
                profile=profile,
                skill_vocab=skill_vocabulary(),
                direction=direction,
                year=year,
            )
        except Exception as exc:  # pragma: no cover - защита от кривого ввода
            logger.warning("rag_analyze_failed", error=str(exc))
            analyzed = RagQuery(raw=text or "", normalized=(text or "").strip())

        scope = AclScope.for_role(role)
        source_list = list(sources) if sources is not None else None

        _ensure_cache_bus()
        cache_key = _cache_key(
            analyzed.normalized,
            analyzed.profile,
            scope.level,
            source_list,
            top_n,
            analyzed.filters.get("direction"),
            analyzed.filters.get("year"),
        )
        cached = _cache_get(cache_key)
        if cached is not None:
            try:
                from src.monitoring.metrics import rag_cache_hits

                rag_cache_hits.inc()
            except Exception:
                logger.debug("service_suppressed")
                pass
            return RagResult(
                query=analyzed,
                context=cached.context,
                latency_ms=(time.perf_counter() - started) * 1000,
                degraded=cached.degraded,
                log_id=None,
            )

        try:
            hits, degraded = await self._retriever.retrieve(
                analyzed, scope=scope, sources=source_list, top_n=top_n
            )
        except Exception as exc:
            logger.warning("rag_retrieve_failed", error=str(exc))
            return RagResult(
                query=analyzed,
                context=RagContext(degraded="retrieval_failed"),
                latency_ms=(time.perf_counter() - started) * 1000,
                degraded="retrieval_failed",
                errors=[str(exc)],
            )

        context = assemble_context(
            hits,
            budget_tokens=budget_tokens or self._settings.context_budget,
            evidence_share=self._settings.evidence_share,
            min_score=self._settings.min_score,
        )
        if degraded and not context.degraded:
            context.degraded = degraded
        latency_ms = (time.perf_counter() - started) * 1000

        result = RagResult(
            query=analyzed,
            context=context,
            latency_ms=latency_ms,
            degraded=context.degraded,
        )
        # Кэшируем только непустой результат: «пусто из-за сбоя ветки» не должно
        # залипнуть на минуту.
        if context.evidence:
            _cache_put(cache_key, result, self._settings.cache_ttl_s)
        try:
            from src.monitoring.metrics import rag_top_score, rag_zero_results

            rag_top_score.labels(profile=analyzed.profile).observe(float(context.top_score))
            if not context.evidence:
                rag_zero_results.labels(
                    profile=analyzed.profile, reason=str(context.degraded or "empty")
                ).inc()
        except Exception:
            logger.debug("service_suppressed")
            pass

        log_id = None
        if log:
            log_id = await self._store.log_query(
                query_hash=hashlib.sha256((analyzed.normalized or "").encode("utf-8")).hexdigest(),
                query_len=len(analyzed.raw),
                intent=analyzed.intent,
                profile=analyzed.profile,
                filters=analyzed.filters,
                hits=len(context.evidence),
                top_score=context.top_score,
                latency_ms=int(latency_ms),
                user_id=user_id,
                role=role,
            )
        result.log_id = log_id
        return result

    async def feedback(
        self, *, query_log_id: int | None, user_id: str | None, verdict: int, comment: str = ""
    ) -> None:
        await self._store.add_feedback(
            query_log_id=query_log_id, user_id=user_id, verdict=verdict, comment=comment
        )

    async def stats(self) -> dict[str, Any]:
        return await self._store.stats()


_shared: RagService | None = None


def get_service() -> RagService:
    """Singleton-фасад (модель эмбеддингов и настройки общие)."""
    global _shared
    if _shared is None:
        _shared = RagService()
    return _shared


def reset_service() -> None:
    global _shared
    _shared = None
