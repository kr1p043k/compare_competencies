"""Гибридный retrieval: dense + FTS + entity -> RRF -> дедуп -> MMR.

Fusion и отбор — чистые функции (тестируются без БД и модели). Сетевые
ветки собираются в ``HybridRetriever`` с общим таймаутом: retrieval никогда
не должен держать чат дольше ``RAG_TIMEOUT_S``.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Iterable

import structlog

from src.rag import query as query_mod
from src.rag.embedder import Embedder
from src.rag.settings import AclScope, RagSettings, get_settings
from src.rag.store import RagStore
from src.rag.types import RagQuery, RetrievalHit

logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-zа-яё0-9+#._\-]{2,}", re.IGNORECASE)

_BRANCH_WEIGHTS = {"dense": 1.0, "sparse": 1.0, "entity": 1.0}

#: Бонус за лексическое совпадение запроса с заголовком документа.
#: Dense-модель ранжирует «похоже по смыслу» и нередко ставит выше документ
#: с другим названием; если запрос почти дословно равен заголовку, это
#: сильнее семантики (замер - docs/RAG-EVAL.md).
_TITLE_BOOST = 0.35
_TITLE_MIN_COVERAGE = 0.6
_TITLE_MIN_TERM_LEN = 3


def _token_prefix_match(left: str, right: str) -> bool:
    """Совпадение с учётом словоформ: «моделирования» ~ «моделирование»."""
    if left == right:
        return True
    limit = min(len(left), len(right), 5)
    return limit >= 4 and left[:limit] == right[:limit]


def title_coverage(title: str, query_text: str) -> float:
    """Доля слов запроса, встречающихся в заголовке (с учётом словоформ)."""
    terms = {token for token in _tokens(query_text) if len(token) >= _TITLE_MIN_TERM_LEN}
    title_tokens = _tokens(title)
    if not terms or not title_tokens:
        return 0.0
    matched = sum(
        1
        for term in terms
        if any(_token_prefix_match(term, title_token) for title_token in title_tokens)
    )
    return matched / len(terms)


def apply_title_boost(
    hits: list[RetrievalHit], query_text: str, *, boost: float = _TITLE_BOOST
) -> list[RetrievalHit]:
    """Поднять документы, чей заголовок лексически совпадает с запросом."""
    if not hits or boost <= 0 or not (query_text or "").strip():
        return hits
    for hit in hits:
        coverage = title_coverage(hit.title, query_text)
        if coverage >= _TITLE_MIN_COVERAGE:
            hit.title_score = round(coverage, 4)
            hit.score += boost * coverage
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


def dedupe(hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    """Убрать одинаковые тексты (один абзац может прийти из двух источников)."""
    best: dict[str, RetrievalHit] = {}
    for hit in hits:
        key = hit.dedup_key
        current = best.get(key)
        if current is None or hit.score > current.score:
            best[key] = hit
    return list(best.values())


def rrf_fuse(
    branches: dict[str, list[RetrievalHit]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[RetrievalHit]:
    """Reciprocal Rank Fusion: score = Σ w_branch / (k + rank).

    Складываем по chunk_id: одна и та же чанка, найденная обеими ветками,
    поднимается выше. Ветки с пустым результатом просто не дают вклада.
    """
    weights = {**_BRANCH_WEIGHTS, **(weights or {})}
    fused: dict[str, RetrievalHit] = {}
    scores: dict[str, float] = {}

    for branch, hits in branches.items():
        weight = float(weights.get(branch, 1.0))
        if weight <= 0:
            continue
        for rank, hit in enumerate(hits):
            key = hit.chunk_id
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
            existing = fused.get(key)
            if existing is None:
                fused[key] = hit
            else:
                existing.dense_score = max(existing.dense_score, hit.dense_score)
                existing.sparse_score = max(existing.sparse_score, hit.sparse_score)
                existing.entity_score = max(existing.entity_score, hit.entity_score)

    if not scores:
        return []
    top = max(scores.values())
    out: list[RetrievalHit] = []
    for key, score in scores.items():
        hit = fused[key]
        hit.score = score / top if top else 0.0
        out.append(hit)
    out.sort(key=lambda item: item.score, reverse=True)
    return out


def hit_confidence(hit: RetrievalHit) -> float:
    """Абсолютная уверенность (не зависит от размера выдачи).

    Dense cosine — калиброванная величина; FTS-ранги маленькие, поэтому
    приводим их к шкале; entity-совпадение само по себе уже сильный сигнал.
    Заголовок, почти дословно совпавший с запросом, тоже считается сильным
    сигналом: иначе документ с нужным названием отсекался бы порогом
    ``min_score`` только из-за невысокого косинуса.
    """
    if hit.dense_score > 0:
        base = min(1.0, hit.dense_score)
    elif hit.entity_score > 0:
        base = min(1.0, 0.5 + 0.1 * hit.entity_score)
    elif hit.sparse_score > 0:
        base = min(1.0, hit.sparse_score * 4.0)
    else:
        base = 0.0
    return min(1.0, max(base, hit.title_score * 0.8))


def aggregate_by_document(
    hits: list[RetrievalHit], *, max_per_document: int = 3
) -> list[RetrievalHit]:
    """Не более N чанков на документ, чтобы один PDF не занял всю выдачу."""
    counts: dict[str, int] = {}
    out: list[RetrievalHit] = []
    for hit in hits:
        used = counts.get(hit.document_id, 0)
        if used >= max_per_document:
            continue
        counts[hit.document_id] = used + 1
        out.append(hit)
    return out


def mmr_select(
    hits: list[RetrievalHit],
    *,
    top_n: int,
    lambda_: float = 0.7,
) -> list[RetrievalHit]:
    """Maximal Marginal Relevance на токенных множествах (без эмбеддингов).

    lambda_=1 — чистый порядок релевантности, 0 — максимальное разнообразие.
    Дешевле векторного MMR и достаточно, чтобы не отдавать пять абзацев
    одного раздела.
    """
    if not hits or top_n <= 0:
        return []
    candidates = list(hits)
    token_sets = {hit.chunk_id: _tokens(hit.text) for hit in candidates}
    selected: list[RetrievalHit] = []
    while candidates and len(selected) < top_n:
        best_index = 0
        best_value = float("-inf")
        for index, hit in enumerate(candidates):
            relevance = hit.score
            if selected:
                overlap = 0.0
                own = token_sets[hit.chunk_id]
                for chosen in selected:
                    other = token_sets[chosen.chunk_id]
                    if not own or not other:
                        continue
                    overlap = max(overlap, len(own & other) / max(1, len(own | other)))
            else:
                overlap = 0.0
            value = lambda_ * relevance - (1.0 - lambda_) * overlap
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(candidates.pop(best_index))
    return selected


class HybridRetriever:
    """Три ветки параллельно, общий бюджет времени, детерминированный fusion."""

    def __init__(
        self,
        *,
        store: RagStore | None = None,
        embedder: Embedder | None = None,
        settings: RagSettings | None = None,
    ) -> None:
        self._store = store or RagStore()
        self._embedder = embedder
        self._settings = settings or get_settings()

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            from src.rag.embedder import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    async def retrieve(
        self,
        query: RagQuery,
        *,
        scope: AclScope | None = None,
        sources: list[str] | None = None,
        top_n: int | None = None,
    ) -> tuple[list[RetrievalHit], str | None]:
        """Вернуть (hits, degraded_reason). Никогда не бросает."""
        started = time.perf_counter()
        settings = self._settings
        profile = settings.profile(query.profile)
        scope = scope or AclScope()
        effective_sources = sources if sources is not None else profile.get("sources")
        if effective_sources is not None:
            effective_sources = list(effective_sources)

        dense_weight = float(profile.get("dense_weight", 0.6))
        sparse_weight = float(profile.get("sparse_weight", 0.4))
        entity_boost = float(profile.get("entity_boost", 0.15))
        limit_top = int(top_n or profile.get("top_n", settings.top_n))

        direction = query.filters.get("direction")
        year = query.filters.get("year")

        degraded: str | None = None
        dense_hits: list[RetrievalHit] = []
        sparse_hits: list[RetrievalHit] = []
        entity_hits: list[RetrievalHit] = []

        dense_task = None
        if dense_weight > 0 and query.normalized:
            dense_task = asyncio.create_task(
                self._dense(query, scope=scope, sources=effective_sources,
                            direction=direction, year=year)
            )
        sparse_task = None
        sparse_text = query_mod.sparse_query_text(query)
        if sparse_weight > 0 and sparse_text:
            sparse_task = asyncio.create_task(
                self._store.sparse_search(
                    sparse_text,
                    scopes=list(scope.allowed),
                    sources=effective_sources,
                    direction=direction,
                    year=year,
                    limit=settings.sparse_k,
                )
            )
        entity_task = None
        pairs = query_mod.entity_pairs(query)
        if entity_boost > 0 and pairs:
            entity_task = asyncio.create_task(
                self._store.entity_search(
                    pairs,
                    scopes=list(scope.allowed),
                    sources=effective_sources,
                    direction=direction,
                    year=year,
                    limit=settings.entity_k,
                )
            )

        async def _collect(name: str, task) -> list[RetrievalHit]:
            if task is None:
                return []
            try:
                return await asyncio.wait_for(task, timeout=settings.timeout_s)
            except TimeoutError:
                logger.warning("rag_branch_timeout", branch=name)
                return []
            except Exception as exc:
                logger.warning("rag_branch_failed", branch=name, error=str(exc))
                return []

        dense_hits, sparse_hits, entity_hits = await asyncio.gather(
            _collect("dense", dense_task),
            _collect("sparse", sparse_task),
            _collect("entity", entity_task),
        )

        if dense_weight > 0 and not dense_hits:
            degraded = "dense_unavailable" if not self._embedder_available() else "dense_empty"

        branches = {
            "dense": dense_hits,
            "sparse": sparse_hits,
            "entity": entity_hits,
        }
        fused = rrf_fuse(
            branches,
            k=settings.rrf_k,
            weights={"dense": dense_weight, "sparse": sparse_weight, "entity": entity_boost},
        )
        fused = apply_title_boost(
            fused, query.normalized or query.raw, boost=settings.title_boost
        )
        fused = dedupe(fused)
        fused = aggregate_by_document(fused, max_per_document=settings.max_per_document)
        selected = mmr_select(fused, top_n=limit_top, lambda_=settings.mmr_lambda)

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "rag_retrieve",
            profile=query.profile,
            dense=len(dense_hits),
            sparse=len(sparse_hits),
            entity=len(entity_hits),
            selected=len(selected),
            latency_ms=round(elapsed, 1),
            degraded=degraded,
        )
        self._observe(query.profile, selected, elapsed)
        return selected, degraded

    @staticmethod
    def _observe(profile: str, selected: list[RetrievalHit], elapsed_ms: float) -> None:
        """Метрики Prometheus: тихо, если registry недоступен."""
        try:
            from src.monitoring.metrics import rag_hits, rag_retrieval_duration

            rag_retrieval_duration.labels(profile=profile).observe(elapsed_ms / 1000)
            for hit in selected:
                rag_hits.labels(source=hit.source, profile=profile).inc()
        except Exception:
            pass

    def _embedder_available(self) -> bool:
        try:
            return self.embedder.available
        except Exception:
            return False

    async def _dense(
        self,
        query: RagQuery,
        *,
        scope: AclScope,
        sources: list[str] | None,
        direction: str | None,
        year: int | None,
    ) -> list[RetrievalHit]:
        vector = await self.embedder.encode_one(query.normalized)
        if not vector:
            return []
        return await self._store.dense_search(
            vector,
            scopes=list(scope.allowed),
            sources=sources,
            direction=direction,
            year=year,
            limit=self._settings.dense_k,
        )
