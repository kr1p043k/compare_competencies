from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
import structlog

from src import Err, Ok, Result
from src.errors import DomainError

logger = structlog.get_logger(__name__)


class RerankerResult:
    __slots__ = ("query", "documents", "scores", "ranked_indices")

    def __init__(
        self,
        query: str,
        documents: list[str],
        scores: list[float],
        ranked_indices: list[int],
    ):
        self.query = query
        self.documents = documents
        self.scores = scores
        self.ranked_indices = ranked_indices

    def top_k(self, k: int = 10) -> list[tuple[str, float]]:
        return [(self.documents[i], self.scores[i]) for i in self.ranked_indices[:k]]


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> Result[RerankerResult, DomainError]:
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> Result[RerankerResult, DomainError]:
        ...

    @property
    def name(self) -> str: ...


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        logger.info("cross_encoder_initialized", model=model_name)

    def _lazy_load(self) -> Result[None, DomainError]:
        if self._model is not None:
            return Ok(None)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
            logger.info("cross_encoder_model_loaded", model=self.model_name)
            return Ok(None)
        except ImportError:
            return Err(DomainError(
                message="sentence-transformers not installed",
                detail="pip install sentence-transformers"
            ))
        except Exception as e:
            return Err(DomainError(
                message="failed to load CrossEncoder model",
                detail=f"{self.model_name}: {e}"
            ))

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> Result[RerankerResult, DomainError]:
        match self._lazy_load():
            case Err(e):
                return Err(e)

        if not documents:
            return Ok(RerankerResult(query=query, documents=[], scores=[], ranked_indices=[]))

        try:
            pairs = [[query, doc] for doc in documents]
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            if isinstance(scores, list) and scores and isinstance(scores[0], list):
                scores = [s[0] for s in scores]
            scores_f = [float(s) for s in scores]

            ranked = sorted(range(len(scores_f)), key=lambda i: scores_f[i], reverse=True)
            if top_k:
                ranked = ranked[:top_k]

            return Ok(RerankerResult(
                query=query,
                documents=documents,
                scores=scores_f,
                ranked_indices=ranked,
            ))
        except Exception as e:
            logger.error("cross_encoder_rerank_failed", error=str(e))
            return Err(DomainError(
                message="reranking failed",
                detail=str(e),
            ))

    @property
    def name(self) -> str:
        return f"CrossEncoder({self.model_name})"


class HybridReranker(BaseReranker):
    def __init__(self, rerankers: list[tuple[BaseReranker, float]]):
        self.rerankers = rerankers
        logger.info("hybrid_reranker_initialized", count=len(rerankers))

    @property
    def name(self) -> str:
        parts = [f"{w:.1f}*{r.name}" for r, w in self.rerankers]
        return f"Hybrid({' + '.join(parts)})"

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> Result[RerankerResult, DomainError]:
        if not documents:
            return Ok(RerankerResult(query=query, documents=[], scores=[], ranked_indices=[]))

        all_scores: dict[str, list[float]] = {doc: [] for doc in documents}

        for ranker, weight in self.rerankers:
            match ranker.rerank(query, documents, top_k=None):
                case Ok(result):
                    score_map = dict(zip(result.documents, result.scores, strict=False))
                    vals = list(score_map.values())
                    vmin, vmax = min(vals), max(vals)
                    if vmax > vmin:
                        for doc in documents:
                            raw = score_map.get(doc, 0.0)
                            all_scores.setdefault(doc, []).append((raw - vmin) / (vmax - vmin) * weight)
                    else:
                        for doc in documents:
                            all_scores.setdefault(doc, []).append(0.5 * weight)
                case Err(e):
                    logger.warning("hybrid_sub_reranker_failed", ranker=ranker.name, error=str(e))

        blended = {doc: sum(scores) for doc, scores in all_scores.items() if scores}
        if not blended:
            return Ok(RerankerResult(query=query, documents=documents, scores=[0.0]*len(documents), ranked_indices=list(range(len(documents)))))

        sorted_docs = sorted(blended, key=blended.get, reverse=True)
        ranked = [documents.index(d) for d in sorted_docs if d in documents]
        if top_k:
            ranked = ranked[:top_k]

        return Ok(RerankerResult(
            query=query,
            documents=documents,
            scores=[blended[d] for d in documents],
            ranked_indices=ranked,
        ))


class RerankerBuilder:
    @staticmethod
    def build_cross_encoder() -> CrossEncoderReranker:
        return CrossEncoderReranker()

    @staticmethod
    def build_hybrid(
        model_names: list[str] | None = None,
        weights: list[float] | None = None,
    ) -> HybridReranker:
        names = model_names or ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
        w = weights or [1.0]
        rerankers = [(CrossEncoderReranker(model_name=n), w[i] if i < len(w) else 1.0) for i, n in enumerate(names)]
        return HybridReranker(rerankers)
