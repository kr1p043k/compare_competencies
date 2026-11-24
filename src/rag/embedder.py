"""Эмбеддинги для RAG: singleton-модель, батчи, async-обёртка.

Синхронный ``model.encode`` блокирует event loop (torch на CPU, 100-300 мс на
батч), поэтому наружу торчит только ``encode_async`` через
``asyncio.to_thread``. Модель та же, что в остальном проекте
(``paraphrase-multilingual-mpnet-base-v2``, 768) — иначе скиллы и чанки жили
бы в разных семантических пространствах.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import structlog

from src import config

logger = structlog.get_logger(__name__)

#: Сериализация инференса: параллельные вызовы torch дают джиттер float.
_ENCODE_LOCK = threading.Lock()

DIMENSION = 768


class EmbeddingUnavailableError(RuntimeError):
    """Модель не загрузилась (нет весов, нет сети при первом скачивании)."""


def model_version() -> str:
    """Версия модели для инвалидации индекса: имя + версия библиотеки."""
    try:
        import sentence_transformers

        lib = sentence_transformers.__version__
    except Exception:
        logger.debug("embedder_suppressed")
        lib = "unknown"
    return f"{config.EMBEDDING_MODEL}_st{lib}"


class Embedder:
    """Ленивая обёртка над sentence-transformers с батчингом."""

    def __init__(self, *, batch_size: int = 64, model_name: str | None = None) -> None:
        self._model_name = model_name or config.EMBEDDING_MODEL
        self._batch_size = max(1, int(batch_size))
        self._model: Any | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- model
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from src.parsing.api.embedding_loader import get_embedding_model

                model = get_embedding_model(self._model_name)
                model.eval()
                self._model = model
                self._load_error = None
                logger.info("rag_embedder_loaded", model=self._model_name)
            except Exception as exc:  # pragma: no cover - зависит от окружения
                self._load_error = str(exc)
                logger.error("rag_embedder_load_failed", model=self._model_name, error=str(exc))
                raise EmbeddingUnavailableError(str(exc)) from exc
        return self._model

    @property
    def available(self) -> bool:
        try:
            self._ensure_model()
            return True
        except EmbeddingUnavailableError:
            return False

    @property
    def dimension(self) -> int:
        return DIMENSION

    @property
    def version(self) -> str:
        return model_version()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # --------------------------------------------------------------- encode
    def encode_sync(self, texts: list[str]) -> list[list[float]]:
        """Синхронный путь (CLI, индексация). Пустой вход — пустой выход."""
        if not texts:
            return []
        model = self._ensure_model()
        out: list[list[float]] = []
        with _ENCODE_LOCK:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                vectors = model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=self._batch_size,
                )
                out.extend([list(map(float, row)) for row in vectors])
        return out

    async def encode_async(self, texts: list[str]) -> list[list[float]]:
        """Async-путь (API): encode уходит в пул потоков, loop свободен."""
        if not texts:
            return []
        return await asyncio.to_thread(self.encode_sync, texts)

    async def encode_one(self, text: str) -> list[float]:
        vectors = await self.encode_async([text])
        return vectors[0] if vectors else []


_shared: Embedder | None = None
_shared_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Общий экземпляр: модель ~1 ГБ, второй экземпляр не нужен."""
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = Embedder(batch_size=int(getattr(config, "RAG_INDEX_BATCH", 64) or 64))
    return _shared


def reset_embedder() -> None:
    global _shared
    _shared = None


def format_vector(vector: list[float]) -> str:
    """pgvector-литерал '[0.1,0.2,...]' (без numpy-зависимостей)."""
    return "[" + ",".join(f"{float(value):.7f}" for value in vector) + "]"
