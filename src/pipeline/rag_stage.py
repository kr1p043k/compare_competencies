"""Стадия пайплайна: обновление RAG-индекса после сбора/разбора вакансий.

Зачем стадия, а не ночной cron: индекс должен быть согласован с данными,
которые только что записались в БД. Стадия:
  1. инкрементально переиндексирует вакансии (по content_hash, без полного
     пересчёта эмбеддингов);
  2. сбрасывает L1-кэш retrieval через cache_bus, иначе минуту после обновления
     чат отдавал бы старые цитаты;
  3. при ``RAG_ENABLED=false`` не делает ничего и не требует модели эмбеддингов.

Ошибки индексации не роняют пайплайн: рынок пересчитан, а RAG — вторичный
слой (деградация важнее падения).
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src import Ok, Result, config
from src.pipeline.stage import PipelineStage

logger = structlog.get_logger(__name__)


def _run_async(coro_factory):
    """Выполнить async-индексацию из синхронной стадии, не ломая текущий loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro_factory())
    finally:
        loop.close()


class RagIndexStage(PipelineStage):
    name = "rag_index"
    pct_range = (70, 74)

    def __init__(self, args: Any = None) -> None:
        self.args = args

    def run(self, **kwargs) -> Result[dict, Any]:
        from src.monitoring.metrics import track_pipeline_stage

        @track_pipeline_stage("rag_index")
        def _run() -> Result[dict, Any]:
            if not bool(getattr(config, "RAG_ENABLED", False)):
                self._progress(100, "RAG выключен: индексация пропущена")
                return Ok({"rag_enabled": False, "indexed": []})

            limit = getattr(self.args, "rag_limit", None) if self.args else None
            self._progress(5, "Обновление RAG-индекса (вакансии)...")
            try:
                stats = _run_async(lambda: self._index(limit))
            except Exception as exc:
                # Индекс не должен ронять сбор рынка.
                logger.warning("rag_index_stage_failed", error=str(exc))
                self._progress(100, "RAG-индекс не обновлён (продолжаем)")
                return Ok({"rag_enabled": True, "error": str(exc)[:500], "indexed": []})

            self._progress(90, "Сброс кэша retrieval...")
            try:
                from src.rag.service import clear_cache

                clear_cache()
            except Exception:
                logger.info("rag_cache_clear_unavailable")
            try:
                from src.cache_bus import invalidate

                invalidate("rag")
            except Exception:
                logger.info("rag_cache_bus_unavailable")

            payload = [item.as_dict() for item in stats]
            self._progress(100, f"RAG-индекс обновлён: {len(payload)} источник(ов)")
            return Ok({"rag_enabled": True, "indexed": payload})

        return _run()

    async def _index(self, limit: int | None):
        from src.cli.rag_index import index_source

        return await index_source("vacancies", from_db=True, limit=limit)
