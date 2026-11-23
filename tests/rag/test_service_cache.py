"""L1-кэш retrieval: попадания, TTL, инвалидация, ключ по ACL."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag import service as service_mod
from src.rag.settings import AclScope, RagSettings
from src.rag.types import Citation, RagContext, RetrievalHit


class CountingRag(service_mod.RagService):
    """Реальный сервис с подменённым retriever: считаем обращения к БД."""

    def __init__(self, *, empty: bool = False):
        self.calls: list[AclScope] = []
        self.empty = empty
        settings = RagSettings(enabled=True, cache_ttl_s=60.0)
        self._settings = settings
        self._store = SimpleNamespace(log_query=self._log)
        self._retriever = self

    async def _log(self, **kwargs):
        return 1

    async def retrieve(self, query, *, scope, sources=None, top_n=None):
        self.calls.append(scope)
        if self.empty:
            return [], None
        hit = RetrievalHit(
            chunk_id="c1",
            document_id="d1",
            source="rpd",
            title="РПД",
            text="УК-3 формирует работу в команде.",
            dense_score=0.9,
        )
        return [hit], None


@pytest.fixture(autouse=True)
def _clean_cache():
    service_mod.clear_cache()
    yield
    service_mod.clear_cache()


async def test_second_identical_search_uses_cache():
    rag = CountingRag()
    first = await rag.search("почему УК-3 не покрыт", role="teacher", log=False)
    second = await rag.search("почему УК-3 не покрыт", role="teacher", log=False)
    assert len(rag.calls) == 1
    assert second.context is first.context


async def test_different_role_is_not_served_from_cache():
    rag = CountingRag()
    await rag.search("вопрос про Python", role="teacher", log=False)
    await rag.search("вопрос про Python", role="student", log=False)
    assert len(rag.calls) == 2
    assert rag.calls[0].level != rag.calls[1].level


async def test_empty_result_is_not_cached():
    rag = CountingRag(empty=True)
    await rag.search("пустой запрос", role="teacher", log=False)
    await rag.search("пустой запрос", role="teacher", log=False)
    assert len(rag.calls) == 2


async def test_clear_cache_forces_new_retrieval():
    rag = CountingRag()
    await rag.search("вопрос", role="teacher", log=False)
    service_mod.clear_cache()
    await rag.search("вопрос", role="teacher", log=False)
    assert len(rag.calls) == 2


async def test_zero_ttl_disables_cache():
    rag = CountingRag()
    rag._settings = RagSettings(enabled=True, cache_ttl_s=0.0)
    await rag.search("вопрос", role="teacher", log=False)
    await rag.search("вопрос", role="teacher", log=False)
    assert len(rag.calls) == 2


async def test_cached_result_keeps_citations():
    rag = CountingRag()
    first = await rag.search("почему УК-3 не покрыт", role="teacher", log=False)
    second = await rag.search("почему УК-3 не покрыт", role="teacher", log=False)
    assert first.context.citations
    assert second.context.citations[0].n == 1
    assert isinstance(second.context, RagContext)
    assert isinstance(second.context.citations[0], Citation)
