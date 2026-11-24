"""Тесты ингестии публикаций hub: чанкер и загрузчик на фейковом клиенте."""

from __future__ import annotations

import pytest

from src.rag.chunking.publication import chunk_publication
from src.rag.loaders.publications import (
    load_publications_documents,
    publications_to_documents,
)


def _publication(**overrides):
    base = {
        "id": 101,
        "title": "Методы машинного обучения в анализе образовательных данных",
        "abstract": (
            "В работе рассматриваются методы машинного обучения для анализа "
            "образовательных данных: кластеризация, классификация и регрессия. "
            "Приводятся результаты экспериментов на реальных наборах данных."
        ),
        "keywords": ["машинное обучение", "анализ данных", "Python"],
        "authors": ["Иванов И.И.", "Петров П.П."],
        "type_name": "Статья в журнале",
        "journal": "Известия ЮФУ",
        "year": 2024,
        "web_url": "https://hub.sfedu.ru/pub/101/",
        "academic_subject": ["Информатика"],
        "education_program": ["09.03.02"],
    }
    base.update(overrides)
    return base


def test_chunk_publication_builds_sections_and_entities():
    doc = chunk_publication(_publication())
    assert doc is not None
    assert doc.source == "publications"
    assert doc.external_id == "101"
    assert doc.year == 2024
    assert doc.url == "https://hub.sfedu.ru/pub/101/"
    assert doc.meta["partial"] is False
    sections = {chunk.section_path.split(" / ")[-1] for chunk in doc.chunks}
    assert "Аннотация" in sections
    assert "Ключевые слова" in sections
    entity_types = {entity[0] for chunk in doc.chunks for entity in chunk.entities}
    assert "skill" in entity_types
    assert "author" in entity_types


def test_chunk_publication_without_abstract_falls_back_to_card():
    doc = chunk_publication(_publication(abstract="", keywords=[]))
    assert doc is not None
    assert doc.meta["partial"] is True
    text = " ".join(chunk.text for chunk in doc.chunks)
    assert "Публикация:" in text
    assert "Статья в журнале" in text


def test_chunk_publication_skips_records_without_title_or_id():
    assert chunk_publication(_publication(title="")) is None
    assert chunk_publication(_publication(id=None)) is None
    assert chunk_publication({}) is None


def test_publications_to_documents_dedupes_by_id():
    docs = publications_to_documents([_publication(), _publication(), _publication(id=202)])
    assert [doc.external_id for doc in docs] == ["101", "202"]


def test_publications_to_documents_skips_broken_items():
    docs = publications_to_documents([{"id": 1}, {"no_id": True}, _publication(id=303)])
    assert [doc.external_id for doc in docs] == ["303"]


class FakeClient:
    def __init__(self, pages=None, searches=None, fail_pages=(), fail_queries=()):
        self.pages = pages or {}
        self.searches = searches or {}
        self.fail_pages = set(fail_pages)
        self.fail_queries = set(fail_queries)
        self.calls: list[tuple] = []
        self.closed = False

    async def list_publications(self, page=1, page_size=None, type_id=None, creator_id=None):
        self.calls.append(("list", page))
        if page in self.fail_pages:
            raise RuntimeError(f"page {page} failed")
        items = self.pages.get(page, [])
        return {
            "items": items,
            "total": sum(len(v) for v in self.pages.values()),
            "page": page,
            "page_size": len(items),
            "num_pages": max(self.pages) if self.pages else 0,
        }

    async def search_publications(self, query, max_pages=10, type_id=None, creator_id=None):
        self.calls.append(("search", query, max_pages))
        if query in self.fail_queries:
            raise RuntimeError(f"query {query} failed")
        return {"matches": self.searches.get(query, []), "scanned_pages": 1,
                "truncated": False, "total_scanned": 1}

    async def aclose(self):
        self.closed = True


async def test_load_publications_list_mode():
    client = FakeClient(pages={1: [_publication(id=1)], 2: [_publication(id=2)]})
    docs = await load_publications_documents(mode="list", pages=5, client=client)
    assert [doc.external_id for doc in docs] == ["1", "2"]
    assert client.calls == [("list", 1), ("list", 2)]


async def test_load_publications_list_mode_survives_page_failure():
    client = FakeClient(pages={1: [_publication(id=1)], 2: [_publication(id=2)]}, fail_pages={2})
    docs = await load_publications_documents(mode="list", pages=5, client=client)
    assert [doc.external_id for doc in docs] == ["1"]


async def test_load_publications_search_mode():
    client = FakeClient(
        searches={
            "машинное обучение": [_publication(id=10)],
            "python": [_publication(id=10), _publication(id=11)],
        }
    )
    docs = await load_publications_documents(
        mode="search", queries=["машинное обучение", "python"], client=client
    )
    assert [doc.external_id for doc in docs] == ["10", "11"]


async def test_load_publications_search_skips_failed_query():
    client = FakeClient(
        searches={"ok": [_publication(id=20)]},
        fail_queries={"bad"},
    )
    docs = await load_publications_documents(mode="search", queries=["bad", "ok"], client=client)
    assert [doc.external_id for doc in docs] == ["20"]


async def test_load_publications_search_without_queries_is_noop():
    client = FakeClient()
    docs = await load_publications_documents(mode="search", queries=[], client=client)
    assert docs == []
    assert client.calls == []


async def test_load_publications_closes_owned_client(monkeypatch):
    client = FakeClient(pages={1: [_publication(id=1)]})
    monkeypatch.setattr(
        "src.infrastructure.hub_publications.get_client", lambda: client, raising=True
    )
    docs = await load_publications_documents(mode="list", pages=1)
    assert len(docs) == 1
    assert client.closed is True


async def test_load_publications_does_not_close_injected_client():
    client = FakeClient(pages={1: [_publication(id=1)]})
    await load_publications_documents(mode="list", pages=1, client=client)
    assert client.closed is False


@pytest.mark.parametrize("mode", ["list", "search"])
def test_chunk_publication_respects_token_budget(mode):
    long_abstract = " ".join(f"Предложение {i} про методы анализа." for i in range(300))
    doc = chunk_publication(_publication(abstract=long_abstract), max_tokens=200)
    assert doc is not None
    assert len(doc.chunks) > 1
    assert all(chunk.token_count <= 320 for chunk in doc.chunks)
