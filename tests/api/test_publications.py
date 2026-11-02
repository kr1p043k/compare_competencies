"""Тесты клиента и эндпоинтов публикаций хаба ЮФУ."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api_pkg.routers import publications as publications_router
from src.infrastructure.hub_publications import (
    HubNotFoundError,
    HubPublicationsClient,
    HubUnavailableError,
    normalize_publication,
)

BASE = "https://hub.sfedu.ru"

RAW_ITEM = {
    "url": "/api/assets/101/",
    "id": 101,
    "creation_datetime": "2026-10-01 12:00:00",
    "creator": {"creator_id": 7, "username": "egorov"},
    "publication_creators": [
        {"pk": 7, "username": "egorov", "last_name": "Егоров", "first_name": "Александр"}
    ],
    "type": {"id": 49, "type_name": "Препринт"},
    "signature_string": "Статья про графы // Егоров Александр",
    "track_id": "801000001",
    "doi": "10.1234/abc",
    "isbn": None,
    "payload": {
        "title": ["Статья про графы"],
        "author": ["Егоров Александр"],
        "fulltext": [
            {
                "url": "/storage/1/101/file/",
                "size": 123,
                "filename": "paper.pdf",
                "content_type": "application/pdf",
            }
        ],
        "academic_subject": ["Математика"],
        "educational_year": ["2025 - 2026"],
    },
}


def _page_response(items: list[dict], page: int = 1, num_pages: int = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json=items,
        headers={
            "Results-Count": str(len(items)),
            "Page-Size": "20",
            "Current-Page": str(page),
            "Num-Pages": str(num_pages),
        },
    )


def test_normalize_publication_flattens_payload():
    item = normalize_publication(RAW_ITEM, BASE)
    assert item["id"] == 101
    assert item["title"] == "Статья про графы"
    assert item["authors"] == ["Егоров Александр"]
    assert item["type_name"] == "Препринт"
    assert item["year"] == 2025
    assert item["academic_subject"] == ["Математика"]
    assert item["files"][0]["url"] == f"{BASE}/storage/1/101/file/"
    assert item["url"] == f"{BASE}/api/assets/101/"


async def test_list_publications_reads_headers_and_filters():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params.multi_items())
        return httpx.Response(
            200,
            json=[RAW_ITEM],
            headers={
                "Results-Count": "1",
                "Page-Size": "20",
                "Current-Page": "2",
                "Num-Pages": "9",
            },
        )

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.list_publications(page=2, page_size=20, type_id=49, creator_id=7)

    assert seen == {"page": "2", "page_size": "20", "type": "49", "creator": "7"}
    assert result["total"] == 1
    assert result["page"] == 2
    assert result["num_pages"] == 9
    assert result["items"][0]["title"] == "Статья про графы"


async def test_search_scans_pages_and_matches_case_insensitively():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        if page == 1:
            return _page_response(
                [
                    RAW_ITEM,
                    {
                        **RAW_ITEM,
                        "id": 102,
                        "signature_string": "Про вакансии // Иванов Иван",
                        "payload": {**RAW_ITEM["payload"], "title": ["Про вакансии"]},
                    },
                ],
                page=1,
                num_pages=2,
            )
        return _page_response([{**RAW_ITEM, "id": 103}], page=2, num_pages=2)

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.search_publications("ГРАФ", max_pages=5)

    assert result["scanned_pages"] == 2
    assert result["truncated"] is False
    assert [item["id"] for item in result["matches"]] == [101, 103]


async def test_get_publication_without_token_scans_pages():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        items = [RAW_ITEM] if page == 2 else []
        return _page_response(items, page=page, num_pages=2)

    client = HubPublicationsClient(base_url=BASE, token="", transport=httpx.MockTransport(handler))
    item = await client.get_publication(101, max_scan_pages=5)

    assert item["id"] == 101
    assert item["doi"] == "10.1234/abc"


async def test_get_publication_with_token_uses_asset_endpoint():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json=RAW_ITEM)

    client = HubPublicationsClient(base_url=BASE, token="abc", transport=httpx.MockTransport(handler))
    item = await client.get_publication(101)

    assert seen["path"] == "/api/assets/101/"
    assert seen["auth"] == "Token abc"
    assert item["title"] == "Статья про графы"


async def test_list_publications_raises_not_found_for_bad_page():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(HubNotFoundError):
        await client.list_publications(page=99999)


class _FakeClient:
    async def list_publications(self, **kwargs):
        return {
            "items": [normalize_publication(RAW_ITEM, BASE)],
            "total": 1,
            "page": 1,
            "page_size": 20,
            "num_pages": 1,
        }

    async def search_publications(self, q, **kwargs):
        return {"matches": [normalize_publication(RAW_ITEM, BASE)], "scanned_pages": 1, "truncated": False}

    async def list_types(self, sample_pages=3):
        return {"types": [{"id": 49, "name": "Препринт", "count": 1}], "scanned_pages": 1, "sample_size": 1}

    async def get_publication(self, publication_id, **kwargs):
        return normalize_publication(RAW_ITEM, BASE)


class _BrokenClient(_FakeClient):
    async def list_publications(self, **kwargs):
        raise HubUnavailableError("down")


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(publications_router, "get_client", lambda: _FakeClient())
    app = FastAPI()
    app.state.limiter = publications_router.limiter
    app.include_router(publications_router.router)
    return TestClient(app)


def test_router_list_publications(api):
    resp = api.get("/publications?page=1&page_size=20")
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"][0]["id"] == 101
    assert body["source"] == "hub.sfedu.ru"


def test_router_search_publications(api):
    resp = api.get("/publications/search?q=граф&limit=1")
    assert resp.status_code == 200
    assert resp.json()["matches"][0]["id"] == 101


def test_router_types(api):
    resp = api.get("/publications/types")
    assert resp.status_code == 200
    assert resp.json()["types"][0]["name"] == "Препринт"


def test_router_detail(api):
    resp = api.get("/publications/101")
    assert resp.status_code == 200
    assert resp.json()["title"] == "Статья про графы"


def test_router_maps_upstream_failure(monkeypatch):
    monkeypatch.setattr(publications_router, "get_client", lambda: _BrokenClient())
    app = FastAPI()
    app.state.limiter = publications_router.limiter
    app.include_router(publications_router.router)
    client = TestClient(app)
    resp = client.get("/publications")
    assert resp.status_code == 503
