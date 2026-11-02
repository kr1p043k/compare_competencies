"""Тесты клиента и эндпоинтов публикаций хаба ЮФУ."""

from __future__ import annotations

import contextlib

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api_pkg.routers import publications as publications_router
from src.infrastructure import hub_publications as hub_infra
from src.infrastructure.hub_publications import (
    HubAuthRequiredError,
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
        "review_fulltext": [
            {"url": "/storage/1/101/review/", "size": 10, "filename": "review.pdf"},
        ],
        "academic_subject": ["Математика"],
        "education_program": ["01.03.01 Математика"],
        "educational_year": ["2025 - 2026"],
    },
}


def _page(
    items: list[dict],
    page: int = 1,
    num_pages: int = 1,
    page_size: int = 20,
    total: int | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json=items,
        headers={
            "Results-Count": str(total if total is not None else len(items)),
            "Page-Size": str(page_size),
            "Current-Page": str(page),
            "Num-Pages": str(num_pages),
        },
    )


def _item(item_id: int, title: str) -> dict:
    return {
        **RAW_ITEM,
        "id": item_id,
        "signature_string": f"{title} // Егоров Александр",
        "payload": {**RAW_ITEM["payload"], "title": [title]},
    }


@pytest.fixture(autouse=True)
def reset_limiter():
    with contextlib.suppress(Exception):
        publications_router.limiter._storage.reset()
    yield


def test_normalize_publication_flattens_payload():
    item = normalize_publication(RAW_ITEM, BASE)
    assert item["id"] == 101
    assert item["title"] == "Статья про графы"
    assert item["authors"] == ["Егоров Александр"]
    assert item["type_name"] == "Препринт"
    assert item["year"] == 2025
    assert item["academic_subject"] == ["Математика"]
    assert item["education_program"] == ["01.03.01 Математика"]
    assert [f["filename"] for f in item["files"]] == ["paper.pdf", "review.pdf"]
    assert item["files"][0]["url"] == f"{BASE}/storage/1/101/file/"
    assert item["url"] == f"{BASE}/api/assets/101/"


def test_normalize_publication_tolerates_empty_fields():
    item = normalize_publication({"id": "abc", "payload": None, "type": None, "creator": None}, BASE)
    assert item["id"] == 0
    assert item["title"] is None
    assert item["authors"] == []
    assert item["files"] == []
    assert item["year"] is None
    assert item["url"] is None


def test_normalize_publication_falls_back_to_creation_year():
    raw = {
        "id": 1,
        "creation_datetime": "2024-05-01 10:00:00",
        "payload": {"title": ["Без учебного года"]},
    }
    assert normalize_publication(raw, BASE)["year"] == 2024


def test_normalize_publication_drops_foreign_file_hosts():
    raw = {
        **RAW_ITEM,
        "payload": {
            **RAW_ITEM["payload"],
            "review_fulltext": [],
            "fulltext": [
                {"url": "http://evil.example/x.pdf", "filename": "x.pdf"},
                {"url": "https://hub.lib.sfedu.ru/storage/ok.pdf", "filename": "ok.pdf"},
                {"url": "javascript:alert(1)", "filename": "js.pdf"},
            ],
        },
    }
    files = normalize_publication(raw, BASE)["files"]
    assert [f["filename"] for f in files] == ["ok.pdf"]
    assert files[0]["url"] == "https://hub.lib.sfedu.ru/storage/ok.pdf"


async def test_list_publications_reads_headers_and_filters():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params.multi_items())
        return _page([RAW_ITEM], page=2, num_pages=9, total=170)

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.list_publications(page=2, page_size=20, type_id=49, creator_id=7)

    assert seen == {"page": "2", "page_size": "20", "type": "49", "creator": "7"}
    assert (result["total"], result["page"], result["num_pages"]) == (170, 2, 9)
    assert result["items"][0]["title"] == "Статья про графы"


async def test_list_publications_falls_back_when_headers_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[RAW_ITEM])

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.list_publications(page=3, page_size=7)

    assert (result["total"], result["page"], result["page_size"], result["num_pages"]) == (0, 3, 7, 0)


async def test_list_publications_raises_not_found_for_bad_page():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(HubNotFoundError):
        await client.list_publications(page=99999)


async def test_list_publications_rejects_non_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(HubUnavailableError):
        await client.list_publications()


async def test_list_publications_rejects_non_list_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(HubUnavailableError):
        await client.list_publications()


async def test_list_publications_maps_auth_rejection():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "forbidden"})

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(HubAuthRequiredError):
        await client.list_publications()


async def test_list_publications_maps_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(HubUnavailableError):
        await client.list_publications()


async def test_list_publications_wraps_transport_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timeout", request=request)

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(HubUnavailableError):
        await client.list_publications()


async def test_response_larger_than_cap_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (hub_infra._MAX_RESPONSE_BYTES + 1))

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(HubUnavailableError):
        await client.list_publications()


async def test_search_scans_pages_and_matches_case_insensitively():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        if page == 1:
            return _page([RAW_ITEM, _item(102, "Про вакансии")], page=1, num_pages=2, page_size=100, total=3)
        return _page([_item(103, "Графовые модели")], page=2, num_pages=2, page_size=100, total=3)

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.search_publications("ГРАФ", max_pages=5)

    assert result["scanned_pages"] == 2
    assert result["truncated"] is False
    assert result["total_scanned"] == 3
    assert [item["id"] for item in result["matches"]] == [101, 103]


async def test_search_reports_truncation_when_cap_reached_without_num_pages():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_item(101, "Граф")])

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.search_publications("граф", max_pages=2)

    assert result["truncated"] is True


async def test_search_stops_at_empty_page():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        items = [_item(101, "Граф")] if page == 1 else []
        return _page(items, page=page, num_pages=50, page_size=100)

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.search_publications("граф", max_pages=5)

    assert result["scanned_pages"] == 2
    assert [item["id"] for item in result["matches"]] == [101]


async def test_scan_stops_when_page_404_mid_scan():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        if page <= 2:
            return _page([_item(page, "Граф")], page=page, num_pages=50, page_size=100)
        return httpx.Response(404, json={"detail": "no page"})

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.search_publications("граф", max_pages=5)

    assert result["scanned_pages"] == 2
    assert len(result["matches"]) == 2


async def test_get_publication_without_token_scans_pages():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        items = [_item(999, "Другое")] if page == 1 else [RAW_ITEM]
        return _page(items, page=page, num_pages=2, page_size=100)

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


async def test_get_publication_token_not_found_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "missing"})

    client = HubPublicationsClient(base_url=BASE, token="abc", transport=httpx.MockTransport(handler))
    with pytest.raises(HubNotFoundError):
        await client.get_publication(101)


async def test_get_publication_rejected_token_falls_back_to_scan():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/assets/"):
            return httpx.Response(403, json={"detail": "expired"})
        return _page([RAW_ITEM], page=1, num_pages=1, page_size=100)

    client = HubPublicationsClient(base_url=BASE, token="abc", transport=httpx.MockTransport(handler))
    item = await client.get_publication(101)

    assert item["id"] == 101
    assert client.token is None


async def test_get_publication_token_non_json_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = HubPublicationsClient(base_url=BASE, token="abc", transport=httpx.MockTransport(handler))
    with pytest.raises(HubUnavailableError):
        await client.get_publication(101)


async def test_get_publication_token_other_id_falls_back_to_scan():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/assets/"):
            return httpx.Response(200, json=_item(777, "Чужая запись"))
        return _page([RAW_ITEM], page=1, num_pages=1, page_size=100)

    client = HubPublicationsClient(base_url=BASE, token="abc", transport=httpx.MockTransport(handler))
    item = await client.get_publication(101)

    assert item["id"] == 101


async def test_get_publication_anonymous_skips_asset_endpoint():
    visited: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        visited.append(request.url.path)
        return _page([RAW_ITEM], page=1, num_pages=1, page_size=100)

    client = HubPublicationsClient(base_url=BASE, token="abc", transport=httpx.MockTransport(handler))
    item = await client.get_publication(101, use_token=False)

    assert item["id"] == 101
    assert all("/api/assets/" not in path for path in visited)


async def test_get_publication_missing_id_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page([RAW_ITEM], page=1, num_pages=1, page_size=100)

    client = HubPublicationsClient(base_url=BASE, token="", transport=httpx.MockTransport(handler))
    with pytest.raises(HubNotFoundError):
        await client.get_publication(999, max_scan_pages=2)


async def test_list_types_counts_sample_and_skips_bad_type_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        items = [
            {**RAW_ITEM, "id": 1, "type": {"id": 49, "type_name": "Препринт"}},
            {**RAW_ITEM, "id": 2, "type": {"id": 49, "type_name": "Препринт"}},
            {**RAW_ITEM, "id": 3, "type": {"id": 51, "type_name": "Студенческая работа"}},
            {**RAW_ITEM, "id": 4, "type": {"id": "abc", "type_name": "Сломанный"}},
            {**RAW_ITEM, "id": 5, "type": None},
        ]
        return _page(items, page=1, num_pages=1, page_size=100)

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.list_types(sample_pages=2)

    assert (result["scanned_pages"], result["sample_size"]) == (1, 5)
    assert result["types"] == [
        {"id": 49, "name": "Препринт", "count": 2},
        {"id": 51, "name": "Студенческая работа", "count": 1},
    ]


async def test_http_client_is_reused_between_requests():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _page([RAW_ITEM], page=1, num_pages=1, page_size=100)

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    await client.list_publications()
    await client.list_publications()
    assert len(calls) == 2

    await client.aclose()
    await client.aclose()
    assert len(calls) == 2


async def test_singleton_shared_and_closable(monkeypatch):
    monkeypatch.setattr(hub_infra, "_client", None)
    first = hub_infra.get_client()
    assert hub_infra.get_client() is first
    await hub_infra.close_client()
    assert hub_infra._client is None
    assert await hub_infra.close_client() is None


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
        matches = [normalize_publication(_item(i, f"Граф {i}"), BASE) for i in (101, 102, 103)]
        return {"matches": matches, "scanned_pages": 1, "truncated": False, "total_scanned": 3}

    async def list_types(self, sample_pages=3):
        return {"types": [{"id": 49, "name": "Препринт", "count": 1}], "scanned_pages": 1, "sample_size": 1}

    async def get_publication(self, publication_id, **kwargs):
        return normalize_publication(RAW_ITEM, BASE)


class _BrokenClient(_FakeClient):
    def __init__(self, exc: Exception):
        self._exc = exc

    async def list_publications(self, **kwargs):
        raise self._exc

    async def search_publications(self, q, **kwargs):
        raise self._exc

    async def list_types(self, sample_pages=3):
        raise self._exc

    async def get_publication(self, publication_id, **kwargs):
        raise self._exc


@pytest.fixture
def api(monkeypatch):
    def factory(client: _FakeClient):
        monkeypatch.setattr(publications_router, "get_client", lambda: client)
        app = FastAPI()
        app.state.limiter = publications_router.limiter
        app.include_router(publications_router.router)
        return TestClient(app)

    return factory


def test_router_list_publications(api):
    resp = api(_FakeClient()).get("/publications?page=1&page_size=20")
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"][0]["id"] == 101
    assert body["source"] == "hub.sfedu.ru"


def test_router_search_slices_by_offset_and_limit(api):
    resp = api(_FakeClient()).get("/publications/search?q=граф&limit=1&offset=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["total_scanned"] == 3
    assert [item["id"] for item in body["matches"]] == [102]


def test_router_types(api):
    resp = api(_FakeClient()).get("/publications/types")
    assert resp.status_code == 200
    assert resp.json()["types"][0]["name"] == "Препринт"


def test_router_detail(api):
    resp = api(_FakeClient()).get("/publications/101")
    assert resp.status_code == 200
    assert resp.json()["title"] == "Статья про графы"


class _RecordingClient(_FakeClient):
    def __init__(self):
        self.calls: list[dict] = []

    async def get_publication(self, publication_id, **kwargs):
        self.calls.append(kwargs)
        return normalize_publication(RAW_ITEM, BASE)


def test_router_detail_anonymous_does_not_use_hub_token(api):
    client = _RecordingClient()
    resp = api(client).get("/publications/101")
    assert resp.status_code == 200
    assert client.calls[0]["use_token"] is False


ENDPOINTS = [
    "/publications",
    "/publications/search?q=граф",
    "/publications/types",
    "/publications/101",
]


@pytest.mark.parametrize("path", ENDPOINTS)
@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (HubUnavailableError("down"), 503),
        (HubAuthRequiredError("nope"), 502),
    ],
)
def test_router_maps_upstream_errors_everywhere(api, path, exc, status):
    assert api(_BrokenClient(exc)).get(path).status_code == status


@pytest.mark.parametrize("path", ["/publications", "/publications/101"])
def test_router_maps_not_found(api, path):
    assert api(_BrokenClient(HubNotFoundError("missing"))).get(path).status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/publications?page=0",
        "/publications?page_size=101",
        "/publications/search?q=a",
        "/publications/types?sample_pages=11",
        "/publications/0",
    ],
)
def test_router_validates_query(api, path):
    assert api(_FakeClient()).get(path).status_code == 422


async def test_scan_is_bounded_by_max_pages():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        page = int(request.url.params["page"])
        return _page([_item(page, "Граф")], page=page, num_pages=1000, page_size=100)

    client = HubPublicationsClient(base_url=BASE, transport=httpx.MockTransport(handler))
    result = await client.search_publications("граф", max_pages=50)
    await client.aclose()

    assert result["scanned_pages"] == 10
    assert calls == 10
