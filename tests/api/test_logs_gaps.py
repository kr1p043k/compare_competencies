"""Logs/monitoring gaps: token sources, student-actions visibility, summary route."""
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from tests.conftest import all_route_paths


def _req(path="/api/x", headers=None, cookies=None, query=""):
    from starlette.requests import Request
    scope = {"type": "http", "method": "GET", "path": path,
             "query_string": query.encode(), "headers": []}
    if headers:
        scope["headers"] = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    req = Request(scope)
    if cookies:
        req._cookies = cookies
    return req


def test_extract_token_all_sources():
    from src.api_pkg.request_logger import _extract_token
    assert _extract_token(_req(headers={"Authorization": "Bearer abc.def"})) == "abc.def"
    assert _extract_token(_req(cookies={"token": "c.t"})) == "c.t"
    # ?token= убран: значение писалось в аудит и утекало в логи дважды.
    assert _extract_token(_req(query="token=q.t")) is None
    assert _extract_token(_req()) is None
    # Bearer wins over cookie
    assert _extract_token(_req(headers={"Authorization": "Bearer b.t"},
                               cookies={"token": "c.t"})) == "b.t"


def test_student_actions_route_registered_and_guarded():
    paths = all_route_paths(create_app())
    assert "/api/admin/student-actions" in paths
    assert "/api/admin/monitoring/summary" in paths
    client = TestClient(create_app(), raise_server_exceptions=False)
    assert client.get("/api/admin/student-actions").status_code == 401
    assert client.get("/api/admin/monitoring/summary").status_code in (200, 401, 503)
