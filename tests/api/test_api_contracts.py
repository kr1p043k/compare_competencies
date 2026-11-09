"""API contract tests (backlog #52): stable read endpoints keep their shape.

The tests pin the CONTRACT, not volatile values: exact HTTP status for
deterministic read paths, presence and type of the required JSON keys, and
the error envelope `{"detail": ..., "request_id": ...}` that
`http_exception_handler` (src/api_pkg/__init__.py) attaches to every
HTTPException. Not asserted: response bodies of routing 404s (Starlette's
default handler does not add request_id) and concrete data values.

No live DB or network is involved:
- `/`, `/health`, `/api/health`, `/ready`, `/openapi.json` are pure;
- `/api/teacher/history` reads a JSONL dir monkeypatched to tmp_path and its
  role gate is stubbed via open_all_gates (auth itself is covered by
  tests/api/test_teacher_history.py and the security suites).

Style follows tests/api/test_teacher_history.py and tests/api/test_health.py.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.history as history_mod
from src.api_pkg import create_app
from tests.conftest import open_all_gates

_HISTORY_PARAMS = {"dir_code": "09.03.02", "year": 2025}


@pytest.fixture
def anon_client() -> TestClient:
    """Real gates: unauthenticated requests hit the real auth errors."""
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture
def teacher_client() -> TestClient:
    """All role gates stubbed with role=teacher: endpoint logic only."""
    app = open_all_gates(create_app(), role="teacher")
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def empty_history_dir(monkeypatch, tmp_path):
    """Point the router's HISTORY_DIR at a tmp_path without JSONL files."""
    monkeypatch.setattr(history_mod, "HISTORY_DIR", tmp_path)
    return tmp_path


class TestRootContract:
    def test_root_returns_200_with_service_keys(self, anon_client):
        resp = anon_client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("service", "version", "docs", "health", "status"):
            assert isinstance(body[key], str), key


class TestHealthContract:
    @pytest.mark.parametrize("path", ["/health", "/api/health"])
    def test_returns_200_with_health_keys(self, anon_client, path):
        resp = anon_client.get(path)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for key in ("status", "version"):
            assert isinstance(body[key], str), key
        for key in ("evaluator", "recommendation_engine"):
            assert isinstance(body[key], bool), key


class TestReadyContract:
    def test_status_code_matches_body_status(self, anon_client):
        resp = anon_client.get("/ready")
        # Without a lifespan and DB pool the endpoint answers 503, but the
        # contract allows both states; the code and the body must agree
        # (same convention as tests/api/test_health.py).
        assert resp.status_code in (200, 503)
        body = resp.json()
        assert isinstance(body["status"], str)
        assert isinstance(body["problems"], list)
        assert isinstance(body["components"], dict)
        assert all(isinstance(v, bool) for v in body["components"].values())
        assert (body["status"] == "ok") == (resp.status_code == 200)


class TestOpenApiContract:
    def test_openapi_schema_shape(self, anon_client):
        resp = anon_client.get("/openapi.json")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["openapi"], str)
        assert isinstance(body["info"], dict)
        assert isinstance(body["info"]["title"], str)
        assert isinstance(body["paths"], dict)
        assert isinstance(body["components"], dict)


class TestTeacherHistoryContract:
    def test_empty_history_dir_returns_200_with_points_keys(self, teacher_client, empty_history_dir):
        resp = teacher_client.get("/api/teacher/history", params=_HISTORY_PARAMS)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for key in ("points", "first", "last", "delta", "skipped_lines"):
            assert key in body, key
        assert isinstance(body["points"], list)
        assert isinstance(body["skipped_lines"], int)


class TestRequestIdHeaderContract:
    def test_success_response_carries_uuid_header(self, anon_client):
        # add_request_id_middleware stamps X-Request-ID on every response.
        resp = anon_client.get("/api/health")
        assert resp.status_code == 200
        uuid.UUID(resp.headers["X-Request-ID"])


class TestErrorEnvelopeContract:
    def _assert_error_envelope(self, resp, status_code: int):
        assert resp.status_code == status_code, resp.text
        body = resp.json()
        assert isinstance(body["detail"], str)
        assert isinstance(body["request_id"], str)
        assert body["request_id"], "request_id must not be empty"

    def test_anonymous_request_gets_401_envelope(self, anon_client, empty_history_dir):
        resp = anon_client.get("/api/teacher/history", params=_HISTORY_PARAMS)
        self._assert_error_envelope(resp, 401)

    def test_invalid_dir_code_gets_400_envelope(self, teacher_client, empty_history_dir):
        resp = teacher_client.get(
            "/api/teacher/history", params={"dir_code": "../evil", "year": 2025}
        )
        self._assert_error_envelope(resp, 400)

    def test_student_role_gets_403_envelope(self, monkeypatch, empty_history_dir):
        # Same trick as tests/api/test_teacher_history.py: substitute
        # get_current_user only, so the real require_any_role answers 403.
        async def _student(_request):
            return {"uid": "s-1", "u": "student@example.com", "r": "student", "t": None}

        monkeypatch.setattr(auth_mod, "get_current_user", _student)
        client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.get("/api/teacher/history", params=_HISTORY_PARAMS)
        self._assert_error_envelope(resp, 403)
