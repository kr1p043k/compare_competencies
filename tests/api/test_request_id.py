"""Сквозной X-Request-ID (7.3.2).

Правило контракта:
- клиент/прокси может прислать свой X-Request-ID, и он сохраняется в ответе,
  в логах и в конверте ошибки;
- небезопасные значения (перевод строки, пробелы, слишком длинные, слишком
  короткие) не принимаются - генерируется новый uuid4;
- заголовок есть на любом ответе, включая ошибки.

Стиль - как в tests/api/test_health_hardening.py: TestClient(create_app())
без lifespan.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api_pkg import create_app


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


class TestIncomingRequestIdAccepted:
    def test_safe_id_is_echoed(self):
        incoming = "abc123-DEF.456:ghi_789"
        resp = _client().get("/api/health", headers={"X-Request-ID": incoming})
        assert resp.status_code == 200
        assert resp.headers["X-Request-ID"] == incoming

    def test_safe_id_reaches_error_envelope(self):
        incoming = "0123456789abcdef"
        resp = _client().get(
            "/api/teacher/history",
            headers={"X-Request-ID": incoming},
            params={"profile": "base", "query": "x", "regions": ""},
        )
        assert resp.status_code == 401
        assert resp.json()["request_id"] == incoming


class TestUnsafeRequestIdRejected:
    @pytest.mark.parametrize(
        "bad",
        [
            "short",
            "has space in it",
            "line\nbreak-value",
            "x" * 65,
            "carriage\rreturn",
            "quote\"inject",
        ],
    )
    def test_bad_id_is_replaced_by_uuid(self, bad):
        resp = _client().get("/api/health", headers={"X-Request-ID": bad})
        assert resp.status_code == 200
        got = resp.headers["X-Request-ID"]
        assert got != bad
        uuid.UUID(got)

    def test_missing_id_is_uuid(self):
        resp = _client().get("/api/health")
        assert resp.status_code == 200
        uuid.UUID(resp.headers["X-Request-ID"])

    def test_two_requests_get_distinct_ids(self):
        client = _client()
        first = client.get("/api/health").headers["X-Request-ID"]
        second = client.get("/api/health").headers["X-Request-ID"]
        assert first != second
