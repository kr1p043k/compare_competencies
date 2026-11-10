"""Единый конверт ошибок: {code, message, request_id} (+ detail для совместимости).

Проверяем, что и «свои» HTTPException, и ошибки валидации, и грубые ошибки
отдают одинаковый набор ключей, а в заголовке есть X-Request-ID.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api_pkg import create_app

REQUIRED_KEYS = {"code", "message", "detail", "request_id"}


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _assert_envelope(resp, expected_code: str | None = None) -> dict:
    body = resp.json()
    assert set(body) >= REQUIRED_KEYS, f"нет ключей конверта: {body}"
    assert isinstance(body["code"], str) and body["code"]
    assert isinstance(body["message"], str) and body["message"]
    assert isinstance(body["request_id"], str) and body["request_id"]
    assert resp.headers.get("X-Request-ID")
    if expected_code is not None:
        assert body["code"] == expected_code, body
    return body


def test_404_uses_envelope() -> None:
    resp = _client().get("/api/definitely-not-a-route-xyz")
    assert resp.status_code == 404
    _assert_envelope(resp, "not_found")


def test_401_uses_envelope() -> None:
    resp = _client().post("/api/admin/whitelist/add", json={})
    assert resp.status_code == 401
    _assert_envelope(resp, "unauthorized")


def test_validation_error_uses_envelope() -> None:
    # email/password отсутствуют -> RequestValidationError
    resp = _client().post("/api/auth/login", json={})
    assert resp.status_code == 422
    body = _assert_envelope(resp, "validation_error")
    assert isinstance(body.get("errors"), list) and body["errors"]
