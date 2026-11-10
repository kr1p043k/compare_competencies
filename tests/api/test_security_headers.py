"""Барьеры security-заголовков API.

Проверяем реальную фабрику приложения: базовые заголовки на всех ответах,
CSP только для API (не для Swagger-доков) и HSTS только по TLS.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api_pkg import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_baseline_headers_present():
    r = _client().get("/api/__no_such_route__")
    assert r.status_code == 404
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("referrer-policy") == "no-referrer"
    assert "geolocation=()" in r.headers.get("permissions-policy", "")
    assert r.headers.get("cache-control") == "no-store"


def test_hsts_absent_over_plain_http():
    r = _client().get("/api/__no_such_route__")
    assert r.headers.get("strict-transport-security") is None


def test_hsts_present_behind_tls_proxy():
    r = _client().get("/api/__no_such_route__", headers={"X-Forwarded-Proto": "https"})
    assert r.headers.get("strict-transport-security", "").startswith("max-age=")


def test_csp_set_for_api_responses():
    r = _client().get("/api/__no_such_route__")
    csp = r.headers.get("content-security-policy", "")
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp


def test_csp_not_set_for_swagger_docs():
    r = _client().get("/openapi.json")
    assert r.status_code == 200, r.text
    assert r.headers.get("content-security-policy") is None
