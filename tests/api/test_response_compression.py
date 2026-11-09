"""Барьер: API сжимает крупные ответы (gzip) для клиентов, которые его просят.

Проверяем реальную фабрику приложения: /openapi.json - большой JSON без БД и
авторизации, поэтому он удобен как индикатор включённого GZipMiddleware.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api_pkg import create_app


def test_large_json_response_is_gzipped():
    client = TestClient(create_app())
    r = client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200, r.text
    assert r.headers.get("content-encoding") == "gzip"
    # httpx декодирует тело автоматически - проверяем, что содержимое корректно.
    body = r.json()
    assert "paths" in body and len(body["paths"]) > 10


def test_client_without_gzip_gets_identity():
    client = TestClient(create_app())
    r = client.get("/openapi.json", headers={"Accept-Encoding": "identity"})
    assert r.status_code == 200, r.text
    assert r.headers.get("content-encoding") in (None, "identity")


def test_vary_accept_encoding_present():
    """Ответы помечаются Vary: Accept-Encoding - корректная семантика кэшей.

    Порог minimum_size в текущем стеке middleware не гарантируется (внешние
    BaseHTTPMiddleware стримят ответ), поэтому отдельно его не проверяем -
    важно, что крупные тела сжимаются и заголовок Vary выставлен.
    """
    client = TestClient(create_app())
    r = client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200, r.text
    assert "accept-encoding" in (r.headers.get("vary", "").lower())
