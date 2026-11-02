"""Token hygiene: no token from query, SSO token encrypted at rest.

Covers the two changes in academic.py / request_logger.py:
  * _request_token reads header or cookie only;
  * stored hub JWT is encrypted and round-trips through decrypt_secret.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from starlette.requests import Request

from src.api_pkg import crypto
from src.api_pkg.routers import academic


def _req(headers=None, cookies=None, query=None):
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/x",
        "raw_path": b"/x",
        "query_string": (query or "").encode(),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("127.0.0.1", 1),
        "server": ("t", 80),
    }
    req = Request(scope)
    if cookies is not None:
        req._cookies = cookies  # noqa: SLF001 - тестовый шов
    return req


def test_token_from_bearer_header():
    assert academic._request_token(_req(headers={"Authorization": "Bearer abc"})) == "abc"


def test_token_from_cookie():
    assert academic._request_token(_req(cookies={"token": "cookie-tok"})) == "cookie-tok"


def test_token_from_query_is_ignored():
    # Раньше токен принимался из ?token=, он утекает в логи прокси и Referer.
    assert academic._request_token(_req(query="token=leaked")) == ""


def test_query_token_does_not_override_cookie():
    req = _req(cookies={"token": "good"}, query="token=evil")
    assert academic._request_token(req) == "good"


def test_sso_token_roundtrip(monkeypatch):
    monkeypatch.setattr(crypto.config, "get_secret_key", lambda: "x" * 64)
    raw = "header.payload.signature"
    stored = crypto.encrypt_secret(raw)
    assert stored != raw
    assert raw not in stored
    assert crypto.decrypt_secret(stored) == raw


def test_plaintext_value_is_not_decrypted(monkeypatch):
    # Старые строки, записанные открытым текстом, не должны приниматься за шифр.
    monkeypatch.setattr(crypto.config, "get_secret_key", lambda: "x" * 64)
    assert crypto.decrypt_secret("header.payload.signature") is None


def test_corrupted_value_returns_none(monkeypatch):
    monkeypatch.setattr(crypto.config, "get_secret_key", lambda: "x" * 64)
    good = crypto.encrypt_secret("secret")
    broken = good[:-4] + "AAAA"
    assert crypto.decrypt_secret(broken) is None


def test_different_secret_key_cannot_decrypt(monkeypatch):
    monkeypatch.setattr(crypto.config, "get_secret_key", lambda: "a" * 64)
    stored = crypto.encrypt_secret("secret")
    monkeypatch.setattr(crypto.config, "get_secret_key", lambda: "b" * 64)
    assert crypto.decrypt_secret(stored) is None


def test_sso_rate_key_uses_client_ip(monkeypatch):
    monkeypatch.setattr(academic, "client_ip", lambda _r: "203.0.113.9")
    assert academic._sso_rate_key(_req()) == "sso:203.0.113.9"


def test_sso_request_rejects_oversized_token():
    with pytest.raises(ValidationError):
        academic.SsoRequest(token="x" * 5000)


def test_sso_request_accepts_normal_token():
    req = academic.SsoRequest(token="a.b.c")
    assert req.token == "a.b.c"


def test_request_logger_has_no_query_token_path():
    import inspect

    import src.api_pkg.request_logger as rl

    src = inspect.getsource(rl._extract_token)
    assert "query_params" not in src, "token must not be read from the query string"


def test_namespace_guard():
    # SsoRequest — обычная pydantic-модель без магии; фиксируем это.
    assert isinstance(academic.SsoRequest(token="x"), SimpleNamespace) is False
