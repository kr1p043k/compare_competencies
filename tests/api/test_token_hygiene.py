"""Token hygiene tests against the upstream implementation.

Upstream (origin/for_Nikita) already:
  * reads the token from header/cookie only (never ?token=);
  * encrypts the stored hub JWT with Fernet (inline in academic.py);
  * rate-limits /auth/sso at 10/minute using the real client IP.
My contribution on top: the token length bound on SsoRequest.
"""

import pytest
from pydantic import ValidationError
from starlette.requests import Request

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


# ---------- token transport ----------

def test_token_from_bearer_header():
    assert academic._request_token(_req(headers={"Authorization": "Bearer abc"})) == "abc"


def test_token_from_cookie():
    assert academic._request_token(_req(cookies={"token": "cookie-tok"})) == "cookie-tok"


def test_token_from_query_is_ignored():
    assert academic._request_token(_req(query="token=leaked")) == ""


def test_query_token_does_not_override_cookie():
    req = _req(cookies={"token": "good"}, query="token=evil")
    assert academic._request_token(req) == "good"


# ---------- SSO JWT at rest ----------

def test_sso_token_roundtrip():
    raw = "header.payload.signature"
    stored = academic._encrypt_sso(raw)
    assert stored != raw
    assert raw not in stored
    assert academic._decrypt_sso(stored) == raw


def test_plaintext_value_is_not_decrypted():
    # Legacy-строки, записанные открытым текстом, не должны считаться шифром.
    assert academic._decrypt_sso("header.payload.signature") is None


def test_corrupted_value_returns_none():
    broken = academic._encrypt_sso("secret")[:-4] + "AAAA"
    assert academic._decrypt_sso(broken) is None


def test_different_secret_key_cannot_decrypt(monkeypatch):
    stored = academic._encrypt_sso("secret")
    monkeypatch.setattr(academic.config, "get_secret_key", lambda: "b" * 64)
    assert academic._decrypt_sso(stored) is None


# ---------- request bound ----------

def test_sso_request_rejects_oversized_token():
    with pytest.raises(ValidationError):
        academic.SsoRequest(token="x" * 5000)


def test_sso_request_accepts_normal_token():
    assert academic.SsoRequest(token="a.b.c").token == "a.b.c"


# ---------- audit log must not accept ?token= ----------

def test_request_logger_has_no_query_token_path():
    import inspect

    import src.api_pkg.request_logger as rl

    assert "query_params" not in inspect.getsource(rl._extract_token)
