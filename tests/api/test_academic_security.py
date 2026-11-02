"""Безопасность academic.py: источник токена, шифрование hub-JWT, лимит SSO.

Отдельный файл, потому что раньше academic-роутер не был покрыт вообще.
"""
from fastapi import Request
from fastapi.testclient import TestClient


def _req(headers=None, query=b"", cookies=""):
    scope = {
        "type": "http",
        "path": "/academic/krm-competencies",
        "headers": [(k.lower(), v) for k, v in (headers or [])],
        "query_string": query,
    }
    if cookies:
        scope["headers"] = [(b"cookie", cookies.encode())] + scope["headers"]
    return Request(scope)


class TestAcademicTokenSources:
    """Токен не должен приниматься из query: он утекает в логи и Referer."""

    def test_query_token_ignored(self):
        from src.api_pkg.routers.academic import _request_token

        assert _request_token(_req(query=b"token=HUBJWT")) == ""

    def test_bearer_accepted(self):
        from src.api_pkg.routers.academic import _request_token

        assert _request_token(_req([(b"authorization", b"Bearer T1")])) == "T1"

    def test_cookie_accepted(self):
        from src.api_pkg.routers.academic import _request_token

        assert _request_token(_req(cookies="token=C1")) == "C1"

    def test_bearer_wins_over_cookie(self):
        from src.api_pkg.routers.academic import _request_token

        assert _request_token(
            _req([(b"authorization", b"Bearer B")], cookies="token=C")
        ) == "B"


class TestSsoTokenAtRest:
    """hub-JWT шифруется в sessions.sso_token, а не лежит открытым."""

    def _token(self):
        # Похоже на JWT хаба: header.payload.signature
        return "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6Iml2YW5vdiJ9.c2ln"

    def test_encrypt_roundtrip(self):
        from src.api_pkg.routers.academic import _decrypt_sso, _encrypt_sso

        token = self._token()
        stored = _encrypt_sso(token)
        assert token not in stored, "открытый токен не должен попадать в хранилище"
        assert _decrypt_sso(stored) == token

    def test_plaintext_legacy_is_detected(self):
        """Старые строки (до миграции) не должны падать - отдаём None."""
        from src.api_pkg.routers.academic import _decrypt_sso

        assert _decrypt_sso(self._token()) is None

    def test_ciphertext_differs_between_calls(self):
        """Fernet солит каждый раз - одинаковые токены дают разный шифр."""
        from src.api_pkg.routers.academic import _encrypt_sso

        token = self._token()
        assert _encrypt_sso(token) != _encrypt_sso(token)


class TestSsoRateLimit:
    """У /auth/sso есть лимит: без него перебирают hub-JWT."""

    def test_sso_has_rate_limit_decorator(self):
        from src.api_pkg.routers import academic

        fn = academic.sso_login
        # slowapi оборачивает функцию и оставляет ссылку на исходную
        assert hasattr(fn, "__wrapped__") or any(
            "limit" in str(a) for a in dir(fn)
        ), "sso_login не обёрнут лимитером"


class TestSsoDisabled:
    """При выключенном SSO эндпоинт сразу отдаёт 503 и не ходит наружу."""

    def test_disabled_returns_503(self, monkeypatch):
        from src import config
        from src.api_pkg import create_app

        monkeypatch.setattr(config, "ACADEMIC_SSO_ENABLED", False)
        client = TestClient(create_app(), raise_server_exceptions=False)
        r = client.post("/api/auth/sso", json={"token": "x"})
        assert r.status_code == 503
