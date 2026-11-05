"""Безопасность academic.py: источник токена, шифрование hub-JWT, лимит SSO.

Отдельный файл, потому что раньше academic-роутер не был покрыт вообще.
"""
import contextlib

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

    @staticmethod
    def _reset_limiter():
        """Сбросить счётчики slowapi.

        academic.py держит свой Limiter (как и другие роутеры), поэтому
        сбрасывать нужно именно его: in-memory хранилище общее на процесс.
        """
        from src.api_pkg.routers import academic

        with contextlib.suppress(Exception):
            academic.limiter.reset()

    def test_eleventh_request_is_429(self, monkeypatch):
        from src import config
        from src.api_pkg import create_app

        self._reset_limiter()
        monkeypatch.setattr(config, "ACADEMIC_SSO_ENABLED", False)
        client = TestClient(create_app(), raise_server_exceptions=False)

        codes = []
        for _ in range(12):
            codes.append(client.post("/api/auth/sso", json={"token": "x"}).status_code)

        # Первые 10 - 503 (SSO выключен), дальше лимитер отдаёт 429.
        assert 429 in codes, f"лимит не сработал: {codes}"

    def test_429_has_retry_after(self, monkeypatch):
        from src import config
        from src.api_pkg import create_app

        self._reset_limiter()
        monkeypatch.setattr(config, "ACADEMIC_SSO_ENABLED", False)
        client = TestClient(create_app(), raise_server_exceptions=False)

        resp = None
        for _ in range(12):
            r = client.post("/api/auth/sso", json={"token": "x"})
            if r.status_code == 429:
                resp = r
                break
        assert resp is not None, "лимит не сработал"
        assert resp.headers.get("Retry-After"), "нет Retry-After в 429"
        assert resp.headers["Retry-After"].isdigit()


class TestSsoDisabled:
    """При выключенном SSO эндпоинт сразу отдаёт 503 и не ходит наружу."""

    def test_disabled_returns_503(self, monkeypatch):
        from src import config
        from src.api_pkg import create_app
        from src.api_pkg.routers import academic

        with contextlib.suppress(Exception):
            academic.limiter.reset()
        monkeypatch.setattr(config, "ACADEMIC_SSO_ENABLED", False)
        client = TestClient(create_app(), raise_server_exceptions=False)
        r = client.post("/api/auth/sso", json={"token": "x"})
        assert r.status_code == 503, r.text


class TestAuditLogTokenSources:
    """request_logger не должен определять пользователя по ?token=.

    Иначе токен попадает в аудит дважды: в URL запроса и в колонку user.
    """

    def test_query_token_ignored_in_audit(self):
        from src.api_pkg.request_logger import _extract_token

        assert _extract_token(_req(query=b"token=SECRET")) is None

    def test_bearer_and_cookie_still_work_for_audit(self):
        from src.api_pkg.request_logger import _extract_token

        assert _extract_token(_req([(b"authorization", b"Bearer T")])) == "T"
        assert _extract_token(_req(cookies="token=C")) == "C"


class TestAcademicRouteRateLimits:
    """/academic/* ограничены slowapi: GET 60/мин, POST 10/мин."""

    @staticmethod
    def _reset():
        from src.api_pkg.routers import academic

        with contextlib.suppress(Exception):
            academic.limiter.reset()

    def _client(self, monkeypatch):
        from unittest.mock import AsyncMock

        from src.api_pkg import create_app
        from src.api_pkg.routers import academic

        self._reset()

        async def _fake_user(_request):
            return {"uid": "u-1", "u": "t@example.com", "r": "teacher"}

        monkeypatch.setattr(academic, "get_current_user", _fake_user)
        pool = AsyncMock()
        pool.fetch = AsyncMock(return_value=[])
        monkeypatch.setattr(academic, "get_pool", lambda: pool)
        return TestClient(create_app(), raise_server_exceptions=False)

    def test_get_krm_competencies_limited_to_60(self, monkeypatch):
        client = self._client(monkeypatch)
        for _ in range(60):
            assert client.get("/api/academic/krm-competencies").status_code == 200
        assert client.get("/api/academic/krm-competencies").status_code == 429

    def test_post_get_competencies_limited_to_10(self, monkeypatch):
        from fastapi import HTTPException

        from src.api_pkg.routers import academic

        client = self._client(monkeypatch)

        async def _no_sso(_request):
            raise HTTPException(status_code=403, detail="no sso")

        async def _no_dirs(_user):
            return []

        monkeypatch.setattr(academic, "_get_sso_token", _no_sso)
        monkeypatch.setattr(academic, "_user_direction_codes", _no_dirs)
        monkeypatch.setattr(
            academic,
            "_local_competencies",
            lambda *_a, **_k: {"recommended_competencies": []},
        )
        for _ in range(10):
            assert client.post(
                "/api/academic/get-competencies", json={"topic": "x"}
            ).status_code == 200
        assert client.post(
            "/api/academic/get-competencies", json={"topic": "x"}
        ).status_code == 429


class TestSsoTokenLengthCap:
    """/auth/sso не принимает hub-JWT длиннее 4096 символов."""

    def test_overlong_token_rejected(self, monkeypatch):
        from src import config
        from src.api_pkg import create_app
        from src.api_pkg.routers import academic

        with contextlib.suppress(Exception):
            academic.limiter.reset()
        monkeypatch.setattr(config, "ACADEMIC_SSO_ENABLED", False)
        client = TestClient(create_app(), raise_server_exceptions=False)
        assert client.post("/api/auth/sso", json={"token": "x" * 4097}).status_code == 422

    def test_token_within_cap_passes_validation(self, monkeypatch):
        from src import config
        from src.api_pkg import create_app
        from src.api_pkg.routers import academic

        with contextlib.suppress(Exception):
            academic.limiter.reset()
        monkeypatch.setattr(config, "ACADEMIC_SSO_ENABLED", False)
        client = TestClient(create_app(), raise_server_exceptions=False)
        # SSO выключен → 503, значит Pydantic пропустил тело (иначе было бы 422).
        assert client.post("/api/auth/sso", json={"token": "x"}).status_code == 503
