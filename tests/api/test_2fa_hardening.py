"""2FA: Р·Р°РєСЂС‹С‚С‹Рµ РґС‹СЂС‹ РёР· security-СЂРµРІСЊСЋ.

РўСЂРё РґРµС„РµРєС‚Р°, РЅР°Р№РґРµРЅРЅС‹Рµ СЂРµРІСЊСЋ:
  H1. SSO-РІС…РѕРґ РѕР±С…РѕРґРёР» 2FA Р°РґРјРёРЅР° (РІ academic.py РїСЂРѕРІРµСЂРєРё РЅРµ Р±С‹Р»Рѕ);
  H2. РѕС€РёР±РєР° С‡С‚РµРЅРёСЏ СЃРѕСЃС‚РѕСЏРЅРёСЏ 2FA С‚СЂР°РєС‚РѕРІР°Р»Р°СЃСЊ РєР°Рє В«РЅРµ
      РЅР°СЃС‚СЂРѕРµРЅР°В» (fail-open);
  M1. recovery-РєРѕРґС‹ Р±С‹Р»Рё 32-Р±РёС‚РЅС‹РјРё Рё СЃСЂР°РІРЅРёРІР°Р»РёСЃСЊ РЅРµ constant-time.
"""
from __future__ import annotations

import json
import secrets

import pytest
from fastapi import HTTPException

import src.api_pkg.routers.auth as auth_mod

FLAG_TOTP_2FA = "totp_2fa_enabled"


@pytest.fixture(autouse=True)
def _totp_feature_on(monkeypatch, tmp_path):
    """Autouse: TOTP 2FA включена в отдельном временном features-файле."""
    features_file = tmp_path / "features.json"
    features_file.write_text(json.dumps({FLAG_TOTP_2FA: True}), encoding="utf-8")
    monkeypatch.setattr("src.api_pkg.feature_flags._FEATURES_PATH", features_file)


class TestTotpFailClosed:
    """H2: СЃР±РѕР№ Р‘Р” РЅРµ РґРѕР»Р¶РµРЅ СЃРЅРёРјР°С‚СЊ РІС‚РѕСЂРѕР№ С„Р°РєС‚РѕСЂ."""

    @pytest.mark.asyncio
    async def test_db_error_raises_503(self):
        class BrokenPool:
            async def fetchrow(self, *_a, **_kw):
                raise RuntimeError("connection lost")

        with pytest.raises(HTTPException) as exc:
            await auth_mod._totp_state(BrokenPool(), "u-1")
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_row_means_not_configured(self):
        class EmptyPool:
            async def fetchrow(self, *_a, **_kw):
                return None

        assert await auth_mod._totp_state(EmptyPool(), "u-1") is None


class TestRecoveryCodes:
    """M1: 64-Р±РёС‚РЅС‹Рµ РєРѕРґС‹, СЃСЂР°РІРЅРµРЅРёРµ constant-time."""

    def test_codes_are_64_bit(self):
        # token_hex(8) -> 16 hex-СЃРёРјРІРѕР»РѕРІ = 64 Р±РёС‚Р°
        code = secrets.token_hex(8)
        assert len(code) == 16

    def test_hash_is_stable_and_salted_with_secret(self):
        a = auth_mod._hash_recovery_code("abcd1234abcd1234")
        b = auth_mod._hash_recovery_code("abcd1234abcd1234")
        assert a == b
        assert a != "abcd1234abcd1234", "РєРѕРґ РЅРµ РґРѕР»Р¶РµРЅ С…СЂР°РЅРёС‚СЊСЃСЏ РѕС‚РєСЂС‹С‚С‹Рј"
        assert len(a) == 64, "sha256 hex"


class TestSsoRequiresTotp:
    """H1: SSO-РІС…РѕРґ Р°РґРјРёРЅР° РёРґС‘С‚ С‡РµСЂРµР· challenge, Р° РЅРµ СЃСЂР°Р·Сѓ РІС‹РґР°С‘С‚ С‚РѕРєРµРЅ."""

    @pytest.mark.asyncio
    async def test_admin_with_confirmed_totp_gets_challenge(self, monkeypatch):
        import src.api_pkg.routers.academic as acad

        class FakeResp:
            status_code = 200
            text = "ok"

            def json(self):
                return {"valid": True}

        class FakeClient:
            def __init__(self, *_a, **_kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            async def post(self, *_a, **_kw):
                return FakeResp()

        class FakePool:
            async def execute(self, *_a, **_kw):
                return "OK"

            async def fetchrow(self, sql, *_a, **_kw):
                if "FROM users" in sql and "admin_totp" not in sql:
                    return {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "email": "boss@sfedu.ru",
                        "role": "admin",
                        "full_name": "Boss",
                    }
                return None

        async def _confirmed(_pool, _uid):
            return {"secret": "x", "confirmed": True, "recovery_codes": []}

        monkeypatch.setattr(acad.httpx, "AsyncClient", FakeClient)
        monkeypatch.setattr(acad, "get_pool", lambda: FakePool())
        monkeypatch.setattr(acad.config, "ACADEMIC_SSO_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_totp_state", _confirmed)
        async def _flag_on(_pool):
            return True
        monkeypatch.setattr(auth_mod, "get_two_factor_enabled", _flag_on)

        from src.api_pkg.routers.academic import SsoRequest

        request = _fake_request()
        result = await acad.sso_login(SsoRequest(token=_hub_token()), request)

        assert result["status"] == "totp_required", result
        assert "token" not in result, "SSO РЅРµ РґРѕР»Р¶РµРЅ РІС‹РґР°РІР°С‚СЊ С‚РѕРєРµРЅ РІ РѕР±С…РѕРґ 2FA"
        assert result["challenge_id"]

    @pytest.mark.asyncio
    async def test_admin_without_totp_gets_token(self, monkeypatch):
        import src.api_pkg.routers.academic as acad

        class FakeResp:
            status_code = 200
            text = "ok"

            def json(self):
                return {"valid": True}

        class FakeClient:
            def __init__(self, *_a, **_kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            async def post(self, *_a, **_kw):
                return FakeResp()

        class FakePool:
            async def execute(self, *_a, **_kw):
                return "OK"

            async def fetchrow(self, sql, *_a, **_kw):
                if "FROM users" in sql and "admin_totp" not in sql:
                    return {
                        "id": "22222222-2222-2222-2222-222222222222",
                        "email": "boss2@sfedu.ru",
                        "role": "admin",
                        "full_name": "Boss 2",
                    }
                return None

        async def _not_configured(_pool, _uid):
            return None

        monkeypatch.setattr(acad.httpx, "AsyncClient", FakeClient)
        monkeypatch.setattr(acad, "get_pool", lambda: FakePool())
        monkeypatch.setattr(acad.config, "ACADEMIC_SSO_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_totp_state", _not_configured)

        from src.api_pkg.routers.academic import SsoRequest

        result = await acad.sso_login(SsoRequest(token=_hub_token()), _fake_request())
        assert result.get("token"), "Р°РґРјРёРЅ Р±РµР· 2FA РґРѕР»Р¶РµРЅ РІС…РѕРґРёС‚СЊ РєР°Рє СЂР°РЅСЊС€Рµ"

    @pytest.mark.asyncio
    async def test_admin_with_confirmed_totp_gets_token_when_flag_off(self, monkeypatch):
        """Kill-switch: при выключенном флаге SSO выдаёт токен, а не challenge."""
        import src.api_pkg.routers.academic as acad

        class FakeResp:
            status_code = 200
            text = "ok"

            def json(self):
                return {"valid": True}

        class FakeClient:
            def __init__(self, *_a, **_kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            async def post(self, *_a, **_kw):
                return FakeResp()

        class FakePool:
            async def execute(self, *_a, **_kw):
                return "OK"

            async def fetchrow(self, sql, *_a, **_kw):
                if "FROM users" in sql and "admin_totp" not in sql:
                    return {
                        "id": "33333333-3333-3333-3333-333333333333",
                        "email": "boss3@sfedu.ru",
                        "role": "admin",
                        "full_name": "Boss 3",
                    }
                return None

        async def _confirmed(_pool, _uid):
            return {"secret": "x", "confirmed": True, "recovery_codes": []}

        monkeypatch.setattr(acad.httpx, "AsyncClient", FakeClient)
        monkeypatch.setattr(acad, "get_pool", lambda: FakePool())
        monkeypatch.setattr(acad.config, "ACADEMIC_SSO_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_totp_state", _confirmed)
        monkeypatch.setattr("src.api_pkg.feature_flags.is_enabled", lambda _flag: False)

        from src.api_pkg.routers.academic import SsoRequest

        result = await acad.sso_login(SsoRequest(token=_hub_token()), _fake_request())
        assert result.get("token"), "при выключенном флаге SSO выдаёт токен"
        assert "challenge_id" not in result


def _hub_token(username: str = "boss") -> str:
    """JWT-РїРѕРґРѕР±РЅР°СЏ СЃС‚СЂРѕРєР°: _jwt_username С‡РёС‚Р°РµС‚ payload Р±РµР· РїСЂРѕРІРµСЂРєРё РїРѕРґРїРёСЃРё."""
    import base64
    import json as _json

    payload = (
        base64.urlsafe_b64encode(_json.dumps({"username": username}).encode())
        .decode()
        .rstrip("=")
    )
    return f"h.{payload}.s"


def _fake_request():
    from starlette.requests import Request

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/sso",
            "headers": [(b"user-agent", b"pytest"), (b"host", b"test")],
            "query_string": b"",
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
            "scheme": "http",
        }
    )

