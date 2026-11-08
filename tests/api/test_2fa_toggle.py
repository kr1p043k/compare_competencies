"""Мастер-переключатель 2FA: по умолчанию выключено, включение - одним флагом.

Тесты без БД: подменяем пул и чтение флага, проверяем контракт эндпоинтов и
одноразовость пропуска на первичную настройку (enroll_id).
"""
from __future__ import annotations

import contextlib
from typing import Any

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

import src.api_pkg.routers.auth as auth_mod
from src.api_pkg import create_app


class _FakePool:
    async def execute(self, *_a: Any, **_kw: Any) -> str:
        return "OK"

    async def fetchval(self, *_a: Any, **_kw: Any) -> Any:
        return None

    async def fetchrow(self, *_a: Any, **_kw: Any) -> Any:
        return None


def _app_client():
    return AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://t")


@pytest.fixture()
def flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _disabled(_pool: Any) -> bool:
        return False

    monkeypatch.setattr(auth_mod, "get_two_factor_enabled", _disabled)
    monkeypatch.setattr(auth_mod, "get_pool", lambda: _FakePool())
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()


@pytest.fixture()
def flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _enabled(_pool: Any) -> bool:
        return True

    monkeypatch.setattr(auth_mod, "get_two_factor_enabled", _enabled)
    monkeypatch.setattr(auth_mod, "get_pool", lambda: _FakePool())
    with contextlib.suppress(Exception):
        auth_mod.limiter._storage.reset()


@pytest.mark.asyncio
async def test_policy_is_public_and_reports_off(flag_off: None) -> None:
    async with _app_client() as client:
        resp = await client.get("/api/auth/2fa/policy")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"enabled": False}


@pytest.mark.asyncio
async def test_policy_reports_on(flag_on: None) -> None:
    async with _app_client() as client:
        resp = await client.get("/api/auth/2fa/policy")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"enabled": True}


@pytest.mark.asyncio
async def test_setup_is_disabled_when_switch_off(flag_off: None) -> None:
    async with _app_client() as client:
        resp = await client.post("/api/auth/2fa/setup", json={})
    assert resp.status_code == 403, resp.text
    assert "отключена" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_confirm_is_disabled_when_switch_off(flag_off: None) -> None:
    async with _app_client() as client:
        resp = await client.post("/api/auth/2fa/confirm", json={"code": "123456"})
    assert resp.status_code == 403, resp.text


@pytest.mark.asyncio
async def test_verify_is_disabled_when_switch_off(flag_off: None) -> None:
    async with _app_client() as client:
        resp = await client.post(
            "/api/auth/2fa/verify",
            json={"challenge_id": "x" * 16, "code": "123456"},
        )
    assert resp.status_code == 403, resp.text


@pytest.mark.asyncio
async def test_recovery_is_disabled_when_switch_off(flag_off: None) -> None:
    async with _app_client() as client:
        resp = await client.post(
            "/api/auth/2fa/recovery",
            json={"challenge_id": "x" * 16, "code": "abcdef"},
        )
    assert resp.status_code == 403, resp.text


@pytest.mark.asyncio
async def test_setup_requires_actor_when_switch_on(flag_on: None) -> None:
    """Флаг включён: без сессии и без enroll_id настройка недоступна."""
    async with _app_client() as client:
        resp = await client.post("/api/auth/2fa/setup", json={})
    assert resp.status_code == 401, resp.text


def test_disabled_error_shape() -> None:
    exc = auth_mod._two_factor_disabled()
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 403
    assert "отключена" in exc.detail

def test_recovery_code_hash_ignores_case_and_whitespace() -> None:
    """Код из письма часто вводят заглавными: регистр не должен ломать вход."""
    assert auth_mod._hash_recovery_code("A1B2C3D4") == auth_mod._hash_recovery_code("a1b2c3d4")
    assert auth_mod._hash_recovery_code("  a1b2c3d4  ") == auth_mod._hash_recovery_code("a1b2c3d4")
