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
    assert resp.status_code == 404, resp.text
    assert "отключена" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_confirm_is_disabled_when_switch_off(flag_off: None) -> None:
    async with _app_client() as client:
        resp = await client.post("/api/auth/2fa/confirm", json={"code": "123456"})
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_verify_is_disabled_when_switch_off(flag_off: None) -> None:
    async with _app_client() as client:
        resp = await client.post(
            "/api/auth/2fa/verify",
            json={"challenge_id": "x" * 16, "code": "123456"},
        )
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_recovery_is_disabled_when_switch_off(flag_off: None) -> None:
    async with _app_client() as client:
        resp = await client.post(
            "/api/auth/2fa/recovery",
            json={"challenge_id": "x" * 16, "code": "abcdef"},
        )
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_setup_requires_actor_when_switch_on(flag_on: None) -> None:
    """Флаг включён: без сессии и без enroll_id настройка недоступна."""
    async with _app_client() as client:
        resp = await client.post("/api/auth/2fa/setup", json={})
    assert resp.status_code == 401, resp.text


def test_enroll_id_is_single_use() -> None:
    token = auth_mod._issue_enroll("uid-1")
    assert auth_mod._enroll_user_id(token) == "uid-1", "проверка не должна сжигать пропуск"
    assert auth_mod._consume_enroll(token) == "uid-1"
    assert auth_mod._consume_enroll(token) is None, "пропуск обязан быть одноразовым"


def test_enroll_id_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1_000_000.0
    monkeypatch.setattr(auth_mod.time, "time", lambda: now)
    token = auth_mod._issue_enroll("uid-2")
    monkeypatch.setattr(auth_mod.time, "time", lambda: now + auth_mod._ENROLL_TTL + 1)
    assert auth_mod._enroll_user_id(token) is None


def test_disabled_error_shape() -> None:
    exc = auth_mod._two_factor_disabled()
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 404
    assert "отключена" in exc.detail
