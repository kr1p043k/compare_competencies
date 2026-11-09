"""Grafana-мониторинг админки: статус дашбордов с fallback.

Эндпоинт не должен падать, когда Grafana не сконфигурирована или недоступна.
Сеть не трогаем: HTTP-вызов к Grafana замокан.
"""
from __future__ import annotations

from pydantic import SecretStr

import src.api_pkg.routers.admin as admin_mod
from src import config


async def test_disabled_when_url_unset(monkeypatch):
    monkeypatch.setattr(config.settings, "GRAFANA_URL", None)
    assert admin_mod._grafana_creds() is None
    out = await admin_mod._grafana_status()
    assert out["enabled"] is False
    assert out["reachable"] is False
    assert out["dashboards"] == []
    assert out["reason"]


async def test_unreachable_falls_back(monkeypatch):
    monkeypatch.setattr(config.settings, "GRAFANA_URL", "http://grafana.local")
    monkeypatch.setattr(config.settings, "GRAFANA_USER", "admin")
    monkeypatch.setattr(config.settings, "GRAFANA_PASSWORD", "secret")

    async def _boom(base, user, password):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(admin_mod, "_grafana_search", _boom)
    out = await admin_mod._grafana_status()
    assert out["enabled"] is True
    assert out["reachable"] is False
    assert "Grafana недоступна" in out["reason"]
    assert out["dashboards"] == []


async def test_ok_maps_dashboards_and_strips_slash(monkeypatch):
    monkeypatch.setattr(config.settings, "GRAFANA_URL", "http://grafana.local/")
    monkeypatch.setattr(config.settings, "GRAFANA_USER", "admin")
    monkeypatch.setattr(config.settings, "GRAFANA_PASSWORD", "secret")

    async def _ok(base, user, password):
        assert base == "http://grafana.local"  # trailing slash stripped
        return [
            {"uid": "api", "title": "API latency", "url": "/d/api/api-latency"},
            {"title": "no uid"},  # skipped
            "not-a-dict",  # skipped
        ]

    monkeypatch.setattr(admin_mod, "_grafana_search", _ok)
    out = await admin_mod._grafana_status()
    assert out["enabled"] is True
    assert out["reachable"] is True
    assert out["reason"] is None
    assert out["dashboards"] == [
        {"uid": "api", "title": "API latency",
         "url": "http://grafana.local/d/api/api-latency"},
    ]


def test_creds_read_password_secret(monkeypatch):
    monkeypatch.setattr(config.settings, "GRAFANA_URL", "http://g.local/")
    monkeypatch.setattr(config.settings, "GRAFANA_USER", "u")
    monkeypatch.setattr(config.settings, "GRAFANA_PASSWORD", SecretStr("p"))
    assert admin_mod._grafana_creds() == ("http://g.local", "u", "p")
