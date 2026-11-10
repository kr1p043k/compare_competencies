"""GRAFANA_PUBLIC_URL: ссылки дашбордов для браузера берутся из публичного
адреса, а серверный поиск идёт по внутреннему GRAFANA_URL.

Регресс: раньше url строился из GRAFANA_URL, из-за чего в докере ссылка вела на
недоступный браузеру http://grafana:3000.
"""
from __future__ import annotations

from src.api_pkg.routers import admin


def test_map_dashboards_uses_given_base() -> None:
    items = [{"uid": "u1", "title": "Dash", "url": "/d/u1/dash"}]
    out = admin._map_dashboards("https://host/grafana", items)
    assert out == [{"uid": "u1", "title": "Dash", "url": "https://host/grafana/d/u1/dash"}]


async def _fake_search(items):
    async def _inner(base, user, password):
        _inner.base = base
        return items
    return _inner


def _set(monkeypatch, **kw):
    from src import config
    for k, v in kw.items():
        monkeypatch.setattr(config.settings, k, v, raising=False)


def test_status_uses_public_url_for_links(monkeypatch) -> None:
    import asyncio
    _set(monkeypatch, GRAFANA_URL="http://grafana:3000",
         GRAFANA_PUBLIC_URL="https://host/grafana",
         GRAFANA_USER="admin", GRAFANA_PASSWORD=None)
    seen = {}

    async def fake_search(base, user, password):
        seen["base"] = base
        return [{"uid": "u1", "title": "D", "url": "/d/u1/d"}]

    monkeypatch.setattr(admin, "_grafana_search", fake_search)
    res = asyncio.run(admin._grafana_status())
    assert res["reachable"] is True
    # поиск идёт по внутреннему адресу, ссылка - по публичному
    assert seen["base"] == "http://grafana:3000"
    assert res["dashboards"][0]["url"] == "https://host/grafana/d/u1/d"


def test_status_falls_back_to_internal_when_public_unset(monkeypatch) -> None:
    import asyncio
    _set(monkeypatch, GRAFANA_URL="http://grafana:3000",
         GRAFANA_PUBLIC_URL=None, GRAFANA_USER="admin", GRAFANA_PASSWORD=None)

    async def fake_search(base, user, password):
        return [{"uid": "u1", "title": "D", "url": "/d/u1/d"}]

    monkeypatch.setattr(admin, "_grafana_search", fake_search)
    res = asyncio.run(admin._grafana_status())
    assert res["dashboards"][0]["url"] == "http://grafana:3000/d/u1/d"
