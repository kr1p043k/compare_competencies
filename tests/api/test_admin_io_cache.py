"""Кэш и потоки в админских читалках (whitelist, метрики, полный отчёт)."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api_pkg.routers.admin as admin_mod
import src.api_pkg.routers.auth as auth_mod


async def _fake_current_user(_request):
    return {"u": "admin@example.com", "r": "admin"}


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(auth_mod, "get_current_user", _fake_current_user)
    monkeypatch.setattr(admin_mod, "get_logs_by_user", lambda: {})
    with contextlib.suppress(Exception):
        admin_mod.limiter._storage.reset()
    app = FastAPI()
    app.state.limiter = admin_mod.limiter
    app.include_router(admin_mod.router)
    return TestClient(app, raise_server_exceptions=False)


def test_whitelist_cached_until_mtime_changes(monkeypatch, tmp_path):
    skills_file = tmp_path / "it_skills.json"
    skills_file.write_text(json.dumps(["python", "sql"]), encoding="utf-8")

    reads = {"n": 0}

    def _fake_load() -> set[str]:
        reads["n"] += 1
        return {"python", "sql"}

    monkeypatch.setattr(admin_mod, "IT_SKILLS_PATH", skills_file)
    monkeypatch.setattr(admin_mod, "load_it_skills", _fake_load)
    monkeypatch.setattr(admin_mod, "_whitelist_cache", None, raising=False)

    first = admin_mod._load_whitelist()
    second = admin_mod._load_whitelist()
    assert first == second == ["python", "sql"]
    assert reads["n"] == 1, "повторный вызов должен идти из кэша"

    # Меняем mtime файла - кэш обязан инвалидироваться.
    skills_file.write_text(json.dumps(["python", "sql", "go"]), encoding="utf-8")
    import os
    os.utime(skills_file, (1, 1))
    admin_mod._load_whitelist()
    assert reads["n"] == 2

    admin_mod._whitelist_cache = None


def test_save_whitelist_invalidates_cache(monkeypatch, tmp_path):
    skills_file = tmp_path / "it_skills.json"
    skills_file.write_text(json.dumps(["python"]), encoding="utf-8")
    monkeypatch.setattr(admin_mod, "IT_SKILLS_PATH", skills_file)
    monkeypatch.setattr(admin_mod, "_whitelist_cache", (123.0, ["stale"]), raising=False)

    admin_mod._save_whitelist(["python", "sql"])
    assert admin_mod._whitelist_cache is None, "сохранение должно сбросить кэш"
    assert json.loads(skills_file.read_text(encoding="utf-8")) == ["python", "sql"]


def test_build_full_report_zip_includes_only_report_files(tmp_path):
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "note.txt").write_text("skip me", encoding="utf-8")
    sub = tmp_path / "charts"
    sub.mkdir()
    (sub / "chart.png").write_bytes(b"PNG")

    payload = admin_mod._build_full_report_zip(tmp_path)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
    assert any(n.endswith("report.json") for n in names)
    assert any(n.endswith("chart.png") for n in names)
    assert not any(n.endswith("note.txt") for n in names)


def test_monitoring_summary_computes_off_loop(monkeypatch):
    calls = {"n": 0}
    real_to_thread = asyncio.to_thread

    async def _spy(func, *args, **kwargs):
        calls["n"] += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _spy)
    monkeypatch.setattr(
        admin_mod,
        "_metrics_p95_and_busiest",
        lambda: ({"GET /api/x": 0.1}, [{"endpoint": "GET /api/x", "requests": 3}]),
    )
    monkeypatch.setattr(admin_mod, "get_pool", lambda: None)

    client = _client(monkeypatch)
    r = client.get("/admin/monitoring/summary")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["latency_p95_by_endpoint"] == {"GET /api/x": 0.1}
    assert body["busiest_by_requests"][0]["endpoint"] == "GET /api/x"
    assert calls["n"] >= 1


def test_monitoring_parses_off_loop(monkeypatch):
    calls = {"n": 0}
    real_to_thread = asyncio.to_thread

    async def _spy(func, *args, **kwargs):
        calls["n"] += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _spy)
    monkeypatch.setattr(
        admin_mod,
        "_parse_metric_families",
        lambda wanted: {"api_requests_total": {"name": "api_requests_total", "type": "counter", "samples": []}},
    )

    client = _client(monkeypatch)
    r = client.get("/admin/monitoring")
    assert r.status_code == 200, r.text
    assert "api_requests_total" in r.json()["families"]
    assert calls["n"] >= 1
