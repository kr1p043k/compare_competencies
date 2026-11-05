"""Тесты глубокого поиска по публикациям.

Обычный поиск читает максимум 50 страниц корпуса из ~26 781. Глубокий поиск
сканирует до сотен страниц в фоне, отдаёт прогресс и умеет отменяться.
Здесь проверяется и клиент (постраничное чтение, устойчивость к сбоям,
отмена через on_progress), и реестр задач (эвикция, истечение TTL).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.api_pkg.routers import publications as pub_mod
from src.infrastructure.hub_publications import HubPublicationsClient


def _client(monkeypatch, pages: dict[int, list[dict]], total_pages: int):
    """Клиент, у которого вместо сети — словарь страниц."""

    async def fake_fetch_raw(page, page_size, type_id=None, creator_id=None):
        if page not in pages:
            from src.infrastructure.hub_publications import HubNotFoundError

            raise HubNotFoundError(f"page {page} missing")
        meta = {"num_pages": total_pages}
        return meta, pages[page]

    client = HubPublicationsClient.__new__(HubPublicationsClient)
    client.base_url = "https://hub.test"
    client.token = None
    monkeypatch.setattr(client, "_fetch_raw", fake_fetch_raw)
    return client


def _pub(pid: int, title: str) -> dict:
    return {"id": pid, "payload": {"title": title}}


def _run(coro):
    # asyncio.run создаёт свежий цикл: get_event_loop() в общем прогоне может
    # вернуть уже закрытый цикл другого теста.
    return asyncio.run(coro)


class TestDeepSearchClient:
    def test_progress_is_monotonic_and_scan_stops_at_max_pages(self, monkeypatch):
        pages = {p: [_pub(p * 10 + i, f"нейросети обзор {p}-{i}") for i in range(5)] for p in range(1, 21)}
        client = _client(monkeypatch, pages, total_pages=100)

        seen_pages: list[int] = []

        def on_progress(scanned, total, matches):
            seen_pages.append(scanned)
            return True

        result = _run(client.deep_search("нейросети", max_pages=8, on_progress=on_progress))

        assert result["scanned_pages"] <= 8
        assert seen_pages == sorted(seen_pages), "прогресс должен быть неубывающим"
        assert result["truncated"] is True
        assert result["total_pages"] == 100
        assert result["total_scanned"] == 5 * result["scanned_pages"]

    def test_cancel_via_on_progress_false(self, monkeypatch):
        pages = {p: [_pub(p * 10, f"тема {p}") for _ in range(5)] for p in range(1, 31)}
        client = _client(monkeypatch, pages, total_pages=100)

        calls = {"n": 0}

        def on_progress(scanned, total, matches):
            calls["n"] += 1
            return False  # просим остановиться сразу

        result = _run(client.deep_search("тема", max_pages=30, on_progress=on_progress))

        assert calls["n"] == 1
        assert result["scanned_pages"] < 30
        assert result["truncated"] is True

    def test_failed_page_does_not_abort_whole_scan(self, monkeypatch):
        # Страницы 3 и 5 отсутствуют — скан должен дойти до конца диапазона.
        pages = {p: [_pub(p * 10, "навыки")] for p in (1, 2, 4, 6)}
        client = _client(monkeypatch, pages, total_pages=6)

        result = _run(client.deep_search("навыки", max_pages=6))

        assert result["scanned_pages"] >= 4
        assert result["total_scanned"] >= 4

    def test_not_found_first_page_returns_empty(self, monkeypatch):
        client = _client(monkeypatch, {}, total_pages=0)

        result = _run(client.deep_search("ничего", max_pages=5))

        assert result["matches"] == []
        assert result["scanned_pages"] == 0
        assert result["truncated"] is False

    def test_matches_are_case_insensitive(self, monkeypatch):
        pages = {1: [_pub(1, "НЕЙРОСЕТИ и обучение"), _pub(2, "про базы данных")]}
        client = _client(monkeypatch, pages, total_pages=1)

        result = _run(client.deep_search("нейросети", max_pages=1))

        assert result["total_scanned"] == 2
        assert len(result["matches"]) == 1


class TestJobRegistry:
    def _job(self, job_id: str, uid: str = "u1", status: str = "running", age: float = 0):
        return pub_mod._DeepSearchJob(
            job_id=job_id,
            uid=uid,
            query="q",
            status=status,
            created_at=time.time() - age,
            updated_at=time.time() - age,
        )

    def test_register_evicts_oldest_when_full(self, monkeypatch):
        monkeypatch.setattr(pub_mod, "_jobs", {})
        monkeypatch.setattr(pub_mod, "_MAX_JOBS", 2)

        assert pub_mod._register_job(self._job("a", status="done")) is True
        assert pub_mod._register_job(self._job("b", status="done")) is True
        # третий вытесняет один из завершённых
        assert pub_mod._register_job(self._job("c")) is True
        assert len(pub_mod._jobs) <= 2
        assert "c" in pub_mod._jobs

    def test_prune_expired_removes_finished_old_jobs(self, monkeypatch):
        monkeypatch.setattr(pub_mod, "_jobs", {})
        old = self._job("old", status="done", age=pub_mod._JOB_TTL_S + 10)
        fresh = self._job("fresh", status="done", age=1)
        pub_mod._jobs = {"old": old, "fresh": fresh}

        pub_mod._prune_expired()

        assert "old" not in pub_mod._jobs
        assert "fresh" in pub_mod._jobs

    def test_prune_keeps_running_jobs(self, monkeypatch):
        monkeypatch.setattr(pub_mod, "_jobs", {})
        running = self._job("run", status="running", age=pub_mod._JOB_TTL_S + 10)
        pub_mod._jobs = {"run": running}

        pub_mod._prune_expired()

        assert "run" in pub_mod._jobs
