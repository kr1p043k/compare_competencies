"""Покрытие роутера publications: фоновый глубокий поиск (backlog #49).

POST/GET/DELETE /publications/deep-search* не имели ни одного HTTP-теста:
grep «deep-search» по tests/ пуст (tests/test_deep_search.py гоняет
внутренности сканера, не роутер). Хаб не дёргаем: фоновую задачу подменяем
noop-функцией, словарь заданий pub._jobs чистим автофикстурой до/после.
Задания в тестах создаём напрямую через dataclass _DeepSearchJob.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.publications as pub
from src.api_pkg import create_app
from tests.conftest import open_all_gates


@pytest.fixture(autouse=True)
def _clean_jobs():
    """_jobs - словарь процесса: изоляция тестов друг от друга."""
    pub._jobs.clear()
    yield
    pub._jobs.clear()


async def _noop_scan(job_id, query, type_id, creator_id, max_pages):
    """Фоновый скан не ходит в сеть и не меняет статус задания."""
    return None


def _client(monkeypatch, *, role="teacher") -> TestClient:
    """Гейты открыты: fake-юзер {"u": "test", "r": role} попадает в user-параметр."""
    app = create_app()
    open_all_gates(app, role=role)
    monkeypatch.setattr(pub, "_run_deep_search", _noop_scan)
    for lim in (getattr(app.state, "limiter", None), getattr(pub, "limiter", None)):
        if lim is not None:
            lim.reset()
    return TestClient(app, raise_server_exceptions=False)


def _plain_client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _job(job_id="job-1", uid="test", status="done", **extra) -> pub._DeepSearchJob:
    now = pub._now()
    data = {
        "job_id": job_id,
        "uid": uid,
        "query": "нейросети",
        "status": status,
        "created_at": now,
        "updated_at": now,
    }
    data.update(extra)
    return pub._DeepSearchJob(**data)


# ------------------------------------------------------------- гейты (401)


def test_deep_search_endpoints_require_auth_401():
    """Все три ручки глубокого поиска - require_authenticated: без токена 401."""
    client = _plain_client()
    assert client.post("/api/publications/deep-search", params={"q": "нейро"}).status_code == 401
    assert client.get("/api/publications/deep-search/job-1").status_code == 401
    assert client.delete("/api/publications/deep-search/job-1").status_code == 401


# ------------------------------------------------- POST /publications/deep-search


def test_start_deep_search_202_registers_job(monkeypatch):
    client = _client(monkeypatch, role="teacher")
    r = client.post("/api/publications/deep-search", params={"q": "нейросети", "max_pages": 10})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "running"
    assert body["job_id"]
    job = pub._jobs[body["job_id"]]
    # uid берётся из user-параметра гейта (open_all_gates -> "test")
    assert job.uid == "test"
    assert job.query == "нейросети"
    assert job.status == "running"


def test_start_deep_search_single_job_per_user(monkeypatch):
    """Активное задание уже есть - второе не создаём, возвращаем то же."""
    client = _client(monkeypatch, role="teacher")
    first = client.post("/api/publications/deep-search", params={"q": "нейросети"})
    second = client.post("/api/publications/deep-search", params={"q": "другой запрос"})
    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    assert len(pub._jobs) == 1


def test_start_deep_search_capacity_503(monkeypatch):
    """Слоты заняты живыми заданиями (_register_job -> False): 503, без регистрации."""
    client = _client(monkeypatch, role="teacher")
    monkeypatch.setattr(pub, "_register_job", lambda job: False)
    r = client.post("/api/publications/deep-search", params={"q": "нейросети"})
    assert r.status_code == 503, r.text
    assert pub._jobs == {}


def test_start_deep_search_q_too_short_422(monkeypatch):
    """Граница: q min_length=2 - односимвольный запрос не принимает валидация."""
    client = _client(monkeypatch, role="teacher")
    r = client.post("/api/publications/deep-search", params={"q": "a"})
    assert r.status_code == 422
    assert pub._jobs == {}


@pytest.mark.parametrize("max_pages", [0, 501])
def test_start_deep_search_max_pages_422(monkeypatch, max_pages):
    """Граница: max_pages ge=1 le=500."""
    client = _client(monkeypatch, role="teacher")
    r = client.post("/api/publications/deep-search", params={"q": "нейро", "max_pages": max_pages})
    assert r.status_code == 422
    assert pub._jobs == {}


# ---------------------------------------------- GET /publications/deep-search/{id}


def test_get_unknown_job_404(monkeypatch):
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/publications/deep-search/no-such-job")
    assert r.status_code == 404
    assert r.json()["detail"] == "Задание не найдено"


def test_get_foreign_job_404_for_regular_user(monkeypatch):
    """Чужое задание студенту/преподавателю не светит: 404, не 403."""
    pub._jobs["job-1"] = _job(uid="someone-else")
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/publications/deep-search/job-1")
    assert r.status_code == 404


def test_get_foreign_job_visible_to_admin(monkeypatch):
    """Админ видит задания всех пользователей: прогресс и совпадения на месте."""
    pub._jobs["job-1"] = _job(
        uid="someone-else",
        status="done",
        match_count=1,
        matches=[{"id": 7}],
    )
    client = _client(monkeypatch, role="admin")
    r = client.get("/api/publications/deep-search/job-1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "job-1"
    assert body["status"] == "done"
    # PublicationOut достраивает недостающие поля - сверяем только id
    assert [m["id"] for m in body["matches"]] == [7]
    assert body["total"] == 1


def test_get_running_job_progress(monkeypatch):
    """Идёт скан: полный список не отдаём (matches []), прогресс - по счётчикам."""
    pub._jobs["job-1"] = _job(
        status="running",
        scanned_pages=5,
        total_pages=10,
        match_count=3,
        matches=[{"id": 7}],  # уже найденные - но не для running-ответа
    )
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/publications/deep-search/job-1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running"
    assert body["matches"] == []
    assert body["total"] == 3
    assert body["progress"] == {"scanned_pages": 5, "total_pages": 10, "pct": 50.0}


# ------------------------------------------- DELETE /publications/deep-search/{id}


def test_cancel_unknown_job_404(monkeypatch):
    client = _client(monkeypatch, role="teacher")
    r = client.delete("/api/publications/deep-search/no-such-job")
    assert r.status_code == 404


def test_cancel_running_sets_event(monkeypatch):
    job = _job(status="running")
    pub._jobs[job.job_id] = job
    client = _client(monkeypatch, role="teacher")
    r = client.delete(f"/api/publications/deep-search/{job.job_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "job-1"
    assert body["status"] == "cancelled"
    assert job.status == "cancelled"
    # фоновый скан сворачивается по этому флагу
    assert job.cancel_event.is_set()
    # запись остаётся в словаре: статус отмены пользователь ещё прочитает
    assert pub._jobs["job-1"] is job


def test_delete_done_job_keeps_status(monkeypatch):
    """Завершённое задание уже не отменить: статус не меняется, ответ 200."""
    job = _job(status="done")
    pub._jobs[job.job_id] = job
    client = _client(monkeypatch, role="teacher")
    r = client.delete("/api/publications/deep-search/job-1")
    assert r.status_code == 200, r.text
    assert r.json() == {"job_id": "job-1", "status": "done"}
    assert job.status == "done"
    assert not job.cancel_event.is_set()
