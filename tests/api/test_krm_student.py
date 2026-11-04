"""KRM студента: направления + merged-вид без дублей.

Плюс покрытие обоих эндпоинтов роутера `krm_student`:
`GET /krm/directions` и `GET /krm/student/competencies` - успешный ответ,
401 без пользователя, 400 на некорректный код направления, 404 на
неизвестное направление.

Приложение создаётся один раз на модуль (`create_app` ~0.2 c), иначе тесты
выходят за бюджет времени.
"""
import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.krm_student as krm_student_mod
from src.api_pkg import create_app, deps
from src.models.student import StudentProfile


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(autouse=True)
def _clear_overrides(app):
    yield
    app.dependency_overrides.clear()


def _override_auth(app, path: str, method: str = "GET"):
    for r in app.routes:
        if getattr(r, "path", "") == path and method in getattr(r, "methods", set()):
            for d in list(getattr(r, "dependencies", []) or []):
                fn = getattr(d, "dependency", None)
                if fn is not None and getattr(fn, "__name__", "") == "dependency":
                    app.dependency_overrides[fn] = lambda: {"u": "t", "r": "student"}
    return app


def _real_base_profile():
    from src.pipeline.runner import (
        build_profiles,
        load_competency_mapping,
        load_student_competencies,
    )
    mp = load_competency_mapping()
    profiles = build_profiles({"base": load_student_competencies("base")}, mp)
    return {"base": profiles["base"]}


def test_directions_listed(app):
    _override_auth(app, "/api/krm/directions")
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/krm/directions")
    assert r.status_code == 200, r.text[:200]
    codes = [d["dir_code"] for d in r.json()["directions"]]
    assert "09.03.02" in codes


def test_merged_no_duplicates(app, monkeypatch):
    from pathlib import Path

    from src import config as _cfg
    # conftest autouse уводит DATA_DIR в tmp: возвращаем настоящий для фикстуры.
    monkeypatch.setattr(_cfg, "DATA_DIR", Path("data").resolve())
    _override_auth(app, "/api/krm/student/competencies")
    app.dependency_overrides[deps.get_student_profiles] = _real_base_profile
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/krm/student/competencies",
                   params={"direction": "09.03.02", "profile": "base"})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["direction"] == "09.03.02"
    assert body["disciplines"], "ожидаются дисциплины КРМ"
    counts = body["counts"]
    assert counts["krm"] > 0 and counts["student"] > 0
    assert counts["merged"] == counts["krm"] + counts["student_only"]
    assert counts["overlap"] == sum(1 for m in body["merged"] if m["source"] == "both")
    # каждый навык - один раз
    norms = [m["skill"].strip().lower() for m in body["merged"]]
    assert len(norms) == len(set(norms))
    assert set(m["source"] for m in body["merged"]) <= {"krm", "student", "both"}


def test_unknown_profile_404(app):
    _override_auth(app, "/api/krm/student/competencies")
    app.dependency_overrides[deps.get_student_profiles] = lambda: {}
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/krm/student/competencies",
                   params={"direction": "09.03.02", "profile": "nope"})
    assert r.status_code == 404


# ─── покрытие обоих эндпоинтов роутера ────────────────────────────────────

async def _no_user(_request):
    return None


def test_directions_unauthorized_401(app, monkeypatch):
    monkeypatch.setattr(auth_mod, "get_current_user", _no_user)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/krm/directions")
    assert r.status_code == 401, r.text


def test_competencies_unauthorized_401(app, monkeypatch):
    monkeypatch.setattr(auth_mod, "get_current_user", _no_user)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/krm/student/competencies",
                   params={"direction": "09.03.02", "profile": "base"})
    assert r.status_code == 401, r.text


def test_directions_success_with_fake_data(app, monkeypatch):
    """Успешный ответ строится из _list_all_dirs (без файловой системы)."""
    fake = [{"dir_code": "09.03.02", "name": "ПИ", "profile": "base",
             "disciplines_count": 2}]
    monkeypatch.setattr(krm_student_mod, "_list_all_dirs", lambda: fake)
    _override_auth(app, "/api/krm/directions")
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/krm/directions")
    assert r.status_code == 200, r.text
    assert r.json()["directions"] == fake


def test_invalid_direction_code_400(app):
    """Некорректный формат кода направления отсекает _validate_dir_code."""
    _override_auth(app, "/api/krm/student/competencies")
    app.dependency_overrides[deps.get_student_profiles] = lambda: {}
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/krm/student/competencies",
                   params={"direction": "bad", "profile": "base"})
    assert r.status_code == 400, r.text


def test_unknown_direction_404(app):
    """Валидный по формату, но отсутствующий KRM-файл -> 404."""
    _override_auth(app, "/api/krm/student/competencies")
    app.dependency_overrides[deps.get_student_profiles] = lambda: {
        "base": StudentProfile(profile_name="base", skills=[]),
    }
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/krm/student/competencies",
                   params={"direction": "99.99.99", "profile": "base"})
    assert r.status_code == 404, r.text


def test_competencies_merged_payload(app, monkeypatch):
    """Успешный ответ: структура, дедуп и согласованность counts."""
    fake = {
        "direction_name": "Прикладная информатика",
        "disciplines": {
            "Базы данных": {
                "competencies": ["ПК-1"],
                "skills": {"ПК-1": ["Python", "SQL"]},
            },
            "Алгоритмы": {
                "competencies": ["ПК-2"],
                "skills": {"ПК-2": ["Docker"]},
            },
        },
    }
    monkeypatch.setattr(krm_student_mod, "_load_krm_direction", lambda code: fake)
    _override_auth(app, "/api/krm/student/competencies")
    app.dependency_overrides[deps.get_student_profiles] = lambda: {
        "base": StudentProfile(profile_name="base", skills=["python", "git"]),
    }
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/krm/student/competencies",
                   params={"direction": "09.03.02", "profile": "base"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["direction"] == "09.03.02"
    assert body["direction_name"] == "Прикладная информатика"
    assert body["profile"] == "base"
    assert len(body["disciplines"]) == 2

    counts = body["counts"]
    assert counts["krm"] == 3
    assert counts["student"] == 2
    assert counts["merged"] == counts["krm"] + counts["student_only"]
    assert counts["overlap"] == sum(
        1 for m in body["merged"] if m["source"] == "both"
    )
    # каждый навык встречается в merged ровно один раз
    norms = [m["skill"].strip().lower() for m in body["merged"]]
    assert len(norms) == len(set(norms))
    assert {m["source"] for m in body["merged"]} <= {"krm", "student", "both"}
