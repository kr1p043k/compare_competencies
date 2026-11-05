"""KRM студента: направления + merged-вид без дублей.

Две части:
  * интеграционные тесты на реальных данных (create_app + REFERENCE_DIR);
  * изолированные тесты на свежем FastAPI с include_router и подменёнными
    auth / _list_all_dirs / _load_krm_direction / профилями (БД и сеть не трогаем).
"""
import contextlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.krm_student as krm_student_mod
from src.api_pkg import create_app, deps
from src.models.student import StudentProfile


def _override_auth(app, path: str, method: str = "GET"):
    from tests.conftest import open_all_gates

    # Раньше искали маршрут по app.routes и снимали гейт только на нём; на
    # FastAPI 0.162+ маршруты вложены в _IncludedRouter, поэтому путь не
    # находился. Снимаем все гейты общим хелпером.
    open_all_gates(app, role="student")
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


def test_directions_listed():
    app = _override_auth(create_app(), "/api/krm/directions")
    client = TestClient(app, raise_server_exceptions=False)
    try:
        r = client.get("/api/krm/directions")
        assert r.status_code == 200, r.text[:200]
        codes = [d["dir_code"] for d in r.json()["directions"]]
        assert "09.03.02" in codes
    finally:
        app.dependency_overrides.clear()


def test_merged_no_duplicates(monkeypatch):
    from pathlib import Path

    from src import config as _cfg
    # conftest autouse уводит DATA_DIR в tmp: возвращаем настоящий для фикстуры.
    monkeypatch.setattr(_cfg, "DATA_DIR", Path("data").resolve())
    app = _override_auth(create_app(), "/api/krm/student/competencies")
    app.dependency_overrides[deps.get_student_profiles] = _real_base_profile
    client = TestClient(app, raise_server_exceptions=False)
    try:
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
    finally:
        app.dependency_overrides.clear()


def test_unknown_profile_404():
    app = _override_auth(create_app(), "/api/krm/student/competencies")
    app.dependency_overrides[deps.get_student_profiles] = lambda: {}
    client = TestClient(app, raise_server_exceptions=False)
    try:
        r = client.get("/api/krm/student/competencies",
                       params={"direction": "09.03.02", "profile": "nope"})
        assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ===========================================================================
# Изолированные тесты: свежий app, include_router, подменённые зависимости.
# ===========================================================================

KRM_DATA = {
    "direction_name": "Информатика и вычислительная техника",
    "disciplines": {
        "Базы данных": {
            "competencies": ["ПК-1"],
            "skills": {"ПК-1": ["Python", "SQL"]},
        },
    },
}


@pytest.fixture(autouse=True)
def reset_limiter():
    with contextlib.suppress(Exception):
        krm_student_mod.limiter._storage.reset()
    yield


# Ключ своего профиля студента s@example.com: self_s_example_com.
# Тесты student-доступа должны содержать его, иначе роут честно отвечает 404.
SELF_KEY = "self_s_example_com"


def _profiles(*skills: str) -> dict[str, StudentProfile]:
    return {
        "base": StudentProfile(profile_name="base", skills=list(skills)),
        SELF_KEY: StudentProfile(profile_name=SELF_KEY, skills=list(skills)),
    }


def _mock_client(
    monkeypatch,
    *,
    role: str = "student",
    user: bool = True,
    profiles: dict | None = None,
    dirs: list | None = None,
    krm: dict | None = None,
) -> TestClient:
    async def _fake_user(_request):
        if not user:
            return None
        return {"uid": "u-1", "u": "s@example.com", "r": role}

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    if dirs is not None:
        monkeypatch.setattr(krm_student_mod, "_list_all_dirs", lambda: dirs)
    if krm is not None:
        monkeypatch.setattr(krm_student_mod, "_load_krm_direction", lambda _d: krm)

    app = FastAPI()
    app.state.limiter = krm_student_mod.limiter
    app.include_router(krm_student_mod.router)
    if profiles is not None:
        app.dependency_overrides[deps.get_student_profiles] = lambda: profiles
    return TestClient(app, raise_server_exceptions=False)


def test_mock_directions_shape(monkeypatch):
    client = _mock_client(monkeypatch, dirs=[{"dir_code": "09.03.02", "name": "ИВТ"}])
    r = client.get("/krm/directions")
    assert r.status_code == 200, r.text
    assert r.json() == {"directions": [{"dir_code": "09.03.02", "name": "ИВТ"}]}


def test_mock_directions_unauthorized(monkeypatch):
    client = _mock_client(monkeypatch, user=False, dirs=[])
    r = client.get("/krm/directions")
    assert r.status_code == 401, r.text


def test_mock_directions_forbidden_role(monkeypatch):
    client = _mock_client(monkeypatch, role="accountant", dirs=[])
    r = client.get("/krm/directions")
    assert r.status_code == 403, r.text


def test_competencies_merges_sources(monkeypatch):
    # Чужой профиль читает только учитель: студенту он теперь 403.
    client = _mock_client(
        monkeypatch,
        role="teacher",
        profiles=_profiles("Python", "Docker"),
        krm=KRM_DATA,
    )
    r = client.get(
        "/krm/student/competencies",
        params={"direction": "09.03.02", "profile": "base"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["direction_name"] == "Информатика и вычислительная техника"
    by_skill = {m["skill"].lower(): m for m in body["merged"]}
    assert by_skill["python"]["source"] == "both"
    assert by_skill["sql"]["source"] == "krm"
    assert by_skill["docker"]["source"] == "student"
    counts = body["counts"]
    assert counts == {
        "krm": 2,
        "student": 2,
        "overlap": 1,
        "merged": 3,
        "student_only": 1,
    }


def test_competencies_unknown_profile_mocked_404(monkeypatch):
    client = _mock_client(monkeypatch, profiles=_profiles("Python"), krm=KRM_DATA)
    r = client.get(
        "/krm/student/competencies",
        params={"direction": "09.03.02", "profile": "ghost"},
    )
    assert r.status_code == 404, r.text
    assert "не найден" in r.json()["detail"]


def test_competencies_foreign_profile_forbidden_for_student(monkeypatch):
    """Студент не читает чужой профиль, даже если тот существует."""
    client = _mock_client(monkeypatch, profiles=_profiles("Python"), krm=KRM_DATA)
    r = client.get(
        "/krm/student/competencies",
        params={"direction": "09.03.02", "profile": "base"},
    )
    assert r.status_code == 403, r.text


def test_competencies_foreign_profile_allowed_for_teacher(monkeypatch):
    client = _mock_client(
        monkeypatch, role="teacher", profiles=_profiles("Python"), krm=KRM_DATA
    )
    r = client.get(
        "/krm/student/competencies",
        params={"direction": "09.03.02", "profile": "base"},
    )
    assert r.status_code == 200, r.text


def test_competencies_invalid_direction_400(monkeypatch):
    client = _mock_client(monkeypatch, profiles=_profiles("Python"), krm=KRM_DATA)
    r = client.get(
        "/krm/student/competencies",
        params={"direction": "not-a-code", "profile": "base"},
    )
    assert r.status_code == 400, r.text


def test_competencies_without_query_returns_defaults(monkeypatch):
    """Без direction/profile эндпоинт отдаёт доступное направление и свой профиль.

    Раньше здесь был 422: студент открывает страницу первым и не знает ни кода
    направления, ни служебного имени профиля. Теперь сервер подставляет то, к
    чему у пользователя есть доступ.
    """
    client = _mock_client(monkeypatch, profiles=_profiles("Python"), krm=KRM_DATA)
    r = client.get("/krm/student/competencies")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["direction"]
    assert body["profile"]
    assert "disciplines" in body


def test_competencies_krm_unavailable_503(monkeypatch):
    from fastapi import HTTPException

    def _boom(_d):
        raise HTTPException(status_code=503, detail="KRM data unavailable")

    client = _mock_client(
        monkeypatch, role="teacher", profiles=_profiles("Python")
    )
    monkeypatch.setattr(krm_student_mod, "_load_krm_direction", _boom)
    r = client.get(
        "/krm/student/competencies",
        params={"direction": "09.03.02", "profile": "base"},
    )
    assert r.status_code == 503, r.text


def test_mock_directions_rate_limited_after_60(monkeypatch):
    client = _mock_client(monkeypatch, dirs=[])
    for _ in range(60):
        assert client.get("/krm/directions").status_code == 200
    assert client.get("/krm/directions").status_code == 429
