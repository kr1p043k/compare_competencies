"""KRM студента: направления + merged-вид без дублей."""
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from src.api_pkg import deps


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
        # каждый навык — один раз
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
