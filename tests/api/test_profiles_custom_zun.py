"""v34: custom profiles with ZUN (picked + new) + technologies + self techs + options.

Real TestClient/ASGI, isolated tmp DATA_DIR, auth bypass like v33 sibling.
"""
import json

import pytest
from fastapi.testclient import TestClient

from src import config
from src.api_pkg import create_app, deps
from src.models.student import StudentProfile


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "students").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed" / "competency_mapping.json").write_text(
        json.dumps({"UK-1": ["python", "sql"], "PK-1": ["docker"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(deps, "student_profiles", {})
    monkeypatch.setattr(deps, "skill_freq", {"python": 100, "sql": 80, "docker": 60, "git": 40})
    monkeypatch.setattr(deps, "skill_weights", {"python": 1.0, "sql": 0.8, "docker": 0.6})
    # reset slowapi in-memory counters so custom/post limits don't leak across tests
    try:
        from src.api_pkg.routers import profiles as _pr
        _pr.limiter._storage.reset()
    except Exception:
        pass
    yield data_dir


def _open_auth(app):
    for route in app.routes:
        dep = getattr(route, "dependant", None)
        for sub in (getattr(dep, "dependencies", None) or []):
            fn = sub.call
            if getattr(fn, "__qualname__", "").startswith("require_any_role"):
                app.dependency_overrides[fn] = lambda: {"r": "admin"}


def _client():
    app = create_app()
    _open_auth(app)
    return TestClient(app)


def test_custom_zun_create_picked_new_techs(isolated):
    deps.student_profiles["base"] = StudentProfile(
        profile_name="base", competencies=["UK-1"], skills=["python"], target_level="middle"
    )
    c = _client()
    payload = {
        "name": "zun_ds",
        "target_level": "middle",
        "base": "base",
        "competency_codes": ["UK-1"],
        "new_competencies": [
            {
                "code": "PK-9",
                "title": "Custom competence",
                "knowledge": ["know containers"],
                "abilities": ["run docker"],
                "skills": ["docker", "git"],
            }
        ],
        "technologies": ["docker", "python"],
    }
    r = c.post("/api/profiles/custom", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "zun_ds"
    assert body["file"] == "zun_ds_competency.json"
    assert body["counts"]["competencies"] == 2
    assert body["counts"]["technologies"] == 2
    # legacy keys still present for v33 compat
    assert body["profile"] == "zun_ds" and body["skills_count"] >= 2
    f = isolated / "students" / "zun_ds_competency.json"
    assert f.exists()
    data = json.loads(f.read_text(encoding="utf-8"))
    assert set(data["competencies"]) == {"UK-1", "PK-9"}
    assert "docker" in data["skills"] and "python" in data["skills"]
    assert set(data["technologies"]) == {"docker", "python"}
    assert data["new_competencies"][0]["code"] == "PK-9"
    # registered in memory like line ~111 flow
    assert "zun_ds" in deps.student_profiles
    assert "zun_ds" in c.get("/api/profiles").json()["profiles"]


def test_custom_zun_validations(isolated):
    c = _client()
    # bad name -> 400
    assert c.post("/api/profiles/custom", json={"name": "Bad Name!", "target_level": "middle"}).status_code == 400
    # new-style with zero competencies -> 422
    r = c.post("/api/profiles/custom", json={"name": "empty1", "target_level": "middle", "technologies": ["x"]})
    assert r.status_code == 422, r.text
    # duplicate codes across picked/new -> 422
    r = c.post("/api/profiles/custom", json={
        "name": "dup1", "target_level": "middle", "competency_codes": ["UK-1"],
        "new_competencies": [{"code": "UK-1", "knowledge": ["k"], "abilities": [], "skills": []}],
    })
    assert r.status_code == 422, r.text
    # ZUN empty entry -> 422
    r = c.post("/api/profiles/custom", json={
        "name": "badzun", "new_competencies": [{"code": "PK-2", "knowledge": ["  "], "abilities": [], "skills": []}],
    })
    assert r.status_code == 422, r.text
    # ZUN too long (>300) -> 422
    r = c.post("/api/profiles/custom", json={
        "name": "longzun", "new_competencies": [{"code": "PK-3", "knowledge": ["y" * 301], "abilities": [], "skills": []}],
    })
    assert r.status_code == 422, r.text
    # unknown base -> 404
    r = c.post("/api/profiles/custom", json={"name": "nobase", "target_level": "middle", "base": "ghost", "competency_codes": ["UK-1"]})
    assert r.status_code == 404, r.text
    # duplicate profile -> 409
    assert c.post("/api/profiles/custom", json={"name": "dup2", "target_level": "middle", "skills": ["a"]}).status_code == 201
    assert c.post("/api/profiles/custom", json={"name": "dup2", "target_level": "middle", "skills": ["a"]}).status_code == 409


def test_self_technologies_roundtrip(isolated, monkeypatch):
    async def fake_user(request):
        return {"u": "testuser@example.com", "r": "student"}

    import src.api_pkg.routers.auth as authmod
    monkeypatch.setattr(authmod, "get_current_user", fake_user)
    c = _client()
    got = c.get("/api/profiles/self").json()
    assert got["technologies"] == [] and got["skills"] == []
    r = c.patch("/api/profiles/self", json={
        "add_skills": ["python"], "add_technologies": ["docker", "python"],
    })
    assert r.status_code == 200, r.text
    assert set(r.json()["technologies"]) == {"docker", "python"}
    got = c.get("/api/profiles/self").json()
    assert set(got["technologies"]) == {"docker", "python"} and "python" in got["skills"]
    r = c.patch("/api/profiles/self", json={"remove_technologies": ["docker"]})
    assert r.status_code == 200
    assert r.json()["technologies"] == ["python"]
    # old per-user file without technologies key defaults to []
    (isolated / "students" / "self_testuser_example_com_competency.json").write_text(
        json.dumps({"skills": ["sql"]}), encoding="utf-8")
    got = c.get("/api/profiles/self").json()
    assert got["technologies"] == [] and got["skills"] == ["sql"]


def test_custom_options_shape(isolated):
    deps.student_profiles["base"] = StudentProfile(
        profile_name="base", competencies=["UK-1"], skills=["python"], target_level="middle"
    )
    c = _client()
    r = c.post("/api/profiles/custom", json={
        "name": "opt1", "target_level": "middle", "competency_codes": ["UK-1"],
        "new_competencies": [{"code": "PK-9", "title": "T", "knowledge": ["k"], "abilities": [], "skills": ["s1"]}],
        "technologies": ["docker"],
    })
    assert r.status_code == 201, r.text
    r = c.get("/api/profiles/custom/options")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["competencies"], list) and isinstance(body["technologies_suggest"], list)
    codes = {it["code"]: it for it in body["competencies"]}
    assert "UK-1" in codes and "PK-9" in codes
    assert codes["PK-9"]["title"] == "T" and codes["PK-9"]["skills_count"] >= 1
    assert codes["UK-1"]["skills_count"] == 2
    assert set(body["technologies_suggest"][:4]) == {"python", "sql", "docker", "git"} or "python" in body["technologies_suggest"]
    assert len(body["technologies_suggest"]) <= 30


def test_custom_options_suggest_fallback_nonempty(isolated, monkeypatch):
    import json as _json
    from src.api_pkg import deps as _deps
    monkeypatch.setattr(_deps, "skill_freq", {})
    monkeypatch.setattr(_deps, "skill_weights", {})
    (isolated / "processed" / "skill_weights.json").write_text(
        _json.dumps({"python": 0.9, "sql": 0.8, "docker": 0.7}), encoding="utf-8")
    c = _client()
    body = c.get("/api/profiles/custom/options").json()
    assert body["technologies_suggest"] == ["python", "sql", "docker"]
