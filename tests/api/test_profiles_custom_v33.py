"""v33: custom profiles (tmp DATA_DIR, isolated registry, no auth on profiles router)."""
import json

import pytest
from fastapi.testclient import TestClient

from src import config
from src.api_pkg import create_app, deps


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "students").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed" / "competency_mapping.json").write_text(
        json.dumps({"UK-1": ["python", "sql"]}), encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(deps, "student_profiles", {})
    yield data_dir


def test_custom_profile_create_list_get(isolated):
    c = TestClient(create_app())
    r = c.post("/api/profiles/custom", json={
        "name": "my_ds", "target_level": "middle",
        "competencies": ["UK-1"], "skills": []})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["profile"] == "my_ds" and body["skills_count"] == 2
    f = isolated / "students" / "my_ds_competency.json"
    assert f.exists()
    assert "my_ds" in c.get("/api/profiles").json()["profiles"]
    got = c.get("/api/profiles/my_ds").json()
    assert got["competencies_count"] == 1 and got["skills_count"] == 2


def test_custom_profile_explicit_skills(isolated):
    c = TestClient(create_app())
    r = c.post("/api/profiles/custom", json={
        "name": "ops", "target_level": "senior",
        "competencies": [], "skills": ["linux", "bash"]})
    assert r.status_code == 201, r.text
    assert r.json()["skills_count"] == 2


def test_custom_profile_validations(isolated):
    c = TestClient(create_app())
    assert c.post("/api/profiles/custom", json={"name": "Bad Name!"}).status_code == 400
    assert c.post("/api/profiles/custom",
                  json={"name": "x", "skills": ["a"]}).status_code == 400  # too short
    assert c.post("/api/profiles/custom",
                  json={"name": "ok1", "target_level": "guru",
                        "skills": ["a"]}).status_code == 400
    assert c.post("/api/profiles/custom",
                  json={"name": "ok2"}).status_code == 400  # empty
    assert c.post("/api/profiles/custom",
                  json={"name": "dup", "skills": ["a"]}).status_code == 201
    assert c.post("/api/profiles/custom",
                  json={"name": "dup", "skills": ["a"]}).status_code == 409
