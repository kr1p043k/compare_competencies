"""Self profile студента: свой уровень, свои навыки, удаление только своих."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from src.api_pkg import deps

STUDENT = {"uid": "00000000-0000-0000-0000-000000000009", "u": "stud@t.local", "r": "student"}


@pytest.fixture()
def actor():
    m = AsyncMock(return_value=dict(STUDENT))
    with patch("src.api_pkg.routers.auth.get_current_user", new=m):
        yield m


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    from src import config
    import src.api_pkg.routers.profiles as pr

    students = tmp_path / "students"
    students.mkdir()
    # Профили читаются из config.DATA_DIR: подменяем целиком на tmp.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    saved = dict(deps.student_profiles)
    deps.student_profiles.clear()
    yield {"dir": students}
    deps.student_profiles.clear()
    deps.student_profiles.update(saved)


def _client():
    return TestClient(create_app())


def test_self_create_and_patch(actor, iso):
    c = _client()
    r = c.get("/api/profiles/self")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skills"] == [] and body["target_level"] == "middle"

    r = c.patch("/api/profiles/self",
                json={"target_level": "senior", "add_skills": ["Python", "SQL"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_level"] == "senior"
    assert body["added"] == ["Python", "SQL"]
    assert body["refused"] == []
    # в памяти тоже (gap-анализ увидит)
    assert deps.student_profiles[body["profile"]].skills == ["Python", "SQL"]


def test_remove_only_own(actor, iso):
    c = _client()
    c.patch("/api/profiles/self", json={"add_skills": ["Python"]})
    # чужой навык (не из user_added) удалить нельзя
    r = c.patch("/api/profiles/self", json={"remove_skills": ["SQL"]})
    assert r.json()["refused"] == ["SQL"]
    assert r.json()["skills"] == ["Python"]
    r = c.patch("/api/profiles/self", json={"remove_skills": ["python"]})
    assert r.json()["removed"] == ["python"]
    assert r.json()["skills"] == []


def test_bad_level_rejected(actor, iso):
    c = _client()
    r = c.patch("/api/profiles/self", json={"target_level": "guru"})
    assert r.status_code == 400


def test_no_token_401(iso):
    c = _client()
    assert c.get("/api/profiles/self").status_code == 401
    assert c.patch("/api/profiles/self", json={}).status_code == 401


def test_self_route_not_shadowed(actor, iso):
    """GET /profiles/self не ловится маской /profiles/{profile}."""
    c = _client()
    r = c.get("/api/profiles/self")
    assert r.status_code == 200, r.text
    assert "profile" in r.json()


def test_patch_logs_student_history(actor, iso):
    """Правка self-профиля пишется в историю запросов студента."""
    import src.api_pkg.student_actions as sa

    sa._action_buffer.clear()
    c = _client()
    r = c.patch("/api/profiles/self", json={"add_skills": ["Go"]})
    assert r.status_code == 200, r.text
    hits = [e for e in sa._action_buffer
            if e.get("action_type") == "profile_edit" and e.get("username") == "stud@t.local"]
    assert len(hits) == 1
    assert "added=1" in hits[0].get("result_ref", "")
    sa._action_buffer.clear()


def test_history_chain_buffer_fallback():
    """Без БД история читается из буфера (не теряется молча)."""
    import src.api_pkg.student_actions as sa

    sa._action_buffer.clear()
    try:
        sa.log_action(username="stud@t.local", action_type="analysis",
                      profession="DS", profile="base")
        got = sa.get_actions(username="stud@t.local", limit=10)
        assert any(e.get("action_type") == "analysis" for e in got)
    finally:
        sa._action_buffer.clear()
