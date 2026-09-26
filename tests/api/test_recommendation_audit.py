"""Admin audit trail for recommendation mutations.

Covers: teacher KRM recommendations CRUD, foundational add/delete, seed/auto,
skill suggest, admin whitelist add, categorize, suggestions approve/reject.
Every mutation must write an AUDIT entry with actor (email/role from token)
+ action + target, readable via GET /admin/logs (AdminDashboard Logs tab).
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src import config
from src.api_pkg import create_app
from src.api_pkg import request_logger as rl
from src.api_pkg.routers import teacher as t

ADMIN = {"uid": "00000000-0000-0000-0000-000000000001", "u": "admin@t.local", "r": "admin"}
TEACHER = {"uid": "00000000-0000-0000-0000-000000000002", "u": "teacher@t.local", "r": "teacher"}


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Isolate every JSON store the mutations touch (never real data/)."""
    import src.api_pkg.deps as deps
    import src.api_pkg.routers.admin as adm
    import src.cli.taxonomy_audit as ta

    recs = tmp_path / "teacher_recommendations.json"
    recs.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(config, "TEACHER_RECOMMENDATIONS_PATH", recs)
    flags = tmp_path / "flags.json"
    monkeypatch.setattr(t, "_foundational_path", lambda: flags)

    tax = tmp_path / "skill_taxonomy.json"
    tax.write_text(json.dumps({"categories": {"programming_languages": {"label": "PL", "skills": []}}}),
                   encoding="utf-8")
    monkeypatch.setattr(ta, "TAXONOMY_PATH", tax)
    skills = tmp_path / "it_skills.json"
    skills.write_text(json.dumps(["python"]), encoding="utf-8")
    monkeypatch.setattr(config, "IT_SKILLS_PATH", skills)
    monkeypatch.setattr(adm, "IT_SKILLS_PATH", skills)

    saved_set = set(getattr(deps, "current_skills_set", set()))
    yield {"recs": recs, "tax": tax, "skills": skills, "tmp": tmp_path}
    deps.current_skills_set = saved_set


@pytest.fixture()
def actor():
    """Mock session-checked user; switch roles via actor.return_value."""
    m = AsyncMock(return_value=dict(ADMIN))
    with patch("src.api_pkg.routers.auth.get_current_user", new=m):
        yield m


@pytest.fixture(autouse=True)
def _clean_buffer():
    rl._log_buffer.clear()
    yield
    rl._log_buffer.clear()


def _client():
    return TestClient(create_app())


def _audits(action_prefix=""):
    return [e for e in rl._log_buffer
            if e.method == "AUDIT" and (e.detail or "").startswith(action_prefix)]


def test_krm_add_writes_audit_with_actor(iso, actor):
    actor.return_value = dict(TEACHER)
    c = _client()
    r = c.post("/api/teacher/krm/recommendations",
               json={"discipline_id": "Math", "suggestion": "add graphs and trees"})
    assert r.status_code == 200, r.text
    rows = _audits("krm.recommendation.add")
    assert len(rows) == 1
    assert rows[0].user_email == "teacher@t.local"
    assert "teacher@t.local" in (rows[0].detail or "") and "(teacher)" in (rows[0].detail or "")
    assert "discipline=Math" in (rows[0].detail or "")
    assert "add graphs" in (rows[0].detail or "")


def test_krm_delete_writes_audit_with_target(iso, actor):
    actor.return_value = dict(TEACHER)
    c = _client()
    assert c.post("/api/teacher/krm/recommendations",
                  json={"discipline_id": "Math", "suggestion": "drop this"}).status_code == 200
    assert c.delete("/api/teacher/krm/recommendations/0").status_code == 200
    rows = _audits("krm.recommendation.delete")
    assert len(rows) == 1
    assert "index=0" in (rows[0].detail or "")
    assert "drop this" in (rows[0].detail or "")
    assert "teacher@t.local" in (rows[0].detail or "")


def test_foundational_add_delete_audit(iso, actor):
    c = _client()
    assert c.post("/api/teacher/krm/foundational", json={"skill": "Python"}).status_code == 200
    assert c.delete("/api/teacher/krm/foundational/python").status_code == 200
    adds = _audits("krm.foundational.add")
    dels = _audits("krm.foundational.delete")
    assert len(adds) == 1 and "skill=python" in (adds[0].detail or "")
    assert len(dels) == 1 and "skill=python" in (dels[0].detail or "")
    assert adds[0].user_email == "admin@t.local"


def test_seed_auto_writes_audit(iso, actor, monkeypatch):
    base = iso["tmp"] / "teacher_result"
    sub = base / "09.03.02" / "math"
    sub.mkdir(parents=True)
    (sub / "a.json").write_text(json.dumps({
        "discipline": "Math",
        "recommendations": [
            {"message": "add graphs", "priority": "high", "type": "add_new_content", "skill": "graphs"},
            {"message": "reviewBeginning algebra", "priority": "low", "type": "review_content", "skill": "algebra"},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(t, "_teacher_result_base", lambda: base)
    c = _client()
    r = c.post("/api/teacher/krm/recommendations/seed/auto?dir_code=09.03.02")
    assert r.status_code == 200, r.text
    assert r.json()["seeded"] == 2
    rows = _audits("krm.recommendations.seed_auto")
    assert len(rows) == 1
    assert "dir=09.03.02" in (rows[0].detail or "") and "seeded=2" in (rows[0].detail or "")


def test_suggest_writes_audit(iso, actor):
    actor.return_value = dict(TEACHER)
    c = _client()
    r = c.post("/api/teacher/skills/suggest",
               json={"skill": "Go", "category_hint": "programming_languages"})
    assert r.status_code == 200, r.text
    rows = _audits("skills.suggest")
    assert len(rows) == 1
    assert "skill=go" in (rows[0].detail or "")
    assert rows[0].user_email == "teacher@t.local"
    assert r.json()["suggestion"]["created_by"] == "teacher@t.local"


def test_approve_reject_audit_and_decider(iso, actor):
    from src.api_pkg.skill_suggestions import load_all

    actor.return_value = dict(TEACHER)
    c = _client()
    sid1 = c.post("/api/teacher/skills/suggest", json={"skill": "Rust"}).json()["suggestion"]["id"]
    sid2 = c.post("/api/teacher/skills/suggest", json={"skill": "Elixir"}).json()["suggestion"]["id"]

    actor.return_value = dict(ADMIN)
    r = c.post(f"/api/admin/skills/suggestions/{sid1}/approve", json={"category": "programming_languages"})
    assert r.status_code == 200, r.text
    r = c.post(f"/api/admin/skills/suggestions/{sid2}/reject", json={})
    assert r.status_code == 200, r.text

    by_id = {s["id"]: s for s in load_all()}
    assert by_id[sid1]["decided_by"] == "admin@t.local"
    assert by_id[sid2]["decided_by"] == "admin@t.local"

    ap = _audits("skills.suggestion.approve")
    rj = _audits("skills.suggestion.reject")
    assert len(ap) == 1 and f"id={sid1}" in (ap[0].detail or "") and "rust" in (ap[0].detail or "")
    assert "programming_languages" in (ap[0].detail or "")
    assert len(rj) == 1 and f"id={sid2}" in (rj[0].detail or "") and "elixir" in (rj[0].detail or "")


def test_categorize_and_whitelist_audit(iso, actor):
    c = _client()
    r = c.post("/api/admin/skills/categorize",
               json={"assignments": [{"skill": "go", "category": "programming_languages"}]})
    assert r.status_code == 200, r.text
    assert r.json()["added"] == 1
    r = c.post("/api/admin/whitelist/add", json={"skills": ["Go"]})
    assert r.status_code == 200, r.text
    assert r.json()["added"] == 1

    cat = _audits("skills.categorize")
    wl = _audits("skills.whitelist.add")
    assert len(cat) == 1 and "added=1" in (cat[0].detail or "") and "go" in (cat[0].detail or "")
    assert len(wl) == 1 and "added=1" in (wl[0].detail or "")
    assert all("admin@t.local" in (e.detail or "") for e in cat + wl)


def test_audit_readable_via_admin_logs(iso, actor):
    actor.return_value = dict(TEACHER)
    c = _client()
    assert c.post("/api/teacher/krm/recommendations",
                  json={"discipline_id": "Math", "suggestion": "visible via logs"}).status_code == 200
    r = c.get("/api/admin/logs?action=krm.recommendation.add&limit=100")
    assert r.status_code == 200, r.text
    logs = r.json()["logs"]
    assert len(logs) == 1
    assert logs[0]["method"] == "AUDIT"
    assert logs[0]["user"] == "teacher@t.local"
    assert logs[0]["detail"] and "visible via logs" in logs[0]["detail"]
    # unfiltered view carries the entry too
    all_logs = c.get("/api/admin/logs?limit=100").json()["logs"]
    assert any(l.get("detail", "") == logs[0]["detail"] for l in all_logs)


def test_audit_action_anonymous_fallback(actor):
    import asyncio

    from starlette.requests import Request

    actor.return_value = None
    scope = {"type": "http", "method": "POST", "path": "/api/x", "headers": [],
             "query_string": b"", "server": ("t", 80), "scheme": "http"}
    out = asyncio.run(rl.audit_action(Request(scope), "unit.probe", "target=y"))
    assert out["actor_email"] == "anonymous"
    assert out["detail"] == "unit.probe | target=y | by anonymous (?)"
    rows = _audits("unit.probe")
    assert len(rows) == 1 and rows[0].user_email == "anonymous"


def test_anonymous_actor_never_breaks_mutation(iso, actor):
    actor.return_value = None
    c = _client()
    r = c.post("/api/teacher/krm/recommendations",
               json={"discipline_id": "Math", "suggestion": "anon falls back"})
    # auth dependency still rejects unauthenticated callers at the gate
    assert r.status_code in (401, 403)
    assert _audits("krm.recommendation.add") == []
