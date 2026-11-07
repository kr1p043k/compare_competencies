"""Route-level coverage for zun.py using a fake pool and auth override.

Covers the read-heavy gaps left by existing IDOR/write tests:
  * metadata: directions, my-directions
  * summaries: stats
  * tree: disciplines list, single tree, lookup by name
  * search/filter: text search, semantic fallback, competency filter
  * competency market links: skills list, coverage
  * results/status: analyze results from disk, pipeline status
  * additional write/import/scope success and failure branches
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.zun as zun_mod
from src.api_pkg import create_app
from tests.conftest import open_all_gates


@pytest.fixture(autouse=True)
def _allow_all_dirs(monkeypatch):
    """Bypass IDOR guards for route-logic tests."""

    async def _allow(*_args, **_kwargs):
        return None

    monkeypatch.setattr(zun_mod, "_require_dir_access", _allow)
    monkeypatch.setattr(zun_mod, "_ensure_competency_access", _allow)
    monkeypatch.setattr(zun_mod, "_ensure_discipline_access", _allow)
    monkeypatch.setattr(zun_mod, "_ensure_ksa_access", _allow)

    async def _fake_user(_request):
        return {"uid": "u-1", "u": "test@example.com", "r": "admin", "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)


def _norm_sql(text: str) -> str:
    """Схлопнуть пробелы: отступы в тестовых ключах и в коде не совпадают."""
    return "".join(text.split())


class FakePool:
    """Simple scriptable pool that records SQL and returns canned data."""

    def __init__(self, script=None):
        self.script = script or {}
        self.exec_log: list[str] = []

    async def fetchrow(self, sql, *a):
        self.exec_log.append(sql)
        flat = _norm_sql(sql)
        for key, val in self.script.get("fetchrow", []):
            if _norm_sql(key) in flat:
                return val() if callable(val) else val
        return None

    async def fetchval(self, sql, *a):
        self.exec_log.append(sql)
        flat = _norm_sql(sql)
        for key, val in self.script.get("fetchval", []):
            if _norm_sql(key) in flat:
                return val() if callable(val) else val
        return None

    async def fetch(self, sql, *a):
        self.exec_log.append(sql)
        flat = _norm_sql(sql)
        for key, val in self.script.get("fetch", []):
            if _norm_sql(key) in flat:
                return val() if callable(val) else val
        return []

    async def execute(self, sql, *a):
        self.exec_log.append(sql)
        return "OK"

    async def executemany(self, sql, rows):
        self.exec_log.append(sql)
        return None

    def acquire(self):
        pool = self

        class _Conn:
            async def executemany(self, sql, rows):
                await pool.executemany(sql, rows)

            def transaction(self):
                return _NullAsyncCtx()

        return _ConnCtx(_Conn())


class _NullAsyncCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _ConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


def _client(pool: FakePool):
    app = create_app()
    open_all_gates(app, role="admin")
    for lim in (app.state.limiter, zun_mod.limiter):
        lim.reset()
        lim._key_func = lambda _req: uuid.uuid4().hex
    zun_mod.get_pool = lambda: pool
    return TestClient(app)


DIR_ID = str(uuid.uuid4())
DISC_ID = str(uuid.uuid4())
COMP_ID = str(uuid.uuid4())
KSA_ID = str(uuid.uuid4())
SKILL_ID = str(uuid.uuid4())
RUN_ID = str(uuid.uuid4())

# ---------- metadata / summary ----------


def _direction_script():
    return {
        "fetch": [
            (
                "FROM directions d\n           LEFT JOIN disciplines",
                [{"code": "09.03.02", "name": "ИВТ", "profile": "бакалавриат", "disciplines_count": 3}],
            ),
            ("SELECT DISTINCT d.code\n           FROM directions", [{"code": "09.03.02"}]),
            ("SELECT d.code, COUNT(k.id)", [{"code": "09.03.02", "ksa_total": 10}]),
        ]
    }


def test_directions_lists():
    pool = FakePool(_direction_script())
    r = _client(pool).get("/api/teacher/zun/directions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    assert body[0]["dir_code"] == "09.03.02"
    assert body[0]["has_ksa"] is True
    assert body[0]["ksa_total"] == 10


def test_my_directions_admin_branch():
    pool = FakePool(_direction_script())
    r = _client(pool).get("/api/teacher/zun/my-directions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list)
    assert body[0]["dir_code"] == "09.03.02"


def test_my_directions_rop_branch(monkeypatch):
    async def _rop_user(_request):
        return {"uid": "u-rop", "u": "rop@example.com", "r": "rop", "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _rop_user)
    pool = FakePool({"fetch": [("FROM user_directions WHERE user_id", [{"dir_code": "09.03.02"}])]})
    r = _client(pool).get("/api/teacher/zun/my-directions")
    assert r.status_code == 200, r.text
    assert r.json() == [{"dir_code": "09.03.02"}]


def test_stats_shape_and_empty_direction_404():
    pool = FakePool(
        {
            "fetch": [
                ("SELECT id FROM directions WHERE code=", [{"id": DIR_ID}]),
                ("SELECT code, name, profile FROM directions", [{"code": "09.03.02", "name": "ИВТ", "profile": "бак"}]),
                (
                    "COUNT(DISTINCT disc.id) AS total_disciplines",
                    [
                        {
                            "total_disciplines": 5,
                            "total_competencies": 8,
                            "total_indicators": 2,
                            "ksa_knowledge": 3,
                            "ksa_abilities": 2,
                            "ksa_skills": 1,
                            "ksa_total": 6,
                            "competencies_with_ksa": 4,
                            "disciplines_with_ksa": 3,
                        }
                    ],
                ),
                ("FROM competency_skills cs", [{"n": 4}]),
                ("FROM coverage_analyses", [{"avg": 0.75, "covered": 3, "latest": datetime.now()}]),
            ],
        }
    )
    r = _client(pool).get("/api/teacher/zun/stats?dir_code=09.03.02")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dir_code"] == "09.03.02"
    assert body["ksa_counts"]["total"] == 6
    assert body["coverage"]["avg_coverage_ratio"] == 0.75

    assert _client(FakePool()).get("/api/teacher/zun/stats?dir_code=01.01.01").status_code == 404


def test_stats_invalid_dir_code():
    assert _client(FakePool()).get("/api/teacher/zun/stats?dir_code=boom").status_code == 400


# ---------- disciplines / tree ----------


def test_disciplines_list_and_not_found():
    disc_id = str(uuid.uuid4())
    pool = FakePool(
        {
            "fetch": [
                ("SELECT id FROM directions WHERE code=", [{"id": DIR_ID}]),
                (
                    "FROM disciplines disc\n            JOIN competencies c",
                    [
                        {
                            "id": disc_id,
                            "name": "Базы данных",
                            "semester": 3,
                            "control_form": "экзамен",
                            "comps": 2,
                            "inds": 1,
                            "ksa_total": 5,
                            "k_knowledge": 2,
                            "k_abilities": 2,
                            "k_skills": 1,
                            "linked_skills": 3,
                        }
                    ],
                ),
                (
                    "FROM coverage_analyses",
                    [
                        {
                            "discipline_id": disc_id,
                            "total_skills": 10,
                            "market_matched_skills": 5,
                            "coverage_ratio": 0.5,
                            "weighted_coverage": 0.3,
                            "analysis_date": datetime.now(),
                        }
                    ],
                ),
            ],
        }
    )
    c = _client(pool)
    r = c.get("/api/teacher/zun/disciplines?dir_code=09.03.02")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    assert body[0]["name"] == "Базы данных"
    assert body[0]["coverage_level"] == "medium"

    # Направление, которого нет: пул не отдаёт id направления -> 404.
    empty = FakePool()
    empty_client = _client(empty)
    assert empty_client.get("/api/teacher/zun/disciplines?dir_code=01.01.01").status_code == 404


def test_discipline_tree_by_id():
    comp_id = str(uuid.uuid4())
    pool = FakePool(
        {
            "fetchrow": [
                (
                    "FROM disciplines disc\n                  JOIN directions d",
                    {
                        "id": DISC_ID,
                        "name": "БД",
                        "semester": 3,
                        "control_form": "экзамен",
                        "direction_code": "09.03.02",
                    },
                ),
                ("FROM coverage_analyses\n           WHERE discipline_id", None),
            ],
            "fetch": [
                (
                    "FROM competencies c\n            WHERE c.discipline_id = $1",
                    [
                        {
                            "id": comp_id,
                            "code": "ПК-1",
                            "category": "ПК",
                            "name": "Базы данных",
                            "development_level": "базовый",
                            "sort_order": 1,
                            "parent_id": None,
                        }
                    ],
                ),
                ("FROM ksa_entries WHERE competency_id = ANY", []),
                ("FROM competency_skills cs\n               JOIN skills s", []),
            ],
        }
    )
    r = _client(pool).get(f"/api/teacher/zun/disciplines/{DISC_ID}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == DISC_ID
    assert body["coverage"] is None
    assert len(body["competencies"]) == 1


def test_discipline_by_name_fuzzy_and_not_found():
    pool = FakePool(
        {
            "fetch": [
                ("SELECT id FROM directions WHERE code=", [{"id": DIR_ID}]),
                ("SELECT id FROM disciplines\n           WHERE direction_id = ANY($1::uuid[]) AND name = $2", None),
                (
                    "SELECT id, name FROM disciplines\n                WHERE direction_id = ANY",
                    [
                        {"id": DISC_ID, "name": "Базы данных и СУБД"},
                    ],
                ),
            ],
            "fetchrow": [
                (
                    "FROM disciplines disc\n                  JOIN directions d",
                    {
                        "id": DISC_ID,
                        "name": "Базы данных и СУБД",
                        "semester": 3,
                        "control_form": "экзамен",
                        "direction_code": "09.03.02",
                    },
                ),
                ("FROM coverage_analyses\n           WHERE discipline_id", None),
            ],
            "fetchval": [],
        }
    )
    r = _client(pool).get("/api/teacher/zun/disciplines/name/Базы%20данных%20и%20СУБД?dir_code=09.03.02")
    assert r.status_code == 200, r.text

    not_found_pool = FakePool(
        {
            "fetch": [
                ("SELECT id FROM directions WHERE code=", [{"id": DIR_ID}]),
                ("SELECT id, name FROM disciplines", []),
            ],
        }
    )
    assert _client(not_found_pool).get("/api/teacher/zun/disciplines/name/Unknown?dir_code=09.03.02").status_code == 404


def test_tree_invalid_uuid_returns_400():
    assert _client(FakePool()).get("/api/teacher/zun/disciplines/not-a-uuid").status_code == 400


# ---------- search / filter ----------


def test_text_search_validations_and_success():
    pool = FakePool(
        {
            "fetch": [
                ("SELECT id FROM directions WHERE code=", [{"id": DIR_ID}]),
                (
                    "FROM ksa_entries k\n            JOIN competencies c",
                    [
                        {
                            "id": KSA_ID,
                            "ksa_type": "knowledge",
                            "original_text": "SQL",
                            "cleaned_text": None,
                            "competence_id": COMP_ID,
                            "competence_code": "ПК-1",
                            "parent_id": None,
                            "parent_code": None,
                            "discipline_id": DISC_ID,
                            "discipline_name": "БД",
                        }
                    ],
                ),
            ],
        }
    )
    c = _client(pool)
    r = c.get("/api/teacher/zun/search?q=SQL&dir_code=09.03.02&ksa_type=knowledge")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body[0]["original_text"] == "SQL"

    assert c.get("/api/teacher/zun/search?q=S&dir_code=09.03.02").status_code == 400
    assert c.get("/api/teacher/zun/search?q=SQL&dir_code=09.03.02&ksa_type=powers").status_code == 400


def test_semantic_fallback_and_validations():
    pool = FakePool(
        {
            "fetchval": [("SELECT MAX(k.created_at)", "")],
            "fetch": [
                ("SELECT id FROM directions WHERE code=$1", [{"id": DIR_ID}]),
                (
                    "FROM ksa_entries k "
                    "JOIN competencies c "
                    "JOIN competencies parent "
                    "JOIN disciplines disc "
                    "JOIN directions d "
                    "WHERE disc.direction_id = ANY",
                    [],
                ),
                (
                    "WHERE d.code = $2",
                    [
                        {
                            "id": KSA_ID,
                            "ksa_type": "skills",
                            "original_text": "Docker",
                            "created_at": None,
                            "competence_id": COMP_ID,
                            "competence_code": "КС-9",
                            "parent_id": None,
                            "parent_code": None,
                            "discipline_id": DISC_ID,
                            "discipline_name": "КС",
                            "code": "КС-9",
                        }
                    ],
                ),
            ],
        }
    )
    c = _client(pool)
    r = c.get("/api/teacher/zun/search/semantic?q=Docker&dir_code=09.03.02")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "fallback_substring"
    assert "matches" in body

    assert c.get("/api/teacher/zun/search/semantic?q=X&dir_code=09.03.02").status_code == 400
    assert c.get("/api/teacher/zun/search/semantic?q=Docker&dir_code=bad").status_code == 400


def test_filter_validations_and_success():
    pool = FakePool(
        {
            "fetch": [
                ("SELECT id FROM directions WHERE code=", [{"id": DIR_ID}]),
                (
                    "FROM competencies c\n            JOIN disciplines disc",
                    [
                        {
                            "id": COMP_ID,
                            "code": "ПК-1",
                            "category": "ПК",
                            "name": "БД",
                            "development_level": "базовый",
                            "discipline_id": DISC_ID,
                            "discipline_name": "Базы данных",
                            "ksa_total": 2,
                        }
                    ],
                ),
            ],
        }
    )
    c = _client(pool)
    r = c.get("/api/teacher/zun/filter?dir_code=09.03.02&category=ПК")
    assert r.status_code == 200, r.text
    assert len(r.json()) == 1

    assert c.get("/api/teacher/zun/filter?dir_code=09.03.02&category=ZZ").status_code == 400
    assert c.get("/api/teacher/zun/filter?dir_code=09.03.02&competence_id=not-uuid").status_code == 400
    # Несуществующее направление: пустой пул не отдаёт id направления -> 404.
    assert _client(FakePool()).get("/api/teacher/zun/filter?dir_code=01.01.01").status_code == 404


# ---------- competency skills / coverage ----------


def test_competency_skills_and_match_type():
    pool = FakePool(
        {
            "fetchrow": [
                (
                    "SELECT id FROM competencies WHERE id = $1",
                    {"id": COMP_ID, "discipline_id": DISC_ID, "discipline_name": "КС", "code": "КС-1"},
                )
            ],
            "fetch": [
                (
                    "FROM competency_skills cs\n             JOIN skills s",
                    [
                        {
                            "skill_id": SKILL_ID,
                            "skill_name": "Docker",
                            "match_type": "exact",
                            "source_text": "docker",
                            "ksa_type": "skills",
                        }
                    ],
                )
            ],
        }
    )
    c = _client(pool)
    r = c.get(f"/api/teacher/zun/competencies/{COMP_ID}/skills?match_type=exact")
    assert r.status_code == 200, r.text
    assert r.json()[0]["skill_name"] == "Docker"

    assert c.get(f"/api/teacher/zun/competencies/{COMP_ID}/skills?match_type=bad").status_code == 400
    # Чужая компетенция: пустой пул -> 404 на проверке существования.
    assert _client(FakePool()).get(f"/api/teacher/zun/competencies/{str(uuid.uuid4())}/skills").status_code == 404


def test_competency_coverage_with_and_without():
    pool = FakePool(
        {
            "fetchrow": [
                (
                    "FROM competencies c\n           JOIN disciplines disc",
                    {"id": COMP_ID, "code": "ПК-1", "discipline_id": DISC_ID, "discipline_name": "БД"},
                ),
                (
                    "FROM coverage_analyses\n           WHERE discipline_id = $1",
                    {
                        "total_skills": 10,
                        "market_matched_skills": 5,
                        "coverage_ratio": 0.5,
                        "analysis_date": datetime.now(),
                    },
                ),
            ],
        }
    )
    c = _client(pool)
    r = c.get(f"/api/teacher/zun/competencies/{COMP_ID}/coverage")
    assert r.status_code == 200, r.text
    assert r.json()["coverage"]["coverage_ratio"] == 0.5

    empty = FakePool(
        {
            "fetchrow": [
                (
                    "FROM competencies c\n           JOIN disciplines disc",
                    {"id": COMP_ID, "code": "ПК-1", "discipline_id": DISC_ID, "discipline_name": "БД"},
                ),
                ("FROM coverage_analyses\n           WHERE discipline_id = $1", None),
            ],
        }
    )
    assert _client(empty).get(f"/api/teacher/zun/competencies/{COMP_ID}/coverage").status_code == 200


# ---------- analyze results / status ----------


def test_analyze_results_summary_and_per_discipline(tmp_path, monkeypatch):
    monkeypatch.setattr(zun_mod, "RESULT_TEACHER", tmp_path)
    base = tmp_path / "09.03.02"
    base.mkdir()
    base.joinpath("_summary.json").write_text(json.dumps({"disciplines": 5}), encoding="utf-8")
    base.joinpath("Базы данных").mkdir()
    base.joinpath("Базы данных", "Базы данных.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    pool = FakePool(
        {
            "fetchrow": [("SELECT name FROM disciplines WHERE id=", {"name": "Базы данных"})],
        }
    )
    c = _client(pool)
    r = c.get("/api/teacher/zun/analyze/results?dir_code=09.03.02")
    assert r.status_code == 200, r.text
    assert r.json()["disciplines"] == 5

    r2 = c.get(f"/api/teacher/zun/analyze/results?dir_code=09.03.02&discipline_id={DISC_ID}")
    assert r2.status_code == 200, r2.text
    assert r2.json()["ok"] is True

    assert c.get("/api/teacher/zun/analyze/results?dir_code=09.03.02&discipline_id=not-uuid").status_code == 400


def test_analyze_results_no_summary_404(tmp_path, monkeypatch):
    monkeypatch.setattr(zun_mod, "RESULT_TEACHER", tmp_path)
    (tmp_path / "09.03.02").mkdir()
    assert _client(FakePool()).get("/api/teacher/zun/analyze/results?dir_code=09.03.02").status_code == 404


def test_analyze_status():
    pool = FakePool(
        {
            "fetchrow": [
                (
                    "FROM pipeline_runs WHERE id = $1",
                    {
                        "id": uuid.UUID(RUN_ID),
                        "action": "teacher-analysis",
                        "status": "completed",
                        "started_at": datetime.now(),
                        "completed_at": datetime.now(),
                        "error_message": None,
                        "stats": {"x": 1},
                    },
                )
            ],
        }
    )
    c = _client(pool)
    r = c.get(f"/api/teacher/zun/analyze/status/{RUN_ID}")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "completed"

    assert c.get("/api/teacher/zun/analyze/status/bad-uuid").status_code == 400
    assert _client(FakePool()).get(f"/api/teacher/zun/analyze/status/{RUN_ID}").status_code == 404


# ---------- write endpoints additional coverage ----------


def test_add_entry_success_and_validations():
    pool = FakePool(
        {
            "fetchrow": [
                ("FROM competencies c\n           JOIN disciplines", {"id": COMP_ID}),
                ("FROM ksa_entries\n           WHERE competency_id = $1", None),
            ],
            "fetchval": [
                ("SELECT COALESCE(MAX(sort_order)", 1),
                ("INSERT INTO ksa_entries", KSA_ID),
            ],
        }
    )
    c = _client(pool)
    r = c.post(f"/api/teacher/zun/competencies/{COMP_ID}/entries", json={"ksa_type": "skills", "text": "Docker"})
    assert r.status_code == 201, r.text
    assert r.json()["ksa_id"] == KSA_ID

    assert (
        c.post(f"/api/teacher/zun/competencies/{COMP_ID}/entries", json={"ksa_type": "bad", "text": "x"}).status_code
        == 400
    )
    assert (
        c.post(
            f"/api/teacher/zun/competencies/{COMP_ID}/entries", json={"ksa_type": "skills", "text": "  "}
        ).status_code
        == 400
    )
    # Несуществующая компетенция: пустой пул -> 404 на проверке владельца.
    assert (
        _client(FakePool())
        .post(f"/api/teacher/zun/competencies/{str(uuid.uuid4())}/entries", json={"ksa_type": "skills", "text": "x"})
        .status_code
        == 404
    )


def test_add_competency_success_and_validations():
    pool = FakePool(
        {
            "fetchrow": [
                ("FROM disciplines WHERE id = $1", {"id": DISC_ID}),
                ("FROM competencies WHERE discipline_id = $1 AND code = $2", None),
            ],
            "fetchval": [
                ("SELECT COALESCE(MAX(sort_order)", 1),
                ("INSERT INTO competencies", COMP_ID),
            ],
        }
    )
    c = _client(pool)
    r = c.post(f"/api/teacher/zun/disciplines/{DISC_ID}/competencies", json={"code": "ПК-9.9", "name": "DevOps"})
    assert r.status_code == 201, r.text
    assert r.json()["code"] == "ПК-9.9"

    assert (
        c.post(f"/api/teacher/zun/disciplines/{DISC_ID}/competencies", json={"code": "bad", "name": "x"}).status_code
        == 400
    )
    dup_pool = FakePool(
        {
            "fetchrow": [
                ("FROM disciplines WHERE id = $1", {"id": DISC_ID}),
                ("FROM competencies WHERE discipline_id = $1 AND code = $2", {"1": 1}),
            ],
        }
    )
    assert (
        _client(dup_pool)
        .post(f"/api/teacher/zun/disciplines/{DISC_ID}/competencies", json={"code": "ПК-9.9", "name": "x"})
        .status_code
        == 409
    )
    assert (
        _client(FakePool())
        .post(f"/api/teacher/zun/disciplines/{str(uuid.uuid4())}/competencies", json={"code": "ПК-1", "name": "x"})
        .status_code
        == 404
    )


def test_link_skill_success():
    pool = FakePool(
        {
            "fetchrow": [
                ("SELECT id FROM competencies WHERE id=$1", {"id": COMP_ID}),
                ("FROM skills WHERE LOWER(name)", None),
                ("FROM competency_skills cs", None),
            ],
            "fetchval": [("INSERT INTO skills", SKILL_ID)],
        }
    )
    r = _client(pool).post(
        f"/api/teacher/zun/competencies/{COMP_ID}/skills", json={"skill_name": "Docker", "ksa_type": "skills"}
    )
    assert r.status_code == 201, r.text
    assert r.json()["match_type"] == "exact"


def test_patch_and_delete_entry():
    patch_pool = FakePool(
        {
            "fetchrow": [
                ("UPDATE ksa_entries SET original_text", {"id": KSA_ID, "ksa_type": "skills", "competency_id": COMP_ID})
            ],
        }
    )
    c = _client(patch_pool)
    r = c.patch(f"/api/teacher/zun/entries/{KSA_ID}", json={"text": "Updated"})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "Updated"

    assert c.patch(f"/api/teacher/zun/entries/{KSA_ID}", json={"text": ""}).status_code == 400

    delete_pool = FakePool(
        {
            "fetchrow": [("DELETE FROM ksa_entries WHERE id = $1", {"competency_id": COMP_ID})],
        }
    )
    r = _client(delete_pool).delete(f"/api/teacher/zun/entries/{KSA_ID}")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "deleted"

    assert _client(FakePool()).delete(f"/api/teacher/zun/entries/{KSA_ID}").status_code == 404


# ---------- scope ----------


def test_scope_put_success_and_validations():
    pool = FakePool(
        {
            "fetch": [
                ("SELECT d2.name FROM disciplines d2", [{"name": "Базы данных"}, {"name": "Философия"}]),
                ("FROM discipline_scope", []),
            ],
        }
    )
    c = _client(pool)
    r = c.put(
        "/api/teacher/zun/scope",
        json={"dir_code": "09.03.02", "changes": [{"discipline_name": "Базы данных", "included": False}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["updated"] == 1

    assert c.put("/api/teacher/zun/scope", json={"dir_code": "09.03.02", "changes": []}).status_code == 400


# ---------- import / analyze ----------


def _write_krm(reference_dir: Path, dir_code: str):
    krm = {
        dir_code: {
            "disciplines": {
                "Базы данных": {
                    "ksa": {
                        "УК-1": {
                            "knowledge": ["знать SQL"],
                            "abilities": ["уметь проектировать"],
                            "skills": ["навык работы с PostgreSQL"],
                        }
                    }
                }
            }
        }
    }
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.joinpath(f"krm_disciplines_{dir_code}.json").write_text(json.dumps(krm), encoding="utf-8")


def test_import_dry_run_and_real(monkeypatch, tmp_path):
    monkeypatch.setattr(zun_mod.config, "REFERENCE_DIR", tmp_path)
    _write_krm(tmp_path, "09.03.02")
    dir_id = str(uuid.uuid4())
    disc_id = str(uuid.uuid4())
    comp_id = str(uuid.uuid4())
    pool = FakePool(
        {
            "fetch": [
                ("SELECT id, name FROM directions WHERE code=", [{"id": dir_id, "name": "ИВТ"}]),
                ("SELECT id, name FROM disciplines WHERE direction_id = ANY", [{"id": disc_id, "name": "Базы данных"}]),
                (
                    "SELECT id, discipline_id, code FROM competencies\n           WHERE discipline_id = ANY",
                    [{"id": comp_id, "discipline_id": disc_id, "code": "УК-1"}],
                ),
            ],
            "fetchval": [
                ("SELECT COALESCE(MAX(sort_order)", 1),
                ("INSERT INTO ksa_entries", None),
            ],
        }
    )
    c = _client(pool)
    r = c.post("/api/teacher/zun/import/09.03.02?dry_run=true")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["inserted"] == 3

    assert c.post("/api/teacher/zun/import/01.01.01?dry_run=true").status_code == 404

    r2 = c.post("/api/teacher/zun/import/09.03.02?dry_run=false")
    assert r2.status_code == 200, r2.text


def test_analyze_starts(monkeypatch):
    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    monkeypatch.setattr(zun_mod.asyncio, "create_subprocess_exec", lambda *a, **kw: _Proc())
    r = _client(FakePool()).post("/api/teacher/zun/analyze?dir_code=09.03.02")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "started"


def test_competency_search_shape():
    rows = [{"id": COMP_ID, "code": "ПК-1", "discipline_name": "Базы данных"}]
    pool = FakePool({"fetch": [("FROM competencies c\n           JOIN disciplines d2", rows)]})
    r = _client(pool).get("/api/teacher/zun/competencies/search?q=пк&dir_code=09.03.02")
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert items[0]["code"] == "ПК-1"
    assert items[0]["id"] == COMP_ID


# ---------- empty pool branches ----------


def test_empty_directions_and_lists():
    c = _client(FakePool())
    assert c.get("/api/teacher/zun/directions").json() == []
    assert c.get("/api/teacher/zun/disciplines?dir_code=09.03.02").status_code == 404
    assert c.get("/api/teacher/zun/filter?dir_code=09.03.02").status_code == 404
    assert c.get("/api/teacher/zun/competencies/search?q=x&dir_code=09.03.02").json()["items"] == []
