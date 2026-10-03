"""Teacher/ROP view of student skills (live local PG + TestClient, real DB).

Verification accounts (created here, deleted after):
  student: test.student@compare-competencies.local / Student_Test_123
  viewer:  test.teacher.view@compare-competencies.local / Teacher_View_123
Password hashes are generated per-row with pgcrypto
crypt(<pw>, gen_salt('bf', 6)) ($2a$06$ — same scheme as existing rows);
no hash is copied or reused.
Cleanup: sessions + user rows deleted, self files unlinked.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api_pkg import create_app

STUDENT_EMAIL = "test.student@compare-competencies.local"
STUDENT_PW = "Student_Test_123"
STUDENT_NAME = "Test Student"
TEACHER_EMAIL = "test.teacher.view@compare-competencies.local"
TEACHER_PW = "Teacher_View_123"
TEACHER_NAME = "Test Teacher Viewer"
SKILLS = ["Python", "SQL", "Docker"]


async def _drop_user(pool, email: str) -> None:
    await pool.execute(
        "DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE email = $1)",
        email,
    )
    await pool.execute("DELETE FROM users WHERE email = $1", email)


async def _make_user(pool, email: str, pw: str, role: str, name: str) -> None:
    await _drop_user(pool, email)
    pw_hash = await pool.fetchval("SELECT crypt($1, gen_salt('bf', 6))", pw)
    assert str(pw_hash).startswith("$2a$06$"), "unexpected hash scheme"
    await pool.execute(
        "INSERT INTO users (email, password_hash, full_name, role, is_active)"
        " VALUES ($1, $2, $3, $4, true)",
        email, pw_hash, name, role,
    )


def _clean_files() -> None:
    from src.api_pkg.routers.profiles import _self_profile_name
    for em in (STUDENT_EMAIL, TEACHER_EMAIL):
        try:
            Path(_self_profile_name(em)[1]).unlink(missing_ok=True)
        except Exception:
            pass


@pytest.fixture(scope="module")
def live():
    import anyio

    from src import db as _dbmod

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    # One portal loop for pool + all requests: the asyncpg pool is
    # loop-bound, so it must be created in the same loop TestClient uses.
    with anyio.from_thread.start_blocking_portal(backend="asyncio") as portal:
        client.portal = portal  # type: ignore[attr-defined]

        async def _setup():
            if _dbmod.pool is not None:
                await _dbmod.close_pool()
            pool = await _dbmod.create_pool()
            await _make_user(pool, STUDENT_EMAIL, STUDENT_PW, "student", STUDENT_NAME)
            await _make_user(pool, TEACHER_EMAIL, TEACHER_PW, "teacher", TEACHER_NAME)

        portal.call(_setup)
        s = client.post("/api/auth/login",
                        json={"email": STUDENT_EMAIL, "password": STUDENT_PW})
        t = client.post("/api/auth/login",
                        json={"email": TEACHER_EMAIL, "password": TEACHER_PW})
        yield {"client": client, "student_login": s, "teacher_login": t}

        async def _teardown():
            pool = _dbmod.pool
            if pool is not None:
                await _drop_user(pool, STUDENT_EMAIL)
                await _drop_user(pool, TEACHER_EMAIL)
                await _dbmod.close_pool()

        portal.call(_teardown)
    client.portal = None  # type: ignore[attr-defined]


@pytest.fixture()
def auth(live):
    s, t = live["student_login"], live["teacher_login"]
    assert s.status_code == 200, s.text
    assert t.status_code == 200, t.text
    return {
        "client": live["client"],
        "st": {"Authorization": f"Bearer {s.json()['token']}"},
        "tt": {"Authorization": f"Bearer {t.json()['token']}"},
    }


def _patch_skills(auth):
    c = auth["client"]
    try:
        r = c.patch("/api/profiles/self", headers=auth["st"],
                    json={"add_skills": SKILLS})
        assert r.status_code == 200, r.text
        assert r.json()["added"] == SKILLS
    finally:
        pass


def test_login_proves_student_and_teacher_roles(live):
    s, t = live["student_login"], live["teacher_login"]
    assert s.status_code == 200, s.text
    assert s.json()["role"] == "student"
    assert s.json()["token"]
    assert t.status_code == 200, t.text
    assert t.json()["role"] == "teacher"


def test_student_self_patch_and_view(auth):
    try:
        _patch_skills(auth)
        r = auth["client"].get("/api/profiles/self", headers=auth["st"])
        assert r.status_code == 200, r.text
        for sk in SKILLS:
            assert sk in r.json()["skills"]
    finally:
        _clean_files()


def test_teacher_sees_student_skills(auth):
    try:
        _patch_skills(auth)
        r = auth["client"].get("/api/admin/students/skills",
                               params={"email": STUDENT_EMAIL}, headers=auth["tt"])
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["email"] == STUDENT_EMAIL
        assert body["full_name"] == STUDENT_NAME
        for sk in SKILLS:
            assert sk in body["skills"]
            assert sk in body["user_added"]
        assert body["target_level"]
        assert body["updated_at"]
    finally:
        _clean_files()


def test_teacher_students_list_row(auth):
    try:
        _patch_skills(auth)
        r = auth["client"].get("/api/teacher/students", headers=auth["tt"])
        assert r.status_code == 200, r.text
        rows = {u["email"]: u for u in r.json()["students"]}
        assert STUDENT_EMAIL in rows
        row = rows[STUDENT_EMAIL]
        assert row["full_name"] == STUDENT_NAME
        assert row["skills_count"] == len(SKILLS)
        assert row["has_profile"] is True
    finally:
        _clean_files()


def test_guards_student_403_anon_401_unknown_404(auth):
    c = auth["client"]
    r = c.get("/api/admin/students/skills", params={"email": STUDENT_EMAIL},
              headers=auth["st"])
    assert r.status_code == 403, r.text
    r = c.get("/api/admin/students/skills", params={"email": STUDENT_EMAIL})
    assert r.status_code == 401, r.text
    r = c.get("/api/admin/students/skills",
              params={"email": "nobody@compare-competencies.local"}, headers=auth["tt"])
    assert r.status_code == 404, r.text
