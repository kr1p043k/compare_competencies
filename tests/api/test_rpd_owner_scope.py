"""RPD runs must be scoped to their owner/direction.

Before the fix, /teacher/rpd/status/{run_id} and /teacher/rpd/cancel/{run_id}
accepted any run id, so a teacher could read and cancel another teacher's run.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api_pkg.routers import krm_teacher, rpd

OWN_DIR = "09.03.02"
FOREIGN_DIR = "01.03.01"


def _request() -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/teacher/rpd/status/x",
        "raw_path": b"/api/teacher/rpd/status/x",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("t", 80),
    }
    return Request(scope)


def _run(direction_code=None, owner_id=None):
    return SimpleNamespace(direction_code=direction_code, owner_id=owner_id)


@pytest.fixture()
def as_user(monkeypatch):
    def _set(user, dirs=(OWN_DIR,)):
        async def fake_user(_request):
            return user

        async def fake_dirs(_user):
            return list(dirs)

        monkeypatch.setattr(rpd, "get_current_user", fake_user)
        monkeypatch.setattr(krm_teacher, "_user_directions", fake_dirs)

    return _set


ADMIN = {"uid": "u-admin", "r": "admin"}
TEACHER = {"uid": "u-teacher", "r": "teacher"}


async def test_admin_can_read_any_run(as_user):
    as_user(ADMIN, dirs=())
    await rpd._assert_run_access(_request(), _run(direction_code=FOREIGN_DIR))


async def test_teacher_can_read_own_direction(as_user):
    as_user(TEACHER)
    await rpd._assert_run_access(_request(), _run(direction_code=OWN_DIR))


async def test_teacher_denied_for_foreign_direction(as_user):
    as_user(TEACHER)
    with pytest.raises(HTTPException) as exc:
        await rpd._assert_run_access(_request(), _run(direction_code=FOREIGN_DIR))
    assert exc.value.status_code == 403


async def test_unauthenticated_is_401(as_user):
    as_user(None)
    with pytest.raises(HTTPException) as exc:
        await rpd._assert_run_access(_request(), _run(direction_code=OWN_DIR))
    assert exc.value.status_code == 401


async def test_foreign_owner_denied_when_direction_missing(as_user):
    as_user(TEACHER)
    with pytest.raises(HTTPException) as exc:
        await rpd._assert_run_access(_request(), _run(owner_id="u-other"))
    assert exc.value.status_code == 403


async def test_own_run_allowed_when_direction_missing(as_user):
    as_user(TEACHER)
    await rpd._assert_run_access(_request(), _run(owner_id="u-teacher"))


async def test_legacy_run_without_owner_or_direction_allowed(as_user):
    # Данных для сверки нет: блокировать старые записи нельзя, иначе сломаем UI.
    as_user(TEACHER)
    await rpd._assert_run_access(_request(), _run())
