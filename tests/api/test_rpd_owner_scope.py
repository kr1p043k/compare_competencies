"""RPD runs must be scoped to their owner/direction (upstream implementation).

Upstream guards every status/cancel with rpd._require_run_access(request,
dir_code, owner_id): a set direction wins, otherwise the owner is compared,
admin always passes.
"""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api_pkg.routers import krm_teacher, rpd, zun

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


@pytest.fixture()
def as_user(monkeypatch):
    def _set(user, dirs=(OWN_DIR,)):
        async def fake_current(_request):
            return user

        async def fake_dirs(u):
            if (u or {}).get("r") == "admin":
                return [OWN_DIR, FOREIGN_DIR]
            return list(dirs)

        monkeypatch.setattr(zun, "_current_user", fake_current)
        monkeypatch.setattr(krm_teacher, "_user_directions", fake_dirs)
        monkeypatch.setattr(
            "src.api_pkg.routers.auth.get_current_user", fake_current
        )

    return _set


TEACHER = {"uid": "u-teacher", "r": "teacher"}
ADMIN = {"uid": "u-admin", "r": "admin"}
OTHER = {"uid": "u-other", "r": "teacher"}


async def test_admin_reads_any_run(as_user):
    as_user(ADMIN, dirs=())
    await rpd._require_run_access(_request(), FOREIGN_DIR, None)


async def test_teacher_reads_own_direction(as_user):
    as_user(TEACHER)
    await rpd._require_run_access(_request(), OWN_DIR, None)


async def test_teacher_denied_for_foreign_direction(as_user):
    as_user(TEACHER)
    with pytest.raises(HTTPException) as exc:
        await rpd._require_run_access(_request(), FOREIGN_DIR, None)
    assert exc.value.status_code == 403


async def test_owner_allowed_when_direction_missing(as_user):
    as_user(TEACHER)
    await rpd._require_run_access(_request(), None, "u-teacher")


async def test_foreign_owner_denied_when_direction_missing(as_user):
    as_user(TEACHER)
    with pytest.raises(HTTPException) as exc:
        await rpd._require_run_access(_request(), None, "u-other")
    assert exc.value.status_code == 403


async def test_legacy_run_without_owner_or_direction_is_denied(as_user):
    # fail closed: у старой записи нет ни направления, ни владельца.
    as_user(TEACHER)
    with pytest.raises(HTTPException) as exc:
        await rpd._require_run_access(_request(), None, None)
    assert exc.value.status_code == 403


async def test_anonymous_is_denied(as_user):
    as_user(None)
    with pytest.raises(HTTPException) as exc:
        await rpd._require_run_access(_request(), None, "u-teacher")
    assert exc.value.status_code in (401, 403)
