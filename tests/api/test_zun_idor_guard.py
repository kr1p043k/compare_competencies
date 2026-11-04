"""IDOR guards on mutating /teacher/zun endpoints (upstream implementation).

Upstream routes every write through three helpers:
  _require_dir_access(request, dir_code)
  _ensure_competency_access(request, pool, competency_id)
  _ensure_discipline_access(request, pool, discipline_id)
  _ensure_ksa_access(request, pool, ksa_id)

They all end in _require_dir_access -> _current_user + _ensure_dir_access.
This module tests that chain directly: a teacher who owns 09.03.02 must be
refused (403) for rows of another direction, admins pass, anonymous gets 401.
"""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api_pkg.routers import krm_teacher, zun

OWN_DIR = "09.03.02"
OTHER_DIR = "01.03.01"


def _request() -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/teacher/zun/x",
        "raw_path": b"/api/teacher/zun/x",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("t", 80),
    }
    return Request(scope)


class Pool:
    """Answers only the direction lookup used by the guard."""

    def __init__(self, dir_code=OTHER_DIR):
        self.dir_code = dir_code

    async def fetchrow(self, sql, *a):
        return {"code": self.dir_code} if self.dir_code else None

    async def fetchval(self, sql, *a):
        return self.dir_code


@pytest.fixture()
def as_user(monkeypatch):
    """Подменяет слой аутентификации, повторяя семантику продакшена."""

    def _set(user, dirs=(OWN_DIR,)):
        async def fake_current(_request):
            # Прод-версия кидает 401 сама, а не возвращает None.
            if user is None:
                raise HTTPException(status_code=401, detail="Unauthorized")
            return user

        async def fake_dirs(u):
            # admin видит все направления, остальные — только свои.
            if (u or {}).get("r") == "admin":
                return [OWN_DIR, OTHER_DIR]
            return list(dirs)

        monkeypatch.setattr(zun, "_current_user", fake_current)
        monkeypatch.setattr(krm_teacher, "_user_directions", fake_dirs)

    return _set


TEACHER = {"uid": "u-t", "u": "t@example.local", "r": "teacher"}
ADMIN = {"uid": "u-a", "u": "a@example.local", "r": "admin"}


async def test_require_dir_access_allows_own(as_user):
    as_user(TEACHER)
    await zun._require_dir_access(_request(), OWN_DIR)


async def test_require_dir_access_denies_foreign(as_user):
    as_user(TEACHER)
    with pytest.raises(HTTPException) as exc:
        await zun._require_dir_access(_request(), OTHER_DIR)
    assert exc.value.status_code == 403


async def test_require_dir_access_admin_passes(as_user):
    as_user(ADMIN, dirs=())
    await zun._require_dir_access(_request(), OTHER_DIR)


async def test_require_dir_access_anonymous_is_401(as_user):
    as_user(None)
    with pytest.raises(HTTPException) as exc:
        await zun._require_dir_access(_request(), OWN_DIR)
    assert exc.value.status_code == 401


async def test_competency_guard_denies_foreign(as_user):
    as_user(TEACHER)
    with pytest.raises(HTTPException) as exc:
        await zun._ensure_competency_access(_request(), Pool(OTHER_DIR), "c-1")
    assert exc.value.status_code == 403


async def test_competency_guard_allows_own(as_user):
    as_user(TEACHER)
    await zun._ensure_competency_access(_request(), Pool(OWN_DIR), "c-1")


async def test_discipline_guard_denies_foreign(as_user):
    as_user(TEACHER)
    with pytest.raises(HTTPException) as exc:
        await zun._ensure_discipline_access(_request(), Pool(OTHER_DIR), "d-1")
    assert exc.value.status_code == 403


async def test_discipline_guard_allows_own(as_user):
    as_user(TEACHER)
    await zun._ensure_discipline_access(_request(), Pool(OWN_DIR), "d-1")


async def test_ksa_guard_denies_foreign(as_user):
    as_user(TEACHER)
    with pytest.raises(HTTPException) as exc:
        await zun._ensure_ksa_access(_request(), Pool(OTHER_DIR), "k-1")
    assert exc.value.status_code == 403


async def test_ksa_guard_allows_own(as_user):
    as_user(TEACHER)
    await zun._ensure_ksa_access(_request(), Pool(OWN_DIR), "k-1")


async def test_guards_skip_check_when_row_missing(as_user):
    # Нет строки — нет направления: 404 отдаст сам эндпоинт, а не guard.
    as_user(TEACHER)
    await zun._ensure_competency_access(_request(), Pool(None), "c-1")
    await zun._ensure_ksa_access(_request(), Pool(None), "k-1")


async def test_every_mutation_route_is_guarded():
    """Регрессия: каждый mutating zun-роут обязан вызывать guard."""
    import inspect

    src = inspect.getsource(zun)
    guarded = [
        ("zun_put_scope", "_require_dir_access"),
        ("zun_add_entry", "_ensure_competency_access"),
        ("zun_add_competency", "_ensure_discipline_access"),
        ("zun_link_skill", "_ensure_competency_access"),
        ("zun_patch_entry", "_ensure_ksa_access"),
        ("zun_delete_entry", "_ensure_ksa_access"),
        ("zun_import", "_require_dir_access"),
    ]
    for func, helper in guarded:
        i = src.index(f"async def {func}(")
        body = src[i:i + 2500]
        assert helper in body, f"{func} does not call {helper}"
