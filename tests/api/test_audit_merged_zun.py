"""Audit durability (buffer+DB merge) + ZUN mutation audit rows.

Covers: get_logs_merged fallback + DB union, zun.scope.put / entry.add /
entry.delete audit rows (representative for all 7 ZUN writers).
"""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from src.api_pkg import request_logger as rl

ADMIN = {"uid": "00000000-0000-0000-0000-000000000001", "u": "admin@t.local", "r": "admin"}


@pytest.fixture(autouse=True)
def _clean_buffer():
    rl._log_buffer.clear()
    yield
    rl._log_buffer.clear()


@pytest.fixture()
def actor():
    m = AsyncMock(return_value=dict(ADMIN))
    with patch("src.api_pkg.routers.auth.get_current_user", new=m):
        yield m


def _client():
    return TestClient(create_app())


def _entry(method="AUDIT", detail="krm.x | t | by a@b.c (admin)"):
    return rl.LogEntry(method=method, path="/api/x", status=200,
                       duration_ms=1, user_email="a@b.c", detail=detail)


def test_merged_falls_back_to_buffer_on_db_error(actor):
    rl._log_buffer.append(_entry())
    with patch("src.database.async_session_factory", side_effect=RuntimeError("db down")):
        import asyncio
        out = asyncio.run(rl.get_logs_merged(limit=10))
    assert len(out) == 1 and out[0]["detail"].startswith("krm.x")


def test_merged_unions_db_rows(actor):
    rl._log_buffer.append(_entry(detail="new-one | t | by a@b.c (admin)"))
    row = SimpleNamespace(method="AUDIT", path="/api/old", status=200, duration_ms=0,
                          user_email="old@b.c", source="backend",
                          detail="old-act | t | by old@b.c (admin)",
                          created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, q):
            m = MagicMock()
            m.scalars.return_value.all.return_value = [row]
            return m

    def fake_factory():
        return FakeSession()

    with patch("src.database.async_session_factory", fake_factory):
        import asyncio
        out = asyncio.run(rl.get_logs_merged(limit=10))
    details = [d["detail"] for d in out]
    assert any("new-one" in d for d in details)
    assert any("old-act" in d for d in details)
    # newest first: buffer entry (now) before 2020 row
    assert details[0].startswith("new-one")


def test_merged_action_filter_applies_to_db(actor):
    rl._log_buffer.append(_entry(detail="wanted | t | by a@b.c (admin)"))
    row = SimpleNamespace(method="AUDIT", path="/api/old", status=200, duration_ms=0,
                          user_email="old@b.c", source="backend",
                          detail="unwanted | t | by old@b.c (admin)",
                          created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, q):
            m = MagicMock()
            m.scalars.return_value.all.return_value = [row]
            return m

    with patch("src.database.async_session_factory", lambda: FakeSession()):
        import asyncio
        out = asyncio.run(rl.get_logs_merged(limit=10, action="wanted"))
    assert [d["detail"] for d in out] == ["wanted | t | by a@b.c (admin)"]


def _mock_pool(**answers):
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(side_effect=answers.get("fetchrow", []))
    pool.fetchval = AsyncMock(side_effect=answers.get("fetchval", []))
    pool.fetch = AsyncMock(return_value=answers.get("fetch", []))
    pool.execute = AsyncMock()
    return pool


def test_zun_scope_put_audits(actor):
    import src.api_pkg.routers.zun as z
    pool = _mock_pool(fetch=[{"name": "Math"}])
    c = _client()
    with patch.object(z, "get_pool", return_value=pool):
        r = c.put("/api/teacher/zun/scope",
                  json={"dir_code": "09.03.02",
                        "changes": [{"discipline_name": "Math", "included": True}]})
    assert r.status_code == 200, r.text
    rows = [e for e in rl._log_buffer if e.method == "AUDIT"]
    assert len(rows) == 1 and rows[0].detail.startswith("zun.scope.put")
    assert "admin@t.local" in rows[0].detail


def test_zun_entry_add_delete_audit(actor):
    import src.api_pkg.routers.zun as z
    kid = "11111111-1111-1111-1111-111111111111"
    cid = "22222222-2222-2222-2222-222222222222"
    c = _client()
    pool = _mock_pool(fetchrow=[{"id": cid}, None], fetchval=[1, kid])
    with patch.object(z, "get_pool", return_value=pool):
        r = c.post(f"/api/teacher/zun/competencies/{cid}/entries",
                   json={"ksa_type": "skills", "text": "линейная алгебра"})
    assert r.status_code == 201, r.text
    pool2 = _mock_pool(fetchrow=[{"competency_id": cid}])
    with patch.object(z, "get_pool", return_value=pool2):
        r = c.delete(f"/api/teacher/zun/entries/{kid}")
    assert r.status_code == 200, r.text
    details = [e.detail for e in rl._log_buffer if e.method == "AUDIT"]
    assert any((d or "").startswith("zun.entry.add") for d in details)
    assert any((d or "").startswith("zun.entry.delete") for d in details)
    assert all("admin@t.local" in (d or "") for d in details)
