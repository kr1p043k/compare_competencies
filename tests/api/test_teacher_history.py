"""GET /api/teacher/history: time series of snapshot summaries (no live DB).

Covers `src/api_pkg/routers/history.py`, which reads JSONL produced by
`scripts/snapshot_summary.py`:
- missing history file -> 200 with empty `points` and null first/last/delta;
- valid records -> oldest-first points, first/last set, delta between them;
- a malformed line is skipped and counted in `skipped_lines`;
- role guard: anonymous -> 401, student -> 403 (real `require_any_role` gate);
- the router's `HISTORY_DIR` is monkeypatched to tmp_path - `data/` is never
  written to or read from (the repo-level conftest only isolates
  `config.HISTORY_DIR`, which this router does not use).

Auth style follows tests/api/test_admin_guards.py; TestClient style follows
tests/api/test_teacher_students_pagination.py.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.history as history_mod
from src.api_pkg import create_app
from tests.conftest import open_all_gates, route_gate_roles

ENDPOINT = "/api/teacher/history"
DIR_CODE = "09.03.02"
YEAR = 2025


def _record(captured_at: str, coverage: float, gaps: int, disciplines: int) -> dict:
    """Compact snapshot record (schema of scripts/snapshot_summary.build_record)."""
    return {
        "captured_at": captured_at,
        "generated_at": captured_at,
        "year": YEAR,
        "code_version": "test",
        "total_disciplines": disciplines,
        "average_coverage": coverage,
        "average_quality_coverage": 0.0,
        "average_market_top_coverage": 0.0,
        "average_market_weight_coverage": 0.0,
        "total_gaps": gaps,
        "no_data_count": 0,
    }


def _write_jsonl(history_dir, lines: list[str]) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"{DIR_CODE}_{YEAR}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


@pytest.fixture
def client() -> TestClient:
    """Gates stubbed via open_all_gates (role=teacher): endpoint logic only."""
    app = open_all_gates(create_app(), role="teacher")
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def history_dir(monkeypatch, tmp_path):
    """Point the router's module-level HISTORY_DIR at tmp_path."""
    monkeypatch.setattr(history_mod, "HISTORY_DIR", tmp_path)
    return tmp_path


def test_missing_file_returns_empty_history(client, history_dir):
    r = client.get(ENDPOINT, params={"dir_code": DIR_CODE, "year": YEAR})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["points"] == []
    assert body["first"] is None
    assert body["last"] is None
    assert body["delta"] is None
    assert body["skipped_lines"] == 0


def test_two_valid_records_points_first_last_delta(client, history_dir):
    older = _record("2025-01-01T00:00:00+00:00", coverage=40.0, gaps=12, disciplines=20)
    newer = _record("2025-06-01T00:00:00+00:00", coverage=55.5, gaps=8, disciplines=22)
    # Append order is not guaranteed to be chronological: read_history must
    # sort by captured_at (oldest first), so write the newer record first.
    _write_jsonl(history_dir, [json.dumps(newer), json.dumps(older)])

    r = client.get(ENDPOINT, params={"dir_code": DIR_CODE, "year": YEAR})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["points"]) == 2
    assert [p["captured_at"] for p in body["points"]] == [
        "2025-01-01T00:00:00+00:00",
        "2025-06-01T00:00:00+00:00",
    ], "points must be ordered oldest to newest"
    assert body["first"] == older
    assert body["last"] == newer
    assert body["delta"] == {
        "average_coverage": 15.5,
        "total_gaps": -4,
        "total_disciplines": 2,
    }
    assert body["skipped_lines"] == 0


def test_malformed_line_is_skipped_and_counted(client, history_dir):
    first = _record("2025-01-01T00:00:00+00:00", coverage=30.0, gaps=5, disciplines=10)
    last = _record("2025-02-01T00:00:00+00:00", coverage=35.0, gaps=4, disciplines=11)
    _write_jsonl(
        history_dir,
        [json.dumps(first), "{not-valid-json", json.dumps(last)],
    )

    r = client.get(ENDPOINT, params={"dir_code": DIR_CODE, "year": YEAR})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skipped_lines"] == 1, "the malformed line must be counted"
    assert len(body["points"]) == 2
    assert body["first"] == first
    assert body["last"] == last
    assert body["delta"] == {
        "average_coverage": 5.0,
        "total_gaps": -1,
        "total_disciplines": 1,
    }


def test_limit_trims_points_but_full_range_first_last(client, history_dir):
    records = [
        _record("2025-01-01T00:00:00+00:00", coverage=10.0, gaps=9, disciplines=10),
        _record("2025-02-01T00:00:00+00:00", coverage=20.0, gaps=8, disciplines=10),
        _record("2025-03-01T00:00:00+00:00", coverage=30.0, gaps=7, disciplines=10),
    ]
    _write_jsonl(history_dir, [json.dumps(rec) for rec in records])

    r = client.get(ENDPOINT, params={"dir_code": DIR_CODE, "year": YEAR, "limit": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    # `points` is the limited window (oldest first)...
    assert len(body["points"]) == 1
    assert body["points"][0]["captured_at"] == "2025-01-01T00:00:00+00:00"
    # ...while first/last/delta still describe the FULL history.
    assert body["first"]["captured_at"] == "2025-01-01T00:00:00+00:00"
    assert body["last"]["captured_at"] == "2025-03-01T00:00:00+00:00"
    assert body["delta"]["average_coverage"] == 20.0


def test_invalid_dir_code_rejected_400(client, history_dir):
    """dir_code goes into a filename: bad format must be a 400, not a read."""
    r = client.get(ENDPOINT, params={"dir_code": "../evil", "year": YEAR})
    assert r.status_code == 400, r.text


def test_route_declares_teacher_role_gate():
    roles = route_gate_roles(create_app(), ENDPOINT, "GET")
    assert ("admin", "teacher", "rop") in roles, f"actual gates: {roles}"


def test_anonymous_request_gets_401(history_dir):
    # No open_all_gates here: the real gate must reject a tokenless request
    # before the handler touches any file.
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(ENDPOINT, params={"dir_code": DIR_CODE, "year": YEAR})
    assert r.status_code == 401, r.text


def test_student_role_gets_403(monkeypatch, history_dir):
    # Same trick as test_admin_guards: substitute get_current_user only, so
    # the REAL require_any_role checks the role and answers 403 itself.
    async def _student(_request):
        return {"uid": "s-1", "u": "student@example.com", "r": "student", "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _student)
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(ENDPOINT, params={"dir_code": DIR_CODE, "year": YEAR})
    assert r.status_code == 403, r.text
