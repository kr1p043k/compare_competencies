"""Unit (v32): report staleness contract."""
from src.api_pkg.routers.teacher import report_staleness


def _meta(**kw):
    m = {"report_schema": 1, "code_version": 32, "vac_hash": "abc"}
    m.update(kw)
    return m


def test_fresh():
    assert report_staleness(_meta(), "abc", 32) == (False, "fresh")


def test_no_report_and_schema():
    assert report_staleness({}, "abc", 32)[0] is True
    assert report_staleness(_meta(report_schema=0), "abc", 32)[1] == "schema-mismatch"


def test_code_changed():
    stale, reason = report_staleness(_meta(code_version=31), "abc", 32)
    assert stale and reason == "code-changed:31->32"


def test_vacancies_changed_and_db_down():
    assert report_staleness(_meta(), "zzz", 32) == (True, "vacancies-changed")
    assert report_staleness(_meta(), None, 32) == (False, "db-unavailable-assumed-fresh")
