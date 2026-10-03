"""Unit (v45): shared vacancy filter builder numbering."""
from src.api_pkg.routers.vacancies import build_vacancy_where


def test_empty():
    assert build_vacancy_where() == ("TRUE", [])


def test_allgid():
    clause, params = build_vacancy_where(search="Python", experience="middle",
                                         region="Новосибирск", months=1)
    assert "($1)" not in clause  # sanity
    assert "$1" in clause and "$2" in clause and "$3" in clause and "$4" in clause
    assert params[0] == "middle" and params[1] == "python"
    assert params[2] == "Новосибирск"
    assert "area_name ILIKE" in clause
    assert "published_at >=" in clause


def test_region_all_ignored():
    clause, params = build_vacancy_where(region="all")
    assert (clause, params) == ("TRUE", [])


def test_partial_numbering():
    clause, params = build_vacancy_where(region="X")
    assert "$1" in clause and "$2" not in clause


def test_date_range():
    clause, params = build_vacancy_where(date_from="2026-05-01", date_to="2026-06-30")
    assert clause.count("published_at") == 2
    assert "$1" in clause and "$2" in clause and "$3" not in clause
    assert params[0].strftime("%Y-%m-%d") == "2026-05-01"
    assert params[1].strftime("%Y-%m-%d") == "2026-07-01"  # до — не включительно
    from datetime import timedelta, timezone
    assert params[0].utcoffset() == timedelta(hours=3)  # границы — московские


def test_date_garbage_ignored():
    assert build_vacancy_where(date_from="вчера")[0] == "TRUE"
    clause, params = build_vacancy_where(region="X", date_to="nope")
    assert "$1" in clause and "$2" not in clause  # мусор не съедает нумерацию
