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
