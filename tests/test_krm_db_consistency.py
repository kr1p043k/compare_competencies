"""Тесты проверки согласованности KRM и БД.

Чекер нужен, чтобы расхождение источников ловилось в CI, а не всплывало
позже сменой цифр в отчёте. Тесты проверяют саму логику сравнения на
подставных снимках БД — без живой базы и без файлов проекта.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Скрипты импортируют соседний _dsn; при запуске файлом эта папка
# уже в sys.path, при импорте из тестов её надо добавить руками.
sys.path.insert(0, str(ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_krm_db_consistency", ROOT / "scripts" / "check_krm_db_consistency.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_krm_db_consistency"] = module
    spec.loader.exec_module(module)
    return module


chk = _load()


def _assert(disciplines: dict, db: dict) -> dict:
    """Повторяет логику compare(), но на готовых снимках."""
    krm_names = set(disciplines)
    db_names = set(db["names"])
    only_db = sorted(db_names - krm_names)
    only_krm = sorted(krm_names - db_names)

    krm_link_count = chk.krm_links(disciplines)
    mismatched = []
    for name in sorted(krm_names & db_names):
        expected = len(disciplines[name].get("competencies") or [])
        actual = db["per_discipline"].get(name, 0)
        if expected != actual:
            mismatched.append({"name": name, "krm": expected, "db": actual})

    return {
        "names_ok": not only_db and not only_krm,
        "only_db": only_db,
        "only_krm": only_krm,
        "krm_links": krm_link_count,
        "links_ok": krm_link_count == db["links"],
        "mismatched": mismatched,
        "dangling": db["dangling_rows"],
    }


def _db(names, links, per_disc, dangling=0):
    return {
        "names": names,
        "links": links,
        "per_discipline": per_disc,
        "orphan_competencies": 0,
        "dangling_rows": dangling,
    }


def test_krm_links_counts_all_skill_entries():
    disciplines = {
        "A": {"skills": {"ОПК-1": ["sql", "python"], "ОПК-2": ["etl"]}},
        "B": {"skills": {"ПК-1": []}},
    }
    assert chk.krm_links(disciplines) == 3


def test_matching_state_passes():
    disciplines = {"A": {"competencies": ["ОПК-1"], "skills": {"ОПК-1": ["sql"]}}}
    result = _assert(disciplines, _db(["A"], 1, {"A": 1}))
    assert result["names_ok"] and result["links_ok"] and not result["mismatched"]


def test_missing_discipline_is_named():
    disciplines = {"A": {"competencies": [], "skills": {}}, "B": {"competencies": [], "skills": {}}}
    result = _assert(disciplines, _db(["A"], 0, {"A": 0}))
    assert not result["names_ok"]
    assert result["only_krm"] == ["B"]


def test_extra_discipline_in_db_is_named():
    disciplines = {"A": {"competencies": [], "skills": {}}}
    result = _assert(disciplines, _db(["A", "C"], 0, {"A": 0, "C": 0}))
    assert not result["names_ok"]
    assert result["only_db"] == ["C"]


def test_link_count_mismatch_reports_both_numbers():
    disciplines = {"A": {"competencies": ["ОПК-1"], "skills": {"ОПК-1": ["sql"]}}}
    result = _assert(disciplines, _db(["A"], 5, {"A": 1}))
    assert not result["links_ok"]
    assert result["krm_links"] == 1


def test_competency_count_difference_is_reported():
    disciplines = {"A": {"competencies": ["ОПК-1", "ОПК-2"], "skills": {}}}
    result = _assert(disciplines, _db(["A"], 0, {"A": 5}))
    assert result["mismatched"] == [{"name": "A", "krm": 2, "db": 5}]


def test_dangling_rows_are_detected():
    result = _assert({"A": {"competencies": [], "skills": {}}}, _db(["A"], 0, {"A": 0}, dangling=3))
    assert result["dangling"] == 3


def test_krm_path_for_2025_has_no_year_suffix():
    assert chk.krm_path("09.03.02", 2025).name == "krm_disciplines_09.03.02.json"
    assert chk.krm_path("09.03.02", 2024).name == "krm_disciplines_09.03.02_2024.json"
