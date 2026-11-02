# tests/test_krm_merge_scripts.py
"""Регрессия трёх продовых инцидентов в KRM-скриптах (нигде не было тестов):

1. seed_all_directions._parse_comp_code: "DL-3" разворачивался в категорию
   "DL-3" целиком и падал на competencies_category_check.
2. merge_krm_aliases._merge: ksa {код: {тип: [тексты]}} сливался в плоский
   список, сид умирал на "'list' object has no attribute 'get'".
3. merge_annotations_to_krm._merge_whitespace_keys: дубль с двойным пробелом
   удалялся целиком, теряя код ПК-18 из анализа разрыва.

Скрипты - не пакет: грузим через importlib из scripts/. Модульный код у всех
трёх безопасен (argparse/asyncio под if __name__ == "__main__", engine в
src.database ленивый), поэтому exec_module не делает побочных эффектов.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def seed_mod():
    return _load("test_krm_seed_all_directions", "scripts/seed_all_directions.py")


@pytest.fixture(scope="module")
def alias_mod():
    return _load("test_krm_merge_aliases", "scripts/merge_krm_aliases.py")


@pytest.fixture(scope="module")
def ann_mod():
    return _load("test_krm_merge_annotations", "scripts/merge_annotations_to_krm.py")


# --------------------------------------------------------------------------
# A. seed_all_directions._parse_comp_code + _ALLOWED_CATEGORIES
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ПК-18", ("ПК", "18")),
        ("ОПК-2.1", ("ОПК", "2.1")),
        # Регрессия: было ("DL-3", "0") - категория "DL-3" валилась на
        # competencies_category_check и сид обрывался.
        ("DL-3", ("DL", "3")),
        ("ВПК-5", ("ВПК", "5")),
    ],
)
def test_parse_comp_code_known_codes(seed_mod, raw, expected):
    assert seed_mod._parse_comp_code(raw) == expected


@pytest.mark.parametrize("raw", ["???", "123", ""])
def test_parse_comp_code_garbage_category_not_allowed(seed_mod, raw):
    # Мусор не должен попадать в БД: категория вне _ALLOWED_CATEGORIES -
    # сид пропускает код (seed_direction: `if category not in ...: continue`).
    category, _number = seed_mod._parse_comp_code(raw)
    assert category not in seed_mod._ALLOWED_CATEGORIES


@pytest.mark.parametrize(
    ("raw", "category"),
    [("ПК-18", "ПК"), ("ОПК-2.1", "ОПК"), ("ВПК-5", "ВПК")],
)
def test_known_codes_map_to_allowed_categories(seed_mod, raw, category):
    parsed_category, _ = seed_mod._parse_comp_code(raw)
    assert parsed_category == category
    assert parsed_category in seed_mod._ALLOWED_CATEGORIES


def test_dl_category_is_not_allowed(seed_mod):
    # DL - legacy-категория прошлого набора: сид обязан её пропустить,
    # а не вставлять (иначе CheckViolation).
    assert "DL" not in seed_mod._ALLOWED_CATEGORIES
    assert seed_mod._parse_comp_code("DL-3")[0] == "DL"


# --------------------------------------------------------------------------
# B. merge_krm_aliases._merge (структуры skills/ksa)
# --------------------------------------------------------------------------


def test_merge_skills_union_dedupe(alias_mod):
    target = {"competencies": [], "skills": {"ПК-18": ["a", "b"]}, "ksa": {}}
    alias = {"competencies": [], "skills": {"ПК-18": ["b", "c"], "ОПК-2": ["x"]}}
    alias_mod._merge(alias, target)
    assert target["skills"] == {"ПК-18": ["a", "b", "c"], "ОПК-2": ["x"]}


def test_merge_ksa_keeps_dict_shape(alias_mod):
    # Регрессия "'list' object has no attribute 'get'": ksa обязан остаться
    # {код: {тип: [тексты]}} после слияния.
    target = {"competencies": [], "skills": {},
              "ksa": {"ОПК-1": {"knowledge": ["k1"], "abilities": [], "skills": []}}}
    alias = {"competencies": [], "skills": {},
             "ksa": {"ОПК-1": {"knowledge": ["k2"], "abilities": ["a1"], "skills": []}}}
    alias_mod._merge(alias, target)
    assert isinstance(target["ksa"]["ОПК-1"], dict)
    assert target["ksa"]["ОПК-1"]["knowledge"] == ["k1", "k2"]
    assert target["ksa"]["ОПК-1"]["abilities"] == ["a1"]


def test_merge_ksa_union_per_type(alias_mod):
    target = {"competencies": [], "skills": {},
              "ksa": {"ПК-5": {"knowledge": [], "abilities": [], "skills": ["s0"]}}}
    alias = {"competencies": [], "skills": {},
             "ksa": {"ПК-5": {"knowledge": ["k1"], "abilities": [], "skills": ["s1"]},
                     "ВПК-2": {"knowledge": ["k2"], "abilities": [], "skills": []}}}
    alias_mod._merge(alias, target)
    assert target["ksa"]["ПК-5"]["skills"] == ["s0", "s1"]
    assert target["ksa"]["ВПК-2"]["knowledge"] == ["k2"]
    assert set(target["ksa"]) == {"ПК-5", "ВПК-2"}


def test_merge_competencies_sorted_union(alias_mod):
    target = {"competencies": ["ПК-18"], "skills": {}, "ksa": {}}
    alias = {"competencies": ["ПК-18", "ОПК-1"], "skills": {}, "ksa": {}}
    alias_mod._merge(alias, target)
    assert target["competencies"] == sorted({"ОПК-1", "ПК-18"})


def test_merge_skips_non_dict_ksa_branch(alias_mod):
    # Защитная проверка на строках 74-75: если у алиаса ksa-значение не dict
    # (тот самый продовый битый формат), ветка пропускается, а не падает.
    target = {"competencies": [], "skills": {},
              "ksa": {"ОПК-1": {"knowledge": ["k1"], "abilities": [], "skills": []}}}
    alias = {"competencies": [], "skills": {},
             "ksa": {"ПК-9": ["плоский", "список", "вместо", "словаря"]}}
    alias_mod._merge(alias, target)
    assert isinstance(target["ksa"]["ОПК-1"], dict)
    assert "ПК-9" not in target["ksa"]


# --------------------------------------------------------------------------
# C. merge_krm_aliases._norm
# --------------------------------------------------------------------------


def test_norm_ii_abbreviation_matches_full_form(alias_mod):
    assert alias_mod._norm("систем ИИ") == alias_mod._norm(
        "систем искусственного интеллекта")


def test_norm_po_abbreviation_matches_full_form(alias_mod):
    assert alias_mod._norm("разработки ПО с") == alias_mod._norm(
        "разработки программного обеспечения с")


def test_norm_separators_collapsed(alias_mod):
    assert alias_mod._norm("AI_ML-систем") == alias_mod._norm("AI/ML-систем")


def test_norm_punctuation_stripped(alias_mod):
    assert alias_mod._norm("Технологии…™") == alias_mod._norm("Технологии…")


# --------------------------------------------------------------------------
# D. merge_annotations_to_krm._merge_whitespace_keys
# --------------------------------------------------------------------------


def test_whitespace_keys_collapse_to_one(ann_mod):
    double = "Инструменты и инженерные практики  разработки систем ИИ"
    single = "Инструменты и инженерные практики разработки систем ИИ"
    discs = {double: {"competencies": ["ОПК-1"], "skills": {}, "ksa": {}},
             single: {"competencies": [], "skills": {}, "ksa": {}}}
    merged = ann_mod._merge_whitespace_keys(discs)
    assert len(merged) == 1
    assert single in merged


def test_whitespace_merge_unions_competencies_keeps_pc18(ann_mod):
    # Регрессия: удаление дубля целиком теряло ПК-18 из анализа разрыва
    # (_load_krm строит индекс компетенций по skills).
    double = "Инструменты и инженерные практики  разработки систем ИИ"
    single = "Инструменты и инженерные практики разработки систем ИИ"
    discs = {double: {"competencies": ["ОПК-1"], "skills": {}, "ksa": {}},
             single: {"competencies": ["ПК-18"], "skills": {}, "ksa": {}}}
    merged = ann_mod._merge_whitespace_keys(discs)
    assert merged[single]["competencies"] == sorted({"ОПК-1", "ПК-18"})


def test_whitespace_merge_skills_ksa_both_sides(ann_mod):
    double = "Инструменты и инженерные практики  разработки систем ИИ"
    single = "Инструменты и инженерные практики разработки систем ИИ"
    discs = {
        double: {
            "competencies": ["ОПК-1"],
            "skills": {"ОПК-1": ["s1"]},
            "ksa": {"ОПК-1": {"knowledge": ["k1"], "abilities": [], "skills": []}},
        },
        single: {
            "competencies": ["ПК-18"],
            "skills": {"ПК-18": ["s2"], "ОПК-1": ["s3"]},
            "ksa": {"ПК-18": {"knowledge": ["k2"], "abilities": ["a1"], "skills": []},
                    "ОПК-1": {"knowledge": ["k3"], "abilities": [], "skills": []}},
        },
    }
    merged = ann_mod._merge_whitespace_keys(discs)
    body = merged[single]
    # skills обеих сторон, без дублей
    assert body["skills"] == {"ОПК-1": ["s1", "s3"], "ПК-18": ["s2"]}
    # ksa остаётся словарём словарей и содержит обе стороны
    assert isinstance(body["ksa"]["ОПК-1"], dict)
    assert isinstance(body["ksa"]["ПК-18"], dict)
    assert body["ksa"]["ОПК-1"]["knowledge"] == ["k1", "k3"]
    assert body["ksa"]["ПК-18"]["abilities"] == ["a1"]


# --------------------------------------------------------------------------
# E. merge_annotations_to_krm.discipline_from_file
# --------------------------------------------------------------------------


def test_discipline_from_file_collapses_double_space(ann_mod):
    # Двойной пробел в имени файла создавал дубль рядом с официальным
    # названием.
    got = ann_mod.discipline_from_file(
        "Аннотация РПД Практикум по подготовке  инженерной документации.pdf")
    assert got == "Практикум по подготовке инженерной документации"


def test_discipline_from_file_strips_pdf_and_prefix(ann_mod):
    got = ann_mod.discipline_from_file(
        "Аннотация РПД Численные методы оптимизации.pdf")
    assert got == "Численные методы оптимизации"
    # Префикс может идти с любым количеством пробелов и в любом регистре.
    assert ann_mod.discipline_from_file(
        "аннотация РПД  Информационные системы.pdf") == "Информационные системы"


def test_discipline_from_file_requires_annotacia_rpd_prefix(ann_mod):
    # Адаптировано к реальному контракту: discipline_from_file узнаёт
    # ТОЛЬКО префикс "Аннотация РПД " (ANN_RPD_PREFIX). Префиксы вида
    # "РПД_Название.pdf" она не strip-ает (это делает norm_name) и возвращает
    # None, чтобы add_missing не добавил мусор.
    assert ann_mod.discipline_from_file("РПД_Название.pdf") is None
    assert ann_mod.discipline_from_file("") is None
    assert ann_mod.discipline_from_file(None) is None
