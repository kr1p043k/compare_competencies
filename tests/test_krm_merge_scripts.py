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
import json
import pathlib
import sys

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


# --------------------------------------------------------------------------
# F. merge_krm_aliases.main() - план/apply-цикл (продовый инцидент:
#    "KRM 55 vs DB 49" и удалённый двойник с потерей кодов ПК-18).
#
#    main() строит пути из модульной константы REFERENCE_DIR ВНУТРИ вызова
#    (scripts/merge_krm_aliases.py:92-93), поэтому monkeypatch константы на
#    временный каталог изолирует тест от реальных данных полностью: реальные
#    data/reference/*.json не открываются ни на чтение, ни на запись.
#    Проверено эмпирически: план в stdout собирается по временным файлам.
# --------------------------------------------------------------------------

REAL_KRM_PATH = ROOT / "data" / "reference" / "krm_disciplines_09.03.02.json"
REAL_OFFICIAL_PATH = ROOT / "data" / "reference" / "official_curriculum_09.03.02_2025.json"

OFFICIAL_NAME = ("Инструменты и инженерные практики разработки систем "
                 "искусственного интеллекта")
ALIAS_NAME = "Инструменты и инженерные практики разработки систем ИИ"
# norm() не совпадает с официальным (хвост "2025"), но fuzz.ratio ~ 0.97
# >= THRESHOLD (0.86) - уходит в fuzzy-ветку плана, а не в exact.
FUZZY_NAME = OFFICIAL_NAME + " 2025"


@pytest.fixture(autouse=True)
def guard_real_reference_bytes():
    """Ни один тест модуля не имеет права менять реальные reference-файлы.

    Снимок байтов до теста, побайтовое сравнение после. Тест, записавший
    хоть байт в реальную data/reference, роняет сравнение - страховка от
    повторения продового инцидента при правке самого скрипта.
    """
    watched = [p for p in (REAL_KRM_PATH, REAL_OFFICIAL_PATH) if p.exists()]
    before = {p: p.read_bytes() for p in watched}
    yield
    for p, snapshot in before.items():
        assert p.read_bytes() == snapshot, f"тест изменил реальный файл {p.name}"


def _make_ref_dir(tmp_path: pathlib.Path, official: list, discs: dict) -> pathlib.Path:
    """Собрать изолированный reference-каталог: official + KRM файлы."""
    (tmp_path / "official_curriculum_09.03.02_2025.json").write_text(
        json.dumps({"disciplines": official}, ensure_ascii=False), encoding="utf-8")
    krm_path = tmp_path / "krm_disciplines_09.03.02.json"
    krm_path.write_text(
        json.dumps({"09.03.02": {"year": 2025, "disciplines": discs}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    return krm_path


def _run_main(monkeypatch, alias_mod, krm_path: pathlib.Path, apply: bool) -> int:
    argv = ["merge_krm_aliases.py", "--direction", "09.03.02", "--year", "2025"]
    if apply:
        argv.append("--apply")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(alias_mod, "REFERENCE_DIR", krm_path.parent)
    return alias_mod.main()


def _discs_after(krm_path: pathlib.Path) -> dict:
    return json.loads(krm_path.read_text(encoding="utf-8"))["09.03.02"]["disciplines"]


def test_main_dry_run_leaves_krm_byte_identical(alias_mod, monkeypatch, tmp_path, capsys):
    body = {"competencies": ["ПК-18"], "skills": {"ПК-18": ["s1"]}, "ksa": {}}
    krm_path = _make_ref_dir(tmp_path, [OFFICIAL_NAME], {ALIAS_NAME: body})
    before = krm_path.read_bytes()

    assert _run_main(monkeypatch, alias_mod, krm_path, apply=False) == 0
    assert krm_path.read_bytes() == before, "dry-run обязан не писать KRM"
    out = capsys.readouterr().out
    # План построен ИМЕННО по временным данным: доказательство, что monkeypatch
    # REFERENCE_DIR сработал и реальный reference не читался.
    assert "к слиянию 1" in out
    assert ALIAS_NAME in out
    assert "(dry-run" in out


def test_main_apply_merges_alias_into_official_pc18_survives(alias_mod, monkeypatch, tmp_path):
    # Регрессия ПК-18: 12 skills жили только на аббревиатуре; слияние обязано
    # перенести их в официальную запись, а не удалить двойника целиком.
    alias_skills = [f"s_ali_{i:02d}" for i in range(1, 13)]
    discs = {
        OFFICIAL_NAME: {"competencies": ["ОПК-1"],
                        "skills": {"ПК-18": ["s_off_1", "s_off_2"]},
                        "ksa": {"ПК-18": {"knowledge": ["k_off"],
                                          "abilities": [], "skills": []}}},
        ALIAS_NAME: {"competencies": ["ПК-18"],
                     "skills": {"ПК-18": list(alias_skills)},
                     "ksa": {}},
    }
    krm_path = _make_ref_dir(tmp_path, [OFFICIAL_NAME], discs)

    assert _run_main(monkeypatch, alias_mod, krm_path, apply=True) == 0
    after = _discs_after(krm_path)
    assert list(after) == [OFFICIAL_NAME]  # ровно одна, в официальном написании
    body = after[OFFICIAL_NAME]
    assert body["competencies"] == ["ОПК-1", "ПК-18"]
    assert body["skills"]["ПК-18"] == ["s_off_1", "s_off_2"] + alias_skills
    assert body["ksa"]["ПК-18"]["knowledge"] == ["k_off"]


def test_main_apply_renames_alias_when_official_spelling_absent(alias_mod, monkeypatch,
                                                                tmp_path):
    # Официального написания в KRM нет: алиас ПЕРЕИМЕНОВЫВАЕТСЯ в него,
    # содержимое сохраняется дословно (rename-ветка, не удаление).
    body = {"competencies": ["ПК-18"],
            "skills": {"ПК-18": ["s1", "s2"]},
            "ksa": {"ПК-18": {"knowledge": ["k1"], "abilities": [], "skills": []}}}
    krm_path = _make_ref_dir(tmp_path, [OFFICIAL_NAME], {ALIAS_NAME: dict(body)})

    assert _run_main(monkeypatch, alias_mod, krm_path, apply=True) == 0
    after = _discs_after(krm_path)
    assert list(after) == [OFFICIAL_NAME]
    assert after[OFFICIAL_NAME] == body  # verbatim


def test_main_redirect_second_alias_merges_into_renamed_target(alias_mod, monkeypatch,
                                                               tmp_path):
    # Цепочка: ALIAS -> официальное имя, которого в KRM нет (rename), затем
    # FUZZY -> то же официальное имя: обязан СЛИТЬСЯ в уже переименованную
    # запись (resolve + повторная проверка tgt in discs после мутаций).
    # Логика, решающая "tgt in discs" по снимку до мутаций, перезаписала бы
    # переименованную запись и потеряла бы данные первой стороны - тот самый
    # продовый инцидент "удалённый двойник".
    body_b = {"competencies": ["ПК-18"], "skills": {"ПК-18": ["s_from_b"]}, "ksa": {}}
    body_a = {"competencies": ["ОПК-1"], "skills": {"ОПК-1": ["s_from_a"]}, "ksa": {}}
    krm_path = _make_ref_dir(tmp_path, [OFFICIAL_NAME],
                             {ALIAS_NAME: body_b, FUZZY_NAME: body_a})

    assert _run_main(monkeypatch, alias_mod, krm_path, apply=True) == 0  # без KeyError
    after = _discs_after(krm_path)
    assert list(after) == [OFFICIAL_NAME]  # транзитивно одна дисциплина
    body = after[OFFICIAL_NAME]
    assert body["skills"]["ПК-18"] == ["s_from_b"]  # данные первой стороны живы
    assert body["skills"]["ОПК-1"] == ["s_from_a"]  # данные второй стороны живы
    assert body["competencies"] == ["ОПК-1", "ПК-18"]


def test_main_apply_is_idempotent(alias_mod, monkeypatch, tmp_path):
    discs = {
        OFFICIAL_NAME: {"competencies": ["ОПК-1"],
                        "skills": {"ПК-18": ["s_off"]}, "ksa": {}},
        ALIAS_NAME: {"competencies": ["ПК-18"],
                     "skills": {"ПК-18": ["s_ali_1", "s_ali_2"]}, "ksa": {}},
    }
    krm_path = _make_ref_dir(tmp_path, [OFFICIAL_NAME], discs)

    assert _run_main(monkeypatch, alias_mod, krm_path, apply=True) == 0
    after_first = krm_path.read_bytes()
    first = _discs_after(krm_path)

    assert _run_main(monkeypatch, alias_mod, krm_path, apply=True) == 0
    assert krm_path.read_bytes() == after_first, "второй apply обязан быть no-op"
    assert _discs_after(krm_path) == first  # и ничего не потеряно


def test_main_apply_leaves_no_double_space_keys(alias_mod, monkeypatch, tmp_path):
    # Двойной пробел в ключе создавал дубль рядом с официальным названием;
    # удаление дубля целиком теряло его коды (инцидент ПК-18).
    double = "Численные методы  оптимизации"
    single = "Численные методы оптимизации"
    discs = {
        double: {"competencies": [], "skills": {"ПК-18": ["s_double"]}, "ksa": {}},
        single: {"competencies": ["ОПК-1"], "skills": {}, "ksa": {}},
    }
    krm_path = _make_ref_dir(tmp_path, [single], discs)

    assert _run_main(monkeypatch, alias_mod, krm_path, apply=True) == 0
    after = _discs_after(krm_path)
    assert list(after) == [single]
    assert not any("  " in key for key in after)  # двойных пробелов не осталось
    assert after[single]["skills"]["ПК-18"] == ["s_double"]  # двойник не потерян
