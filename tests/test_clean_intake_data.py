"""Тесты очистки данных набора: двуязычные имена и слияние записей.

Мусорное имя ``Algorithmization and Programming (Алгоритмизация и
программирование)`` появилось из английского заголовка PDF: внешняя часть
латинская, в скобках - настоящее русское название, и рядом в базе уже есть
правильная дисциплина. Оба экземпляра надо свести в один, не потеряв
содержимое.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# Скрипты импортируют соседний _dsn; при запуске файлом эта папка
# уже в sys.path, при импорте из тестов её надо добавить руками.
sys.path.insert(0, str(ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "clean_intake_data", ROOT / "scripts" / "clean_intake_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["clean_intake_data"] = module
    spec.loader.exec_module(module)
    return module


clean = _load()


class TestDetectBilingualJunk:
    def test_latin_outer_cyrillic_inner_is_junk(self):
        got = clean.detect_bilingual_junk(
            "Algorithmization and Programming (Алгоритмизация и программирование)"
        )
        assert got == "Алгоритмизация и программирование"

    @pytest.mark.parametrize(
        "name",
        [
            "Базы данных и СУБД",
            "Алгоритмизация и программирование",
            "Python для научных вычислений и машинного обучения",
            "Машинное обучение (базовый уровень)",
            "",
        ],
    )
    def test_normal_names_are_not_flagged(self, name):
        assert clean.detect_bilingual_junk(name) is None

    def test_latin_inner_is_not_flagged(self):
        # Скобки с латиницей - это уточнение, а не перевод.
        assert clean.detect_bilingual_junk("Машинное обучение (ML)") is None


class TestMergeEntries:
    def test_unions_competencies_without_duplicates(self):
        target = {"competencies": ["ОПК-3"]}
        alias = {"competencies": ["ОПК-3", "ОПК-4"]}
        merged = clean.merge_entries(target, alias)
        assert merged["competencies"] == ["ОПК-3", "ОПК-4"]

    def test_skills_are_union_of_lists(self):
        target = {"skills": {"ОПК-3": ["sql"]}}
        alias = {"skills": {"ОПК-3": ["sql", "python"], "ОПК-4": ["etl"]}}
        merged = clean.merge_entries(target, alias)
        assert merged["skills"]["ОПК-3"] == ["sql", "python"]
        assert merged["skills"]["ОПК-4"] == ["etl"]

    def test_ksa_stays_dict_of_dicts(self):
        target = {"ksa": {"ОПК-3": {"knowledge": ["a"]}}}
        alias = {"ksa": {"ОПК-3": {"knowledge": ["a", "b"], "abilities": ["c"]}}}
        merged = clean.merge_entries(target, alias)
        assert isinstance(merged["ksa"]["ОПК-3"], dict)
        assert merged["ksa"]["ОПК-3"]["knowledge"] == ["a", "b"]
        assert merged["ksa"]["ОПК-3"]["abilities"] == ["c"]

    def test_empty_alias_does_not_break_target(self):
        target = {"competencies": ["ОПК-3"], "skills": {}, "ksa": {}}
        merged = clean.merge_entries(target, {})
        assert merged["competencies"] == ["ОПК-3"]


class TestPlanAction:
    def test_merge_when_target_exists(self):
        action = clean.plan_action(
            "Алгоритмизация и программирование",
            existing_names={"Algorithmization and Programming (Алгоритмизация и программирование)",
                            "Алгоритмизация и программирование"},
            official_names=set(),
            merge_target="Алгоритмизация и программирование",
        )
        assert action == "merge"

    def test_rename_when_name_is_official(self):
        action = clean.plan_action(
            "Алгоритмизация и программирование",
            existing_names={"Algorithmization and Programming (Алгоритмизация и программирование)"},
            official_names={"Алгоритмизация и программирование"},
            merge_target=None,
        )
        assert action == "rename"

    def test_skip_when_collision_with_different_course(self):
        # Целевое имя занято другой записью, которую не сливаем - трогать нельзя.
        action = clean.plan_action(
            "Базы данных",
            existing_names={"Algorithmization and Programming (Базы данных)",
                            "Базы данных"},
            official_names=set(),
            merge_target=None,
        )
        assert action == "skip"
