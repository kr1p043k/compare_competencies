"""Консистентность whitelist и таксономии навыков (6.4.4).

Две группы тестов:

* синтетика - проверяем логику отчёта (повторы, неоднозначность, расхождение);
* реальные данные - фиксируем текущий уровень расхождений, чтобы новые правки
  не ухудшали картину незаметно. Базовые числа можно только уменьшать:
  если данные почистили, обновите константы ниже.
"""

import json

import pytest

from src import config
from src.analyzers.skills.skill_consistency import (
    BASELINE_AMBIGUOUS_SKILLS,
    BASELINE_DUPLICATE_CATEGORIES,
    BASELINE_TAXONOMY_ONLY,
    BASELINE_WHITELIST_ONLY,
    analyze,
    exceeds_baseline,
    format_report,
    load_taxonomy,
    load_whitelist,
    skill_key,
)
from src.analyzers.skills.skill_taxonomy import SkillTaxonomy

# Замер 2026-11-24 на данных Max_Opik_Dev. Растёт - значит появился новый дрейф.
# Значения берутся из модуля, чтобы CLI --strict и тест не расходились.
MAX_AMBIGUOUS_SKILLS = BASELINE_AMBIGUOUS_SKILLS
MAX_DUPLICATE_CATEGORIES = BASELINE_DUPLICATE_CATEGORIES
MAX_TAXONOMY_ONLY = BASELINE_TAXONOMY_ONLY
MAX_WHITELIST_ONLY = BASELINE_WHITELIST_ONLY


def _tax(categories: dict[str, list[str]]) -> dict:
    return {"categories": {cid: {"label": cid, "icon": "", "skills": skills} for cid, skills in categories.items()}}


class TestSkillKey:
    def test_case_and_spaces_normalized(self):
        assert skill_key("  Python ") == "python"
        assert skill_key("C++") == skill_key("c++")

    def test_inner_spaces_preserved(self):
        assert skill_key("ci / cd") != skill_key("ci/cd")

    def test_idempotent(self):
        for raw in ["Python", " 1C:ERP ", "Тонкая настройка"]:
            assert skill_key(skill_key(raw)) == skill_key(raw)


class TestAnalyze:
    def test_clean_data_has_no_problems(self):
        report = analyze(_tax({"langs": ["Python", "Java"], "db": ["PostgreSQL"]}), ["python", "java"])
        assert report.is_structurally_clean
        assert report.within_category_duplicates == {}
        assert report.ambiguous_skills == {}
        # PostgreSQL есть в таксономии, но не в whitelist.
        assert report.taxonomy_only == ("postgresql",)
        assert report.whitelist_only == ()

    def test_within_category_duplicates_detected(self):
        report = analyze(_tax({"langs": ["Java", "java", "Python"]}), ["java"])
        assert report.within_category_duplicates == {"langs": ("java",)}
        assert report.duplicate_categories == ("langs",)
        assert report.is_structurally_clean is False

    def test_ambiguous_skill_reported_with_all_categories(self):
        report = analyze(_tax({"frontend": ["JavaScript"], "langs": ["javascript"]}), ["javascript"])
        assert report.ambiguous_skills == {"javascript": ("frontend", "langs")}
        assert "ambiguous_skills=1" in report.structural_problems

    def test_drift_reported_both_directions(self):
        report = analyze(_tax({"langs": ["Python"]}), ["python", "rust"])
        assert report.taxonomy_only == ()
        assert report.whitelist_only == ("rust",)

        report = analyze(_tax({"langs": ["Python", "Rust"]}), ["python"])
        assert report.taxonomy_only == ("rust",)
        assert report.whitelist_only == ()

    def test_invalid_entries_are_reported_not_crashing(self):
        taxonomy = {"categories": {"langs": {"skills": ["Python", "", None, 42, "  "]}}}
        report = analyze(taxonomy, ["python"])
        assert len(report.invalid_entries) == 4
        assert report.taxonomy_entries == 1
        assert "invalid_entries=4" in report.structural_problems

    def test_missing_categories_key_is_tolerated(self):
        report = analyze({}, ["python"])
        assert report.categories == 0
        assert report.taxonomy_entries == 0
        assert report.whitelist_only == ("python",)

    def test_format_report_mentions_all_sections(self):
        report = analyze(_tax({"langs": ["Java", "java", "Rust"]}), ["java", "go"])
        text = format_report(report, limit=5)
        assert "within-category duplicates: 1" in text
        assert "whitelist without category" in text
        assert "taxonomy without whitelist" in text
        assert "go" in text
        assert "rust" in text

    def test_exceeds_baseline_flags_only_growth(self):
        clean = analyze(_tax({"langs": ["Python"]}), ["python"])
        assert exceeds_baseline(clean) == ()

        # Генерируем дрейф сверх базового уровня по каждому направлению:
        # одни и те же навыки в двух категориях + повторы внутри категорий.
        shared = [f"shared{i}" for i in range(BASELINE_AMBIGUOUS_SKILLS + 1)]
        many_categories: dict[str, list[str]] = {"a": list(shared), "b": list(shared)}
        for i in range(BASELINE_DUPLICATE_CATEGORIES + 1):
            many_categories[f"dup{i}"] = ["Same", "same"]
        drifted = analyze(_tax(many_categories), [])
        problems = exceeds_baseline(drifted)
        assert any("ambiguous_skills" in p for p in problems)
        assert any("duplicate_categories" in p for p in problems)
        assert any("taxonomy_only" in p for p in problems)


class TestRealData:
    """Инварианты на боевых файлах data/reference."""

    @pytest.fixture(scope="class")
    def report(self):
        return analyze(
            load_taxonomy(config.SKILL_TAXONOMY_PATH),
            load_whitelist(config.IT_SKILLS_PATH),
        )

    def test_whitelist_is_clean(self):
        whitelist = load_whitelist(config.IT_SKILLS_PATH)
        assert whitelist, "whitelist пуст"
        assert all(item.strip() for item in whitelist), "в whitelist есть пустые строки"
        # Сам whitelist хранится в нижнем регистре - иначе сравнение с таксономией
        # теряет смысл и появляются «дубли» вида Java/java.
        assert all(item == skill_key(item) for item in whitelist)

    def test_no_invalid_entries(self, report):
        assert report.invalid_entries == ()

    def test_categories_are_non_empty(self, report):
        assert report.categories >= 20
        assert report.taxonomy_unique > 1000
        assert report.whitelist_skills > 1000

    def test_drift_does_not_grow(self, report):
        assert len(report.ambiguous_skills) <= MAX_AMBIGUOUS_SKILLS
        assert len(report.within_category_duplicates) <= MAX_DUPLICATE_CATEGORIES
        assert len(report.taxonomy_only) <= MAX_TAXONOMY_ONLY
        assert len(report.whitelist_only) <= MAX_WHITELIST_ONLY
        assert exceeds_baseline(report) == ()

    def test_every_taxonomy_skill_resolves_to_known_category(self, report):
        taxonomy = load_taxonomy(config.SKILL_TAXONOMY_PATH)
        known = set(taxonomy["categories"])
        for cat in report.within_category_duplicates:
            assert cat in known
        for cats in report.ambiguous_skills.values():
            assert set(cats) <= known


class TestDeterminism:
    """Категория навыка не должна зависеть от порядка ключей в JSON."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        def clear():
            SkillTaxonomy._instance = None
            SkillTaxonomy._taxonomy = None
            SkillTaxonomy._skill_to_category = {}
            SkillTaxonomy._skill_to_categories = {}
            SkillTaxonomy._category_info = {}
            SkillTaxonomy._category_skills = {}

        clear()
        yield
        clear()

    def _write(self, tmp_path, monkeypatch, categories: dict[str, list[str]]) -> None:
        path = tmp_path / "skill_taxonomy.json"
        path.write_text(json.dumps(_tax(categories), ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr("src.config.SKILL_TAXONOMY_PATH", path)

    def test_last_category_wins_for_ambiguous_skill(self, tmp_path, monkeypatch):
        """Историческое правило: побеждает последняя категория в файле.

        На него опираются дедап ролей и имена кластеров (см. _build_closest_roles),
        поэтому тест фиксирует именно его - молчаливый флип сломает рекомендации.
        """
        self._write(tmp_path, monkeypatch, {"frontend": ["JavaScript"], "langs": ["JavaScript"]})
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category("javascript") == "langs"
        assert taxonomy.get_categories_for_skill("JavaScript") == ["frontend", "langs"]

    def test_category_follows_file_order_deterministically(self, tmp_path, monkeypatch):
        """Правило разрешения неоднозначности - «последняя категория в файле».

        Порядок ключей в JSON - часть данных, поэтому он осознанно влияет на
        результат. Проверяем, что выбор стабилен и повторная загрузка того же
        файла даёт тот же ответ.
        """
        self._write(tmp_path, monkeypatch, {"frontend": ["JavaScript"], "langs": ["JavaScript"]})
        assert SkillTaxonomy().get_category("javascript") == "langs"

        # Перезагрузка того же файла - тот же результат.
        SkillTaxonomy._instance = None
        SkillTaxonomy._skill_to_category = {}
        SkillTaxonomy._skill_to_categories = {}
        SkillTaxonomy._category_info = {}
        SkillTaxonomy._category_skills = {}
        assert SkillTaxonomy().get_category("javascript") == "langs"

        # Обратный порядок категорий - побеждает уже frontend, и это тоже стабильно.
        SkillTaxonomy._instance = None
        SkillTaxonomy._skill_to_category = {}
        SkillTaxonomy._skill_to_categories = {}
        SkillTaxonomy._category_info = {}
        SkillTaxonomy._category_skills = {}
        self._write(tmp_path, monkeypatch, {"langs": ["JavaScript"], "frontend": ["JavaScript"]})
        assert SkillTaxonomy().get_category("javascript") == "frontend"
        assert SkillTaxonomy().get_categories_for_skill("javascript") == ["langs", "frontend"]

    def test_repeated_loads_are_identical(self, tmp_path, monkeypatch):
        categories = {
            "langs": ["Python", "JavaScript"],
            "frontend": ["JavaScript", "React"],
            "db": ["PostgreSQL"],
        }
        self._write(tmp_path, monkeypatch, categories)
        first = {s: SkillTaxonomy().get_category(s) for s in ["python", "javascript", "react"]}

        for _ in range(3):
            SkillTaxonomy._instance = None
            SkillTaxonomy._skill_to_category = {}
            SkillTaxonomy._skill_to_categories = {}
            SkillTaxonomy._category_info = {}
            SkillTaxonomy._category_skills = {}
            again = {s: SkillTaxonomy().get_category(s) for s in first}
            assert again == first

    def test_within_category_case_duplicates_are_deduped(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, {"langs": ["Java", "java", "JAVA", "Python"]})
        skills = SkillTaxonomy().get_skills_in_category("langs").unwrap()
        assert skills == ["Java", "Python"]

    def test_invalid_skill_entries_are_skipped(self, tmp_path, monkeypatch):
        taxonomy = {"categories": {"langs": {"skills": ["Python", "", None]}}}
        path = tmp_path / "skill_taxonomy.json"
        path.write_text(json.dumps(taxonomy), encoding="utf-8")
        monkeypatch.setattr("src.config.SKILL_TAXONOMY_PATH", path)
        taxonomy_obj = SkillTaxonomy()
        assert taxonomy_obj.get_category("python") == "langs"
        assert taxonomy_obj.get_skills_in_category("langs").unwrap() == ["Python"]

    def test_unknown_skill_has_no_categories(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, {"langs": ["Python"]})
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_categories_for_skill("cobol") == []
        assert taxonomy.get_category("cobol") == "other"
