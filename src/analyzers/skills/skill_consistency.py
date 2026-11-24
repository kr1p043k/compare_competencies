"""Консистентность whitelist навыков и таксономии (6.4.4).

Два источника правды живут отдельными файлами:

* ``data/reference/it_skills.json`` - плоский whitelist IT-навыков (нормализован
  к нижнему регистру), по нему фильтруется рынок;
* ``data/reference/skill_taxonomy.json`` - категоризированная таксономия с
  человекочитаемым написанием навыков, по ней строится UI и кластеры.

Исторически файлы расходились: навык есть в whitelist, но без категории (или
наоборот), внутри одной категории встречается ``Java`` и ``java``, а один и тот
же навык лежит сразу в нескольких категориях. Модуль считает эти расхождения
одним проходом и отдаёт отчёт - без записи в файлы и без БД, чтобы его можно
было гонять в тестах и в CI.

Модуль намеренно не зависит от ``SkillTaxonomy`` (синглтон с кэшем) и от
``SkillNormalizer`` (тянет rapidfuzz): проверка должна быть дешёвой и
детерминированной.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Порог, выше которого расхождение считается «шумным» и требует ручной правки.
#: Сейчас используется только в CLI-отчёте, тесты опираются на явные срезы.
NOISE_THRESHOLD = 50

#: Базовый уровень расхождений, замеренный 2026-11-24 на Max_Opik_Dev.
#: Контракт: эти числа можно ТОЛЬКО уменьшать (по мере чистки данных).
#: Рост любого из них означает новый дрейф - его ловят тест и ``--strict``.
BASELINE_AMBIGUOUS_SKILLS = 140
BASELINE_DUPLICATE_CATEGORIES = 14
BASELINE_TAXONOMY_ONLY = 46
BASELINE_WHITELIST_ONLY = 156


def skill_key(skill: str) -> str:
    """Ключ сравнения навыков.

    Повторяет индексацию ``SkillTaxonomy._load`` (``skill.lower().strip()``),
    чтобы отчёт и рантайм видели одни и те же группы. Пробелы внутри не
    схлопываются: если таксономия различает ``"ci / cd"`` и ``"ci/cd"``, это
    должно быть видно в отчёте, а не замазано нормализацией.
    """
    return skill.lower().strip()


@dataclass(frozen=True)
class SkillConsistencyReport:
    """Снимок расхождений whitelist <-> таксономия."""

    categories: int
    taxonomy_entries: int
    taxonomy_unique: int
    whitelist_skills: int
    #: категория -> навыки, которые в ней повторяются после нормализации
    within_category_duplicates: Mapping[str, tuple[str, ...]]
    #: ключ навыка -> категории, в которых он встречается (только неоднозначные)
    ambiguous_skills: Mapping[str, tuple[str, ...]]
    #: пустые/нестроковые записи в таксономии (данные битые)
    invalid_entries: tuple[str, ...]
    #: есть в таксономии, но нет в whitelist
    taxonomy_only: tuple[str, ...]
    #: есть в whitelist, но нет в таксономии
    whitelist_only: tuple[str, ...]

    @property
    def duplicate_categories(self) -> tuple[str, ...]:
        """Категории, где есть повторы после нормализации."""
        return tuple(sorted(self.within_category_duplicates))

    @property
    def structural_problems(self) -> tuple[str, ...]:
        """Проблемы, которые ломают рантайм, а не просто «шумят».

        Повтор внутри категории искажает счётчики, а неоднозначный навык
        делает ``get_category`` зависимым от порядка ключей в JSON. Расхождение
        whitelist/таксономии - вопрос полноты данных, а не корректности.
        """
        problems: list[str] = []
        if self.invalid_entries:
            problems.append(f"invalid_entries={len(self.invalid_entries)}")
        if self.within_category_duplicates:
            problems.append(f"within_category_duplicates={len(self.within_category_duplicates)}")
        if self.ambiguous_skills:
            problems.append(f"ambiguous_skills={len(self.ambiguous_skills)}")
        return tuple(problems)

    @property
    def is_structurally_clean(self) -> bool:
        return not self.structural_problems


def exceeds_baseline(report: SkillConsistencyReport) -> tuple[str, ...]:
    """Расхождения, которые выросли относительно зафиксированного уровня.

    Пустой кортеж - дрейф не ухудшился (улучшение тоже даёт пустой кортеж).
    """
    regressions: list[str] = []
    if len(report.ambiguous_skills) > BASELINE_AMBIGUOUS_SKILLS:
        regressions.append(
            f"ambiguous_skills {len(report.ambiguous_skills)} > {BASELINE_AMBIGUOUS_SKILLS}"
        )
    if len(report.within_category_duplicates) > BASELINE_DUPLICATE_CATEGORIES:
        regressions.append(
            f"duplicate_categories {len(report.within_category_duplicates)} > {BASELINE_DUPLICATE_CATEGORIES}"
        )
    if len(report.taxonomy_only) > BASELINE_TAXONOMY_ONLY:
        regressions.append(f"taxonomy_only {len(report.taxonomy_only)} > {BASELINE_TAXONOMY_ONLY}")
    if len(report.whitelist_only) > BASELINE_WHITELIST_ONLY:
        regressions.append(f"whitelist_only {len(report.whitelist_only)} > {BASELINE_WHITELIST_ONLY}")
    return tuple(regressions)


def _iter_taxonomy_skills(taxonomy: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    """(категория, сырая запись) для всех навыков таксономии."""
    categories = taxonomy.get("categories", {})
    if not isinstance(categories, Mapping):
        return
    for cat_id, cat_data in categories.items():
        if not isinstance(cat_data, Mapping):
            continue
        for raw in cat_data.get("skills", []) or []:
            yield str(cat_id), raw


def analyze(
    taxonomy: Mapping[str, Any],
    whitelist: Iterable[str],
) -> SkillConsistencyReport:
    """Считает расхождения между таксономией и whitelist."""
    categories = taxonomy.get("categories", {}) or {}
    whitelist_keys = {skill_key(s) for s in whitelist if isinstance(s, str) and skill_key(s)}

    per_category: dict[str, list[str]] = defaultdict(list)
    skill_categories: dict[str, set[str]] = defaultdict(set)
    invalid: list[str] = []
    entries = 0

    for cat_id, raw in _iter_taxonomy_skills(taxonomy):
        if not isinstance(raw, str) or not raw.strip():
            invalid.append(f"{cat_id}:{raw!r}")
            continue
        entries += 1
        key = skill_key(raw)
        per_category[cat_id].append(key)
        skill_categories[key].add(cat_id)

    within: dict[str, tuple[str, ...]] = {}
    for cat_id, keys in per_category.items():
        seen: set[str] = set()
        dups: list[str] = []
        for key in keys:
            if key in seen and key not in dups:
                dups.append(key)
            seen.add(key)
        if dups:
            within[cat_id] = tuple(sorted(dups))

    ambiguous = {
        key: tuple(sorted(cats))
        for key, cats in skill_categories.items()
        if len(cats) > 1
    }
    taxonomy_keys = set(skill_categories)

    return SkillConsistencyReport(
        categories=len(categories) if isinstance(categories, Mapping) else 0,
        taxonomy_entries=entries,
        taxonomy_unique=len(taxonomy_keys),
        whitelist_skills=len(whitelist_keys),
        within_category_duplicates=within,
        ambiguous_skills=ambiguous,
        invalid_entries=tuple(invalid),
        taxonomy_only=tuple(sorted(taxonomy_keys - whitelist_keys)),
        whitelist_only=tuple(sorted(whitelist_keys - taxonomy_keys)),
    )


def load_taxonomy(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"taxonomy root must be an object, got {type(data).__name__}")
    return data


def load_whitelist(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"whitelist root must be an array, got {type(data).__name__}")
    return [item for item in data if isinstance(item, str)]


def format_report(report: SkillConsistencyReport, *, limit: int = 10) -> str:
    """Человекочитаемый отчёт для CLI и логов."""
    lines = [
        f"categories: {report.categories}",
        f"taxonomy entries: {report.taxonomy_entries} (unique {report.taxonomy_unique})",
        f"whitelist skills: {report.whitelist_skills}",
        f"within-category duplicates: {len(report.within_category_duplicates)}"
        f" in {len(report.duplicate_categories)} categories",
        f"ambiguous skills (2+ categories): {len(report.ambiguous_skills)}",
        f"taxonomy only (no whitelist): {len(report.taxonomy_only)}",
        f"whitelist only (no category): {len(report.whitelist_only)}",
        f"invalid entries: {len(report.invalid_entries)}",
    ]
    if report.structural_problems:
        lines.append("structural problems: " + ", ".join(report.structural_problems))
    else:
        lines.append("structural problems: none")

    if report.whitelist_only:
        lines.append("")
        lines.append(f"whitelist without category (first {limit}):")
        lines.extend(f"  - {name}" for name in report.whitelist_only[:limit])
    if report.taxonomy_only:
        lines.append("")
        lines.append(f"taxonomy without whitelist (first {limit}):")
        lines.extend(f"  - {name}" for name in report.taxonomy_only[:limit])
    if report.ambiguous_skills:
        lines.append("")
        lines.append(f"ambiguous skills (first {limit}):")
        for key in sorted(report.ambiguous_skills)[:limit]:
            lines.append(f"  - {key}: {', '.join(report.ambiguous_skills[key])}")
    return "\n".join(lines)


__all__ = [
    "BASELINE_AMBIGUOUS_SKILLS",
    "BASELINE_DUPLICATE_CATEGORIES",
    "BASELINE_TAXONOMY_ONLY",
    "BASELINE_WHITELIST_ONLY",
    "NOISE_THRESHOLD",
    "SkillConsistencyReport",
    "analyze",
    "exceeds_baseline",
    "format_report",
    "load_taxonomy",
    "load_whitelist",
    "skill_key",
]
