#!/usr/bin/env python
"""
Скрипт авторасширения it_skills.json из новых вакансий.

РЕЖИМЫ ЗАПУСКА:
  # Только анализ (без изменений) — по умолчанию
  python scripts/extend_it_skills.py

  # Интерактивный режим: анализ + выбор навыков для добавления
  python scripts/extend_it_skills.py --interactive

  # Автоматически добавить все новые навыки
  python scripts/extend_it_skills.py --yes

  # Интерактивный + покрытие + мёртвые навыки
  python scripts/extend_it_skills.py --interactive --coverage --dead

  # Только навыки с частотой ≥ 3
  python scripts/extend_it_skills.py --interactive --min-frequency 3
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.analyzers.skill_taxonomy import SkillTaxonomy
from src.parsing.skill_normalizer import SkillNormalizer
from src.parsing.skill_validator import SkillValidator
from src.parsing.utils import load_it_skills, read_json
from src.parsing.vacancy_parser import VacancyParser

logger = structlog.get_logger(__name__)


# =====================================================================
# ИЗВЛЕЧЕНИЕ НАВЫКОВ
# =====================================================================


def extract_all_skills(vacancies: list[dict], min_frequency: int = 1) -> list[tuple[str, int]]:
    """
    Извлекает нормализованные навыки из вакансий с подсчётом частот.
    Использует SkillNormalizer и SkillValidator (без whitelist).
    """
    parser = VacancyParser()
    validator = SkillValidator(whitelist=None)

    skill_counter = Counter()

    for vac in vacancies:
        skills_from_vac = set()

        # key_skills (если есть)
        for s in vac.get("key_skills", []):
            name = s.get("name", "") if isinstance(s, dict) else str(s)
            if name:
                norm = SkillNormalizer.normalize(name)
                if norm:
                    skills_from_vac.add(norm)

        # Текстовые поля
        desc = vac.get("description", "") or ""
        snippet = vac.get("snippet", {}) or {}
        req = snippet.get("requirement", "") or ""
        resp = snippet.get("responsibility", "") or ""
        text_skills = parser.extract_skills_from_description(f"{desc} {req} {resp}")
        for skill in text_skills:
            norm = SkillNormalizer.normalize(skill)
            if norm:
                skills_from_vac.add(norm)

        for skill in skills_from_vac:
            skill_counter[skill] += 1

    # Фильтр по частоте
    filtered = [(s, c) for s, c in skill_counter.items() if c >= min_frequency]

    # Валидация
    valid = []
    for skill, count in filtered:
        if validator.validate(skill).is_valid:
            valid.append((skill, count))

    logger.info(
        "skills_extracted",
        unique=len(skill_counter),
        filtered=len(filtered),
        min_frequency=min_frequency,
        valid=len(valid),
    )
    return sorted(valid, key=lambda x: x[1], reverse=True)


# =====================================================================
# АНАЛИЗ
# =====================================================================


def analyze_coverage(current_skills: set[str], taxonomy: SkillTaxonomy) -> dict[str, dict]:
    """Покрытие категорий таксономии текущим белым списком."""
    coverage = {}
    for cat_id in taxonomy.get_all_categories():
        cat_skills = set(s.lower() for s in taxonomy.get_skills_in_category(cat_id))
        covered = cat_skills & current_skills
        coverage[cat_id] = {
            "label": taxonomy.get_category_label_by_id(cat_id),
            "icon": taxonomy.get_category_icon_by_id(cat_id),
            "total": len(cat_skills),
            "covered": len(covered),
            "percent": round(len(covered) / len(cat_skills) * 100, 1) if cat_skills else 0,
        }
    return coverage


def find_dead_skills(current_skills: set[str], extracted_skills: dict[str, int]) -> list[str]:
    """Навыки из белого списка, не встретившиеся в вакансиях."""
    extracted_lower = {s.lower() for s in extracted_skills}
    return sorted(s for s in current_skills if s.lower() not in extracted_lower)


def print_coverage(coverage: dict[str, dict]):
    """Красивый вывод покрытия категорий."""
    print("\n" + "=" * 70)
    print("📊 ПОКРЫТИЕ КАТЕГОРИЙ ТАКСОНОМИИ")
    print("=" * 70)
    for _cat_id, info in sorted(coverage.items(), key=lambda x: x[1]["percent"]):
        bar = _make_bar(info["percent"])
        print(f"  {info['icon']} {info['label']:<30} {bar} {info['percent']:.1f}% ({info['covered']}/{info['total']})")


def print_new_skills(new_skills: dict[str, int], taxonomy=None):
    """Вывод найденных новых навыков."""
    print("\n" + "=" * 70)
    print(f"🔍 НОВЫЕ НАВЫКИ (отсутствуют в it_skills.json): {len(new_skills)}")
    print("=" * 70)
    for skill, freq in sorted(new_skills.items(), key=lambda x: x[1], reverse=True):
        cat_info = ""
        if taxonomy:
            cat = taxonomy.get_category(skill)
            if cat != "other":
                cat_info = f" [{taxonomy.get_category_icon(skill)} {taxonomy.get_category_label(skill)}]"
        print(f"  {skill:<45} частота: {freq}{cat_info}")
    if not new_skills:
        print("  ✅ Новых навыков не найдено — белый список актуален")


def print_dead_skills(dead_skills: list[str]):
    """Вывод навыков, не встретившихся в вакансиях."""
    print("\n" + "=" * 70)
    print(f"⚠️  НАВЫКИ БЕЗ УПОМИНАНИЙ В ВАКАНСИЯХ: {len(dead_skills)}")
    print("=" * 70)
    print("  (присутствуют в it_skills.json, но не найдены в текущем сборе)")
    for skill in dead_skills[:30]:
        print(f"  • {skill}")
    if len(dead_skills) > 30:
        print(f"  ... и ещё {len(dead_skills) - 30}")
    if not dead_skills:
        print("  ✅ Все навыки из белого списка встретились в вакансиях")


# =====================================================================
# ДОБАВЛЕНИЕ
# =====================================================================


def interactive_confirm(new_skills: dict[str, int]) -> set[str]:
    """Интерактивный выбор навыков для добавления."""
    print("\n" + "=" * 70)
    print("ВЫБОР НАВЫКОВ ДЛЯ ДОБАВЛЕНИЯ")
    print("=" * 70)
    print("  y = добавить этот навык")
    print("  n = пропустить")
    print("  a = добавить все оставшиеся")
    print("  q = закончить (оставить остальные)\n")

    approved = set()
    for skill, freq in sorted(new_skills.items(), key=lambda x: x[1], reverse=True):
        ans = input(f"  [{freq:>3}] {skill:<45} ? [y/n/a/q]: ").strip().lower()
        if ans == "q":
            break
        elif ans == "a":
            approved.update(new_skills.keys())
            break
        elif ans == "y":
            approved.add(skill)
    return approved


def add_skills_to_whitelist(skills_to_add: set[str], output_path: Path, backup: bool = True) -> int:
    """Добавляет навыки в it_skills.json. Возвращает количество добавленных."""
    current = load_it_skills()
    if not current:
        logger.error("failed_to_load_whitelist")
        return 0

    if not skills_to_add:
        logger.info("no_skills_selected")
        return 0

    if backup and output_path.exists():
        import shutil

        backup_path = output_path.with_suffix(".backup.json")
        shutil.copy(output_path, backup_path)
        logger.info("backup_created", path=str(backup_path))

    updated = sorted(current | skills_to_add)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    logger.info(
        "whitelist_updated",
        path=str(output_path),
        before=len(current),
        added=len(skills_to_add),
        after=len(updated),
    )
    return len(skills_to_add)


# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ
# =====================================================================


def _make_bar(percent: float, width: int = 20) -> str:
    filled = int(width * percent / 100)
    return "█" * filled + "░" * (width - filled)


def _print_available_files():
    logger.info("available_vacancy_files")
    for p in config.DATA_RAW_DIR.glob("*.json"):
        logger.info("raw_file", name=f"data/raw/{p.name}")
    for p in config.DATA_RESULT_DIR.glob("*.json"):
        logger.info("result_file", name=f"data/result/{p.name}")


# =====================================================================
# MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Анализ и расширение it_skills.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Режимы запуска:
  python scripts/extend_it_skills.py                         # только анализ
  python scripts/extend_it_skills.py --interactive           # анализ + выбор навыков
  python scripts/extend_it_skills.py --yes                   # анализ + добавить всё
  python scripts/extend_it_skills.py --interactive --coverage --dead  # всё вместе
        """,
    )
    parser.add_argument(
        "--vacancies",
        "-v",
        type=Path,
        default=config.DATA_RESULT_DIR / "hh_vacancies_detailed.json",
        help="JSON с вакансиями",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=config.IT_SKILLS_PATH,
        help="Выходной файл (по умолчанию data/it_skills.json)",
    )
    parser.add_argument("--min-frequency", "-f", type=int, default=1, help="Минимальная частота навыка (default: 1)")

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Интерактивный режим: анализ + выбор навыков для добавления"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Добавить все новые навыки автоматически (без подтверждения)"
    )

    parser.add_argument("--coverage", "-c", action="store_true", help="Показать покрытие категорий таксономии")
    parser.add_argument(
        "--dead", "-d", action="store_true", help="Показать навыки из белого списка без упоминаний в вакансиях"
    )
    parser.add_argument("--no-backup", action="store_true", help="Не создавать резервную копию перед изменением")

    args = parser.parse_args()

    if args.interactive and args.yes:
        logger.error("conflicting_flags_interactive_and_yes")
        sys.exit(1)

    if not args.vacancies.exists():
        logger.error("vacancy_file_not_found", path=str(args.vacancies))
        _print_available_files()
        sys.exit(1)

    try:
        taxonomy = SkillTaxonomy()
    except Exception:
        taxonomy = None

    current_skills = load_it_skills()
    if not current_skills:
        logger.error("failed_to_load_it_skills")
        sys.exit(1)

    vacancies = read_json(args.vacancies)
    if not vacancies:
        logger.error("failed_to_load_vacancies")
        sys.exit(1)

    logger.info("starting_analysis", whitelist=len(current_skills), vacancies=len(vacancies))

    extracted = extract_all_skills(vacancies, min_frequency=args.min_frequency)
    extracted_dict = {skill: freq for skill, freq in extracted}

    new_skills = {}
    for skill, freq in extracted_dict.items():
        if skill.lower() not in current_skills:
            new_skills[skill] = freq
    print_new_skills(new_skills, taxonomy)

    if args.coverage and taxonomy:
        coverage = analyze_coverage(current_skills, taxonomy)
        print_coverage(coverage)

    if args.dead:
        dead = find_dead_skills(current_skills, extracted_dict)
        print_dead_skills(dead)

    if not new_skills:
        print("\n✅ Белый список актуален — новых навыков нет.")
        return

    if args.yes:
        added = add_skills_to_whitelist(
            skills_to_add=set(new_skills.keys()), output_path=args.output, backup=not args.no_backup
        )
        if added > 0:
            print(f"\n✅ Добавлено {added} навыков.")
            print("⚠️  Очистите кэш перед следующим запуском:")
            print("   rm data/processed/parsed_skills.pkl")
            print("   rm -r data/embeddings/cache/")

    elif args.interactive:
        selected = interactive_confirm(new_skills)
        if selected:
            added = add_skills_to_whitelist(skills_to_add=selected, output_path=args.output, backup=not args.no_backup)
            if added > 0:
                print(f"\n✅ Добавлено {added} навыков.")
                print("⚠️  Очистите кэш перед следующим запуском:")
                print("   rm data/processed/parsed_skills.pkl")
                print("   rm -r data/embeddings/cache/")
        else:
            print("\nНичего не добавлено.")

    else:
        print("\n" + "-" * 70)
        print("📋 РЕЖИМ АНАЛИЗА — изменения не вносятся.")
        print("-" * 70)
        print("Чтобы добавить навыки:")
        print("  python scripts/extend_it_skills.py --interactive   # выбрать вручную")
        print("  python scripts/extend_it_skills.py --yes            # добавить все")


if __name__ == "__main__":
    main()
