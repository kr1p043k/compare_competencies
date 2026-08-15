"""Расширение whitelist (it_skills.json) рыночными навыками из вакансий hh.ru.

Сканирует key_skills вакансий, фильтрует шум, категоризует через таксономию
(эмбеддинг-классификатор) и дописывает новые навыки в it_skills.json
и skill_taxonomy.json. Берём только уверенные IT-категории (не "other").

Usage:
    python -m src.cli ingest-market-skills --dry-run        # показать кандидатов
    python -m src.cli ingest-market-skills                  # применить
    python -m src.cli ingest-market-skills --min-freq 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import structlog

from src import config
from src.parsing.skills.skill_normalizer import SkillNormalizer

logger = structlog.get_logger(__name__)

IT_SKILLS_PATH = config.IT_SKILLS_PATH
TAXONOMY_PATH = config.SKILL_TAXONOMY_PATH
VACANCIES_PATH = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
DEFAULT_MIN_FREQ = 3
DEFAULT_THRESHOLD = 0.55


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_market_skills(min_freq: int) -> Counter:
    """Частоты нормализованных key_skills по всем вакансиям."""
    if not VACANCIES_PATH.exists():
        print(f"Файл вакансий не найден: {VACANCIES_PATH}")
        return Counter()
    raw = _load_json(VACANCIES_PATH)
    vacs = raw if isinstance(raw, list) else raw.get("items", [])
    freq: Counter = Counter()
    for v in vacs:
        for s in v.get("key_skills") or []:
            name = s.get("name", "") if isinstance(s, dict) else str(s)
            if not name:
                continue
            norm = SkillNormalizer.normalize(name)
            if norm.is_ok() and norm.unwrap():
                freq[norm.unwrap()] += 1
    # Убираем редкие
    return Counter({k: c for k, c in freq.items() if c >= min_freq})


def _is_clean_it_skill(skill: str) -> bool:
    """Пропускает навык через валидатор (blacklist/generic/длина)."""
    import re
    from src.parsing.skills.skill_validator import SkillValidator

    # Артефакты нормализации (напр. "1С:ERP" -> "сerp") — пропускаем
    if re.fullmatch(r"[1сС]\s*[:]?erp", skill, re.IGNORECASE):
        return False
    validator = SkillValidator(whitelist=None)
    result = validator.validate(skill)
    if result.is_err():
        return False
    return result.unwrap().is_valid


def apply_to_files(candidates: dict[str, str], dry_run: bool) -> tuple[int, int]:
    """Дописывает кандидатов в it_skills.json и skill_taxonomy.json."""
    it_skills = set(_load_json(IT_SKILLS_PATH))
    taxonomy = _load_json(TAXONOMY_PATH)
    cats = taxonomy.get("categories", {})

    new_it = 0
    new_tax = 0
    for skill, cat_id in sorted(candidates.items()):
        if skill not in it_skills:
            if not dry_run:
                it_skills.add(skill)
            new_it += 1
        if cat_id in cats:
            lst = cats[cat_id].setdefault("skills", [])
            existing = {s.strip().lower() for s in lst}
            if skill not in existing:
                if not dry_run:
                    lst.append(skill)
                new_tax += 1

    if not dry_run:
        IT_SKILLS_PATH.write_text(
            json.dumps(sorted(it_skills), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        TAXONOMY_PATH.write_text(
            json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return new_it, new_tax


def main(args: argparse.Namespace) -> None:
    dry_run = getattr(args, "dry_run", False)
    min_freq = getattr(args, "min_freq", DEFAULT_MIN_FREQ)

    freq = collect_market_skills(min_freq)
    print(f"Уникальных key_skills (freq>={min_freq}): {len(freq)}")

    # Уже известные
    it_skills = set(_load_json(IT_SKILLS_PATH))
    unknown = {k: c for k, c in freq.items() if k not in it_skills}
    print(f"Не в it_skills: {len(unknown)}")

    if not unknown:
        print("Новых навыков нет.")
        return

    # Фильтр: только чистые IT-навыки (blacklist/generic отсекает soft)
    unknown = {k: c for k, c in unknown.items() if _is_clean_it_skill(k)}
    print(f"После soft-фильтра (blacklist/generic): {len(unknown)}")

    if not unknown:
        print("Чистых IT-кандидатов нет.")
        return

    # Категоризация: берём только уверенные IT-категории
    from src.cli.taxonomy_audit import categorize_new_skills_with_score

    threshold = getattr(args, "threshold", 0.55)
    scored = categorize_new_skills_with_score(sorted(unknown.keys()))
    candidates = {}
    skipped_other = []
    for skill in sorted(unknown.keys()):
        cat_id, score = scored.get(skill, ("other", 0.0))
        if cat_id == "other" or cat_id == "abstract_concepts" or score < threshold:
            skipped_other.append((skill, unknown[skill]))
            continue
        candidates[skill] = cat_id

    print(f"\nКандидатов для добавления: {len(candidates)}")
    print(f"Отсеяно (не-IT/неоднозначно): {len(skipped_other)}")
    print("\nТоп-кандидаты (по частоте):")
    for skill, cat_id in sorted(candidates.items(), key=lambda x: -unknown[x[0]])[:40]:
        print(f"  {skill:45s} → {cat_id}  (x{unknown[skill]})")

    if dry_run:
        print(f"\n[dry-run] Будет добавлено: {len(candidates)} в it_skills и taxonomy.")
        return

    new_it, new_tax = apply_to_files(candidates, dry_run=False)
    print(f"\nДобавлено в it_skills: {new_it}")
    print(f"Добавлено в skill_taxonomy: {new_tax}")

    # Итоговое покрытие
    final = len(it_skills | set(candidates))
    print(f"Итоговый it_skills: {final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Расширение whitelist рыночными навыками")
    parser.add_argument("--dry-run", action="store_true", help="Только показать кандидатов")
    parser.add_argument("--min-freq", type=int, default=DEFAULT_MIN_FREQ)
    main(parser.parse_args())
