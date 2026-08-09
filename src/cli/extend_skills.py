"""Auto-extend it_skills.json from new vacancies.

Usage:
    python -m src.cli extend-skills [--interactive] [--yes]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import structlog

from src import Err, Ok
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.parsing.skills.skill_validator import SkillValidator
from src.parsing.skills.vacancy_parser import VacancyParser
from src.parsing.utils import read_json

logger = structlog.get_logger(__name__)

SKILLS_PATH = Path(__file__).parent.parent.parent / "data" / "reference" / "it_skills.json"
VACANCIES_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def load_existing_skills() -> set[str]:
    with open(SKILLS_PATH, "r", encoding="utf-8") as f:
        return {s.strip().lower() for s in json.load(f) if s.strip()}


def save_skills(skills: list[str]) -> None:
    existing = load_existing_skills()
    merged = sorted(existing | {s.lower() for s in skills})
    with open(SKILLS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(merged)} skills to {SKILLS_PATH}")


def extract_all_skills(
    parser: VacancyParser,
    validator: SkillValidator,
    vacancies: list,
    min_frequency: int = 1,
) -> list[tuple[str, int]]:
    """Нормализованные навыки из key_skills + текста вакансий с частотой ≥ min_frequency."""
    skill_counter: Counter[str] = Counter()

    for vac in vacancies:
        if not isinstance(vac, dict):
            continue
        skills_from_vac: set[str] = set()

        for s in vac.get("key_skills", []) or []:
            name = s.get("name", "") if isinstance(s, dict) else str(s)
            if not name:
                continue
            match SkillNormalizer.normalize(name):
                case Ok(norm) if norm:
                    skills_from_vac.add(norm)
                case _:
                    pass

        desc = vac.get("description", "") or ""
        snip = vac.get("snippet", {}) or {}
        req = snip.get("requirement", "") or ""
        resp = snip.get("responsibility", "") or ""
        match parser.extract_skills_from_description(f"{desc} {req} {resp}"):
            case Ok(text_skills):
                for skill in text_skills:
                    match SkillNormalizer.normalize(skill):
                        case Ok(norm) if norm:
                            skills_from_vac.add(norm)
                        case _:
                            pass
            case _:
                pass

        for skill in skills_from_vac:
            skill_counter[skill] += 1

    valid = []
    for skill, freq in skill_counter.items():
        if freq < min_frequency:
            continue
        match validator.validate(skill):
            case Ok(result) if result.is_valid:
                valid.append((skill, freq))
            case _:
                pass
    return sorted(valid, key=lambda x: x[1], reverse=True)


def main(args: argparse.Namespace) -> None:
    existing = load_existing_skills()
    print(f"Existing skills: {len(existing)}")

    parser = VacancyParser()
    validator = SkillValidator(whitelist=None)
    min_frequency = getattr(args, "min_frequency", 1)
    all_new: Counter[str] = Counter()

    for path in sorted(VACANCIES_DIR.glob("hh_vacancies*.json")):
        raw = read_json(path)
        match raw:
            case Ok(data):
                pass
            case Err(e):
                print(f"skip {path.name}: {e}")
                continue
        if not data:
            continue
        vacancies = data if isinstance(data, list) else data.get("items", [])
        for skill, freq in extract_all_skills(parser, validator, vacancies, min_frequency):
            if skill not in existing:
                all_new[skill] += freq
        print(f"processed {path.name}: {len(vacancies)} vacancies")

    if not all_new:
        print("No new skills found")
        return

    print(f"\nNew skills found: {len(all_new)}")
    for skill, freq in all_new.most_common(20):
        print(f"  {skill:40s} × {freq}")

    if args.yes:
        save_skills([s for s, _ in all_new.items() if s])
        print("All new skills added")
    elif args.interactive:
        to_add: list[str] = []
        for skill, _ in all_new.most_common():
            answer = input(f"Add '{skill}'? [Y/n/q] ").strip().lower()
            if answer == "q":
                break
            if answer in ("", "y", "yes"):
                to_add.append(skill)
        if to_add:
            save_skills(to_add)
            print(f"Added {len(to_add)} skills")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extend it_skills taxonomy")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--dead", action="store_true")
    parser.add_argument("--min-frequency", type=int, default=1)
    main(parser.parse_args())
