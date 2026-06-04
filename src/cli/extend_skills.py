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

from src.parsing.skills.vacancy_parser import VacancyParser
from src.parsing.utils import read_json

logger = structlog.get_logger(__name__)

SKILLS_PATH = Path(__file__).parent.parent.parent / "data" / "reference" / "it_skills.json"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_FILE = CACHE_DIR / "vacancy_skill_cache.json"
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


def main(args: argparse.Namespace) -> None:
    existing = load_existing_skills()
    print(f"Existing skills: {len(existing)}")

    parser = VacancyParser()
    all_new: Counter[str] = Counter()

    for path in sorted(VACANCIES_DIR.glob("hh_vacancies*.json")):
        raw = read_json(path)
        if not raw:
            continue
        for vac in (raw if isinstance(raw, list) else raw.get("items", [])):
            skills = parser.parse_skills(vac)
            for s in skills:
                s = s.lower()
                if s not in existing:
                    all_new[s] += 1

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
    parser.add_argument("--min-frequency", type=int, default=2)
    main(parser.parse_args())
