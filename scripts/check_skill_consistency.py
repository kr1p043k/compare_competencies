"""Отчёт о расхождениях whitelist навыков и таксономии (6.4.4).

Читает ``data/reference/it_skills.json`` и ``data/reference/skill_taxonomy.json``,
считает расхождения через ``src.analyzers.skills.skill_consistency`` и печатает
сводку. Ничего не пишет в файлы и не трогает БД.

Usage:
    python scripts/check_skill_consistency.py
    python scripts/check_skill_consistency.py --json          # машинный отчёт
    python scripts/check_skill_consistency.py --strict         # exit 1 при структурных проблемах
    python scripts/check_skill_consistency.py --limit 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analyzers.skills.skill_consistency import (  # noqa: E402
    analyze,
    exceeds_baseline,
    format_report,
    load_taxonomy,
    load_whitelist,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка консистентности навыков")
    parser.add_argument("--whitelist", default=str(ROOT / "data" / "reference" / "it_skills.json"))
    parser.add_argument("--taxonomy", default=str(ROOT / "data" / "reference" / "skill_taxonomy.json"))
    parser.add_argument("--json", action="store_true", help="печатать JSON вместо текста")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1, если расхождения выросли относительно зафиксированного базового уровня",
    )
    parser.add_argument("--limit", type=int, default=10, help="сколько примеров показывать")
    args = parser.parse_args(argv)

    report = analyze(load_taxonomy(args.taxonomy), load_whitelist(args.whitelist))
    regressions = exceeds_baseline(report)

    if args.json:
        payload = {
            "categories": report.categories,
            "taxonomy_entries": report.taxonomy_entries,
            "taxonomy_unique": report.taxonomy_unique,
            "whitelist_skills": report.whitelist_skills,
            "within_category_duplicates": {
                cat: list(names) for cat, names in report.within_category_duplicates.items()
            },
            "ambiguous_skills": {key: list(cats) for key, cats in report.ambiguous_skills.items()},
            "invalid_entries": list(report.invalid_entries),
            "taxonomy_only": list(report.taxonomy_only),
            "whitelist_only": list(report.whitelist_only),
            "structural_problems": list(report.structural_problems),
            "regressions": list(regressions),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_report(report, limit=args.limit))

    if args.strict and regressions:
        print("\nFAIL: skill drift grew: " + "; ".join(regressions), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
