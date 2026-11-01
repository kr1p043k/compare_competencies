"""Разделение дисциплин направления по годам набора.

Одна и та же дисциплина встречается в нескольких учебных планах, а часть
курсов существует только в одном. Скрипт собирает официальные перечни
(`official_curriculum_<код>_<год>.json`), сопоставляет названия между годами
с той же нормализацией, что и аудит (аббревиатуры ИИ/ПО/МО/ИС, скобочные
латинско-русские дубли), и пишет карту «дисциплина → годы».

    python scripts/curriculum_years.py --direction 09.03.02
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "data" / "reference"


def _load_audit():
    """Импортировать хелперы нормализации из аудита, не дублируя их."""
    spec = importlib.util.spec_from_file_location(
        "audit_curriculum_coverage", ROOT / "scripts" / "audit_curriculum_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_files(direction: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in REFERENCE.glob(f"official_curriculum_{direction}_*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        year = doc.get("year")
        if isinstance(year, int):
            out[year] = path
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="09.03.02")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    audit = _load_audit()
    files = official_files(args.direction)
    if not files:
        raise SystemExit(f"нет перечней official_curriculum_{args.direction}_<год>.json")

    years = sorted(files)
    print(f"найдены перечни: {', '.join(str(y) for y in years)}")

    # Год -> {нормализованное имя: оригинальное}
    per_year: dict[int, dict[str, str]] = {}
    for year in years:
        doc = json.loads(files[year].read_text(encoding="utf-8"))
        per_year[year] = {audit.norm(d): d for d in doc["disciplines"]}

    # Канонические названия: берём самое длинное из встречающихся вариантов.
    all_keys: set[str] = set()
    for mapping in per_year.values():
        all_keys.update(mapping)

    cards: dict[str, dict] = {}
    for key in all_keys:
        present = [y for y in years if key in per_year[y]]
        names = {per_year[y][key] for y in present}
        canonical = max(names, key=len)
        cards[canonical] = {
            "years": present,
            "name_by_year": {str(y): per_year[y][key] for y in present},
            "only_in": present[0] if len(present) == 1 else None,
        }

    only_2024 = sorted(c for c, v in cards.items() if v["years"] == [2024])
    only_2025 = sorted(c for c, v in cards.items() if v["years"] == [2025])
    both = sorted(c for c, v in cards.items() if len(v["years"]) > 1)

    print(f"\nвсего уникальных дисциплин: {len(cards)}")
    print(f"  только 2024: {len(only_2024)}")
    print(f"  только 2025: {len(only_2025)}")
    print(f"  в обоих годах: {len(both)}")
    for c in only_2024:
        print(f"    · только 2024: {c}")
    for c in only_2025:
        print(f"    · только 2025: {c}")

    payload = {
        "direction": args.direction,
        "years": years,
        "counts": {
            "total": len(cards),
            "both": len(both),
            "only_2024": len(only_2024),
            "only_2025": len(only_2025),
        },
        "disciplines": dict(sorted(cards.items())),
    }
    out = Path(args.out) if args.out else REFERENCE / f"curriculum_years_{args.direction}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nкарта: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
