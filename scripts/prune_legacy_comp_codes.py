"""Удаление устаревших кодов компетенций из KRM.

В KRM могут остаться коды, которых не существует в предметной области:
например `DL-3` у дисциплины «Алгоритмы, технологии и инструментальные
средства обработки изображений и видеопоследовательностей». Категория `DL`
не входит в COMPETENCY_CATEGORIES, БД такие строки отвергает ограничением
`ck_comp_category`, сид их пропускает — но `src/analyzers/academic_gap.py::
_load_krm` читает файл KRM напрямую, поэтому мусорные коды и их навыки
попадают в анализ разрыва и завышают покрытие.

Скрипт ничего не удаляет, если код встречается в аннотациях года: это
признак того, что неверно сопоставлена категория, а не данные.

Использование:
    python scripts/prune_legacy_comp_codes.py --direction 09.03.02 --year 2025
    python scripts/prune_legacy_comp_codes.py --direction 09.03.02 --year 2025 --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.models.krm_models import COMPETENCY_CATEGORIES  # noqa: E402

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
_CODE_RE = re.compile(r"^([A-Za-zА-Яа-яЁё]+)[\s-]*(\d+(?:\.\d+)*)$")


def category_of(code: str) -> str:
    """Категория кода: буквенный префикс (DL-3 -> DL, ВПК-5 -> ВПК)."""
    m = _CODE_RE.match((code or "").strip())
    return m.group(1).upper() if m else (code or "").strip()


def load_annotation_codes(direction: str, year: int) -> set[str]:
    """Коды, реально встречающиеся в аннотациях года (страховка)."""
    path = REFERENCE_DIR / "sfu_annotations" / f"{direction}_{year}.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    codes: set[str] = set()
    items = data if isinstance(data, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        for c in item.get("competencies") or []:
            codes.add(str(c).strip())
    return codes


def prune(direction: str, year: int, apply: bool) -> int:
    krm_path = REFERENCE_DIR / f"krm_disciplines_{direction}.json"
    data = json.loads(krm_path.read_text(encoding="utf-8"))
    sub = next(iter(data.values()))
    discs = sub.get("disciplines", {})
    allowed = set(COMPETENCY_CATEGORIES)
    in_annotations = load_annotation_codes(direction, year)

    total_removed = 0
    print(f"разрешённые категории: {sorted(allowed)}")
    if not in_annotations:
        print("ВНИМАНИЕ: аннотации года не найдены, страховка по источникам отключена")
    print()

    for name, body in discs.items():
        comps = list(body.get("competencies") or [])
        bad = [c for c in comps if category_of(c) not in allowed]
        if not bad:
            continue

        removable, guarded = [], []
        for code in bad:
            (guarded if code in in_annotations else removable).append(code)

        if guarded:
            print(f"  {name}")
            print(f"    СОХРАНёНЫ (есть в аннотациях, проверьте маппинг): {guarded}")

        keep = [c for c in comps if c not in removable]
        if not keep:
            print(f"  {name}")
            print("    ПРОПУЩЕНО: удаление оставило бы дисциплину без компетенций")
            continue

        skills = body.get("skills") or {}
        ksa = body.get("ksa") or {}
        n_skills = sum(len(skills.get(c) or []) for c in removable)
        n_ksa = sum(
            len(v) for c in removable
            for v in (ksa.get(c) or {}).values() if isinstance(ksa.get(c), dict)
        )

        print(f"  {name}")
        for code in removable:
            print(f"    - {code}: skills={len(skills.get(code) or [])}")
        print(f"    итого: кодов={len(removable)} skills={n_skills} ksa={n_ksa}")

        if apply:
            body["competencies"] = keep
            for code in removable:
                skills.pop(code, None)
                ksa.pop(code, None)
            total_removed += len(removable)

    if not apply:
        print("\n(dry-run; добавьте --apply)")
        return 0

    krm_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nудалено кодов: {total_removed}; дисциплин в KRM: {len(discs)}")
    return total_removed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", default="09.03.02")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    prune(args.direction, args.year, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
