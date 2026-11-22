"""Убрать мусор из KSA-текстов KRM-файлов.

Мусор появился из слияния аннотаций: темы лекций попадали в `knowledge`
вместе с буллетами («• Лабораторная работа: …»), а шапка таблицы компетенций
(«Компетенция Индикаторы достижения компетенции Результаты обучения»)
копировалась на каждую страницу и размножалась по всем компетенциям.

Что чистится:
  - ведущий буллет/тире любого вида;
  - пункты-шапки таблицы;
  - пустые и слишком короткие (< 12 символов) обрывки;
  - точные дубли внутри одной компетенции.

Что НЕ трогается: состав компетенций, `skills`-карта, порядок дисциплин.

    python scripts/clean_krm_ksa.py --dry-run
    python scripts/clean_krm_ksa.py --only 09.03.02
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "data" / "reference"

BULLET = re.compile(r"^[\u2012\u2013\u2014\u2212\-\u2022\u2023\u25E6\u2043\u2219•‣⁃]\s*")
TABLE_HEADER = re.compile(
    r"^(?:компетенци[яи]\s+индикаторы|индикаторы\s+достижения|результаты\s+обучения)", re.I
)
#: Структурные заголовки аннотации - не знания и не умения.
STRUCTURAL = re.compile(
    r"^(?:содержание\s+(?:лекционных|практических|лабораторных)"
    r"|модуль\s+\d+\.?\s*$|forms?\s+of\s+control|course\s+contents?)", re.I
)
MIN_LEN = 12


def clean_text(text: str) -> str:
    t = (text or "").strip()
    # Буллетов может быть несколько подряд («– – 36 вопросы…»).
    while BULLET.match(t):
        t = BULLET.sub("", t).strip()
    t = re.sub(r"\s+", " ", t)
    # Хвостовая пунктуация: у записей из разбора РПД её нет (clean_skill
    # срезает), у аннотационных была - приводим к одному виду.
    t = t.rstrip(" ;.")
    return t.strip()


def is_junk(text: str) -> bool:
    return (not text or len(text) < MIN_LEN
            or bool(TABLE_HEADER.match(text)) or bool(STRUCTURAL.match(text)))


def clean_file(path: Path, dry_run: bool) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    removed = deduped = 0
    for sub in data.values():
        if not isinstance(sub, dict):
            continue
        for body in (sub.get("disciplines") or {}).values():
            ksa = body.get("ksa") or {}
            for code, sections in ksa.items():
                if not isinstance(sections, dict):
                    continue
                for kt in ("knowledge", "abilities", "skills"):
                    items = sections.get(kt)
                    if not isinstance(items, list):
                        continue
                    seen: set[str] = set()
                    out: list[str] = []
                    for raw in items:
                        text = clean_text(str(raw))
                        if is_junk(text):
                            removed += 1
                            continue
                        if text in seen:
                            deduped += 1
                            continue
                        seen.add(text)
                        out.append(text)
                    sections[kt] = out
    if not dry_run and (removed or deduped):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return removed, deduped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="код направления")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pattern = f"krm_disciplines_{args.only}.json" if args.only else "krm_disciplines_*.json"
    total_rm = total_dd = 0
    for path in sorted(REFERENCE.glob(pattern)):
        if "_clean" in path.name:
            continue
        # Годовые файлы (krm_..._2024.json) чистим только по явному --only.
        if not args.only and re.search(r"_\d{4}\.json$", path.name):
            continue
        rm, dd = clean_file(path, args.dry_run)
        total_rm += rm
        total_dd += dd
        if rm or dd:
            print(f"{path.name}: удалено мусора {rm}, дублей {dd}"
                  + (" (dry-run)" if args.dry_run else ""))
    print(f"ИТОГО: мусор {total_rm}, дубли {total_dd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
