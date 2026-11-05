"""Аудит «размазанности» навыков по дисциплинам.

Навык, встречающийся почти у всех дисциплин, ничего не различает: он не
помогает ни покрытию, ни рекомендациям, но раздувает знаменатель. Обратная
крайность - навык, привязанный к одной компетенции, но лексически общий.

Скрипт считает распределение и помечает:
  - WIDE: навык у >= порога (по умолчанию 40%) дисциплин направления;
  - RARE: навык ровно у одной дисциплины (кандидат в опечатку или шум);
  - HOMOGLYPH: пары навыков, различающиеся только латиницей/кириллицей.

Использование:
    python scripts/audit_skill_spread.py --direction 09.03.02 --year 2025
    python scripts/audit_skill_spread.py --direction 09.03.02 --wide 0.3 --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env

_LATIN = set("abcdefghijklmnopqrstuvwxyz")
_CYR = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")


def _homoglyph_key(name: str) -> str:
    """Кириллические буквы-двойники латинских схлопываются в латинские."""
    table = str.maketrans({
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
        "к": "k", "м": "m", "т": "t", "в": "b", "н": "h", "і": "i", "ј": "j",
    })
    return re.sub(r"\s+", " ", name.lower().translate(table)).strip()


def collect(dsn: str, direction: str, year: int) -> tuple[dict[str, set[str]], int]:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=15) as conn:
        total = conn.execute(
            """SELECT count(*) FROM disciplines d
               JOIN directions dr ON dr.id = d.direction_id
               WHERE dr.code = %s AND d.year = %s""",
            (direction, year),
        ).fetchone()[0]
        rows = conn.execute(
            """SELECT s.name, d.name
                 FROM skills s
                 JOIN competency_skills cs ON cs.skill_id = s.id
                 JOIN competencies c ON c.id = cs.competency_id
                 JOIN disciplines d ON d.id = c.discipline_id
                 JOIN directions dr ON dr.id = d.direction_id
                WHERE dr.code = %s AND d.year = %s""",
            (direction, year),
        ).fetchall()

    spread: dict[str, set[str]] = {}
    for skill, discipline in rows:
        if not skill:
            continue
        spread.setdefault(skill, set()).add(discipline)
    return spread, total


def report(spread: dict[str, set[str]], total: int, wide: float) -> dict:
    ranked = sorted(spread.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    threshold = max(1, int(total * wide))
    wide_items = [
        {"skill": name, "disciplines": len(ds), "pct": round(100 * len(ds) / total, 1)}
        for name, ds in ranked if len(ds) >= threshold
    ]
    rare_items = [
        {"skill": name, "disciplines": len(ds), "only_in": sorted(ds)[0]}
        for name, ds in ranked if len(ds) == 1
    ]

    groups: dict[str, list[str]] = {}
    for name in spread:
        groups.setdefault(_homoglyph_key(name), []).append(name)
    homoglyphs = [
        sorted(names) for names in groups.values()
        if len(names) > 1 and len({n for n in names}) > 1
        and any(set(n.lower()) & _CYR for n in names)
        and any(set(n.lower()) & _LATIN for n in names)
    ]

    return {
        "total_disciplines": total,
        "distinct_skills": len(spread),
        "wide_threshold_pct": round(wide * 100, 1),
        "wide": wide_items,
        "rare_count": len(rare_items),
        "rare_sample": rare_items[:20],
        "homoglyph_pairs": homoglyphs[:20],
        "top": [
            {"skill": n, "disciplines": len(ds)} for n, ds in ranked[:20]
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", default="09.03.02")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--dsn", default="")
    p.add_argument("--wide", type=float, default=0.4,
                   help="порог 'размазанности' как доля дисциплин (0.4 = 40%%)")
    p.add_argument("--json", dest="json_path", default=None)
    args = p.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)

    spread, total = collect(args.dsn, args.direction, args.year)
    if not total:
        print(f"нет дисциплин для {args.direction} {args.year}")
        return 1
    data = report(spread, total, args.wide)

    print(f"{args.direction} {args.year}: дисциплин {total}, "
          f"уникальных навыков {data['distinct_skills']}")
    print(f"\nWIDE (>= {data['wide_threshold_pct']}% дисциплин): {len(data['wide'])}")
    for item in data["wide"][:15]:
        print(f"   {item['pct']:5.1f}%  {item['skill']}  ({item['disciplines']})")
    print(f"\nRARE (ровно в одной дисциплине): {data['rare_count']}")
    for item in data["rare_sample"][:10]:
        print(f"   {item['skill']}  -> {item['only_in']}")
    if data["homoglyph_pairs"]:
        print(f"\nHOMOGLYPH-пары: {len(data['homoglyph_pairs'])}")
        for pair in data["homoglyph_pairs"][:10]:
            print(f"   {pair}")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
