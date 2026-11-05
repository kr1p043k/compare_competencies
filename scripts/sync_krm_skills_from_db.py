"""Перенос навыков из БД в KRM-файл года.

После обратного заполнения (616 связей `competency_skills` для набора 2024,
построенных из KSA) файл KRM остался без секции `skills`: он собран из
аннотаций, где навыков нет. Из-за этого проверка согласованности честно
падала — файл и БД описывали разные состояния.

Скрипт дописывает в KRM `skills` по данным БД. Структура сохраняется:
`skills` — `{код: [тексты]}`, `ksa` не трогается.

Использование:
    python scripts/sync_krm_skills_from_db.py --direction 09.03.02 --year 2024
    python scripts/sync_krm_skills_from_db.py --direction 09.03.02 --year 2024 --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"
DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env
DEFAULT_YEAR = 2025


def krm_path(direction: str, year: int) -> Path:
    if year == DEFAULT_YEAR:
        return REFERENCE_DIR / f"krm_disciplines_{direction}.json"
    return REFERENCE_DIR / f"krm_disciplines_{direction}_{year}.json"


def run(direction: str, year: int, dsn: str, apply: bool) -> int:
    import psycopg

    path = krm_path(direction, year)
    data = json.loads(path.read_text(encoding="utf-8"))
    sub = data.get(direction) or next(iter(data.values()))
    disciplines = sub.get("disciplines") or {}

    with psycopg.connect(dsn, connect_timeout=15) as conn:
        rows = conn.execute(
            """SELECT d.name, c.code, s.name
                 FROM competency_skills cs
                 JOIN competencies c ON c.id = cs.competency_id
                 JOIN disciplines d ON d.id = c.discipline_id
                 JOIN directions dr ON dr.id = d.direction_id
                 JOIN skills s ON s.id = cs.skill_id
                WHERE dr.code = %s AND d.year = %s AND cs.ksa_type = 'flat'
                ORDER BY d.name, c.code, cs.id""",
            (direction, year),
        ).fetchall()

    by_disc: dict[str, dict[str, list[str]]] = {}
    for disc, code, skill in rows:
        if not disc or not code or not skill:
            continue
        bucket = by_disc.setdefault(disc, {}).setdefault(code, [])
        text = str(skill).strip()
        if text and text not in bucket:
            bucket.append(text)

    changed = 0
    for disc, skills in by_disc.items():
        body = disciplines.get(disc)
        if not isinstance(body, dict):
            continue
        current = body.get("skills") or {}
        if current == skills:
            continue
        changed += 1
        body["skills"] = skills

    total = sum(len(v) for codes in by_disc.values() for v in codes.values())
    print(f"{direction} {year}: дисциплин со связями {len(by_disc)}, связей {total}, "
          f"обновить записей {changed}")

    if not apply:
        print("(dry-run; добавьте --apply)")
        return 0

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"записан KRM: {path.name}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", default="09.03.02")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--dsn", default="")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)
    return run(args.direction, args.year, args.dsn, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
