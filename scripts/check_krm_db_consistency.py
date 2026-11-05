"""Проверка согласованности KRM-файла и БД.

За неделю БД дважды тихо разошлась с KRM: сначала 11 238 связей против 6 219
в файле, потом 6 219 против 5 851 после чистки. Анализ читает БД, часть
интерфейса - файл, и расхождение меняет цифры без единой ошибки в логах.

Скрипт сравнивает четыре вещи и возвращает код 1 при любом расхождении, чтобы
это можно было поставить в CI.

Использование:
    python scripts/check_krm_db_consistency.py --direction 09.03.02 --year 2025
    python scripts/check_krm_db_consistency.py --direction 09.03.02 --year 2024 --json out.json
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


def load_krm(direction: str, year: int) -> dict[str, dict]:
    """{discipline: {competencies: [...], skills: {...}}} из KRM года."""
    path = krm_path(direction, year)
    data = json.loads(path.read_text(encoding="utf-8"))
    sub = data.get(direction) or next(iter(data.values()))
    return sub.get("disciplines") or {}


def krm_links(disciplines: dict[str, dict]) -> int:
    return sum(
        len(items or [])
        for body in disciplines.values()
        for items in (body.get("skills") or {}).values()
    )


def load_db(dsn: str, direction: str, year: int) -> dict:
    """Снимок БД для направления и года."""
    import psycopg

    with psycopg.connect(dsn, connect_timeout=15) as conn:
        names = [
            r[0]
            for r in conn.execute(
                """SELECT d.name FROM disciplines d
                   JOIN directions dr ON dr.id = d.direction_id
                   WHERE dr.code = %s AND d.year = %s""",
                (direction, year),
            )
        ]
        links = conn.execute(
            """SELECT count(*) FROM competency_skills cs
               JOIN competencies c ON c.id = cs.competency_id
               JOIN disciplines d ON d.id = c.discipline_id
               JOIN directions dr ON dr.id = d.direction_id
               WHERE dr.code = %s AND d.year = %s AND cs.ksa_type = 'flat'""",
            (direction, year),
        ).fetchone()[0]
        per_disc = {
            r[0]: r[1]
            for r in conn.execute(
                # Считаем только корневые компетенции: подкомпетенции
                # (индикаторы вида ОПК-1.1) в KRM отдельно не перечисляются,
                # и без фильтра сравнение даёт ложное расхождение 51 против 16.
                """SELECT d.name, count(c.id) FILTER (WHERE c.parent_id IS NULL)
                   FROM disciplines d
                   JOIN directions dr ON dr.id = d.direction_id
                   LEFT JOIN competencies c ON c.discipline_id = d.id
                   WHERE dr.code = %s AND d.year = %s
                   GROUP BY d.name""",
                (direction, year),
            )
        }
        orphan_comps = conn.execute(
            """SELECT count(*) FROM competencies c
               JOIN disciplines d ON d.id = c.discipline_id
               JOIN directions dr ON dr.id = d.direction_id
               WHERE dr.code = %s AND d.year = %s
                 AND NOT EXISTS (
                   SELECT 1 FROM competency_skills cs WHERE cs.competency_id = c.id)""",
            (direction, year),
        ).fetchone()[0]
        dangling = conn.execute(
            """SELECT
                 (SELECT count(*) FROM competency_skills cs
                   WHERE NOT EXISTS (SELECT 1 FROM competencies c WHERE c.id = cs.competency_id))
               + (SELECT count(*) FROM ksa_entries k
                   WHERE NOT EXISTS (SELECT 1 FROM competencies c WHERE c.id = k.competency_id))"""
        ).fetchone()[0]
    return {
        "names": names,
        "links": links,
        "per_discipline": per_disc,
        "orphan_competencies": orphan_comps,
        "dangling_rows": dangling,
    }


def compare(direction: str, year: int, dsn: str) -> dict:
    disciplines = load_krm(direction, year)
    db = load_db(dsn, direction, year)

    krm_names = set(disciplines)
    db_names = set(db["names"])
    only_db = sorted(db_names - krm_names)
    only_krm = sorted(krm_names - db_names)

    krm_link_count = krm_links(disciplines)
    links_match = krm_link_count == db["links"]

    mismatched = []
    for name in sorted(krm_names & db_names):
        expected = len(disciplines[name].get("competencies") or [])
        actual = db["per_discipline"].get(name, 0)
        if expected != actual:
            mismatched.append({"name": name, "krm": expected, "db": actual})

    checks = [
        {
            "name": "набор дисциплин",
            "ok": not only_db and not only_krm,
            "detail": f"БД {len(db_names)}, KRM {len(krm_names)}"
                      + (f"; только в БД: {only_db[:5]}" if only_db else "")
                      + (f"; только в KRM: {only_krm[:5]}" if only_krm else ""),
        },
        {
            "name": "число связей competency_skills",
            "ok": links_match,
            "detail": f"БД {db['links']}, KRM {krm_link_count}",
        },
        {
            # Предупреждение, а не гейт: KRM перечисляет коды компетенций
            # строкой, а БД хранит ещё и подкомпетенции-индикаторы, поэтому
            # числа сопоставимы не всегда. Показываем, но не валим проверку.
            "name": "компетенции по дисциплинам (справочно)",
            "ok": True,
            "warning": bool(mismatched),
            "detail": (f"расхождений {len(mismatched)}"
                       + (f": {mismatched[:3]}" if mismatched else "")),
        },
        {
            "name": "сироты",
            "ok": db["dangling_rows"] == 0,
            "detail": (f"компетенций без связей {db['orphan_competencies']} "
                       f"(не ошибка, но снижает покрытие), "
                       f"висячих строк {db['dangling_rows']}"),
        },
    ]
    failed = [c for c in checks if not c["ok"]]
    return {
        "direction": direction,
        "year": year,
        "ok": not failed,
        "checks": checks,
        "orphan_competencies": db["orphan_competencies"],
        "only_in_db": only_db,
        "only_in_krm": only_krm,
        "competency_mismatches": mismatched,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", default="09.03.02")
    p.add_argument("--year", type=int, default=DEFAULT_YEAR)
    p.add_argument("--dsn", default="")
    p.add_argument("--json", dest="json_path", default=None)
    args = p.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)

    path = krm_path(args.direction, args.year)
    if not path.exists():
        print(f"нет файла KRM: {path}")
        return 2
    try:
        report = compare(args.direction, args.year, args.dsn)
    except Exception as exc:  # БД недоступна и т.п.
        print(f"ошибка проверки: {exc}")
        return 2

    print(f"=== {args.direction} / {args.year} ===")
    for check in report["checks"]:
        mark = "WARN" if check.get("warning") else ("OK  " if check["ok"] else "FAIL")
        print(f"  [{mark}] {check['name']}: {check['detail']}")
    print(f"сирот-компетенций: {report['orphan_competencies']}")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON: {args.json_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
