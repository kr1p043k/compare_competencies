"""Связи competency_skills из собственных KSA записи.

Для набора 2024 в БД есть компетенции и KSA (616 записей), но нет ни одной
связи `competency_skills`: KRM 2024 собран из аннотаций, где перечислены
компетенции, но не навыки. Анализ читает коды компетенций именно из этих
связей, поэтому без них 2024 уходил в файл-фолбэк и терял рыночные метрики.

Скрипт строит плоские связи (`ksa_type='flat'`) из KSA самой дисциплины:
каждая запись KSA становится навыком своей компетенции — ровно так же, как
это выглядит в KRM 2025 после слияния аннотаций.

Использование:
    python scripts/backfill_links_from_ksa.py --direction 09.03.02 --year 2024
    python scripts/backfill_links_from_ksa.py --direction 09.03.02 --year 2024 --apply
"""

from __future__ import annotations

import argparse
import sys

import psycopg
try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env


def run(direction: str, year: int, dsn: str, apply: bool) -> int:
    with psycopg.connect(dsn, connect_timeout=15) as conn:
        rows = conn.execute(
            """
            SELECT c.id::text AS competency_id, c.code, COALESCE(k.cleaned_text, k.original_text) AS text
              FROM competencies c
              JOIN disciplines d ON d.id = c.discipline_id
              JOIN directions dr ON dr.id = d.direction_id
              LEFT JOIN ksa_entries k ON k.competency_id = c.id
             WHERE dr.code = %s AND d.year = %s
               AND COALESCE(k.cleaned_text, k.original_text) IS NOT NULL
            """,
            (direction, year),
        ).fetchall()
        existing = conn.execute(
            """
            SELECT count(*) FROM competency_skills cs
              JOIN competencies c ON c.id = cs.competency_id
              JOIN disciplines d ON d.id = c.discipline_id
              JOIN directions dr ON dr.id = d.direction_id
             WHERE dr.code = %s AND d.year = %s
            """,
            (direction, year),
        ).fetchone()[0]

    pairs = [(comp, text.strip()) for comp, _code, text in rows if text and text.strip()]
    unique_pairs = list(dict.fromkeys(pairs))
    print(f"{direction} {year}: связей сейчас {existing}, "
          f"KSA-текстов {len(pairs)}, уникальных пар {len(unique_pairs)}")

    if not apply:
        print("(dry-run; добавьте --apply)")
        return 0

    inserted = 0
    with psycopg.connect(dsn, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            for comp_id, text in unique_pairs:
                name = text[:500]
                row = cur.execute(
                    "SELECT id::text FROM skills WHERE lower(name) = lower(%s) LIMIT 1",
                    (name,),
                ).fetchone()
                if row:
                    skill_id = row[0]
                else:
                    skill_id = cur.execute(
                        "INSERT INTO skills (name, source) VALUES (%s, 'rpd_skills')"
                        " RETURNING id::text",
                        (name,),
                    ).fetchone()[0]
                cur.execute(
                    "INSERT INTO competency_skills"
                    " (competency_id, skill_id, ksa_type, source_text, match_type)"
                    " VALUES (%s, %s, 'flat', %s, 'explicit')"
                    " ON CONFLICT DO NOTHING",
                    (comp_id, skill_id, name),
                )
                inserted += 1
        conn.commit()
    print(f"вставлено связей: {inserted}")
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
