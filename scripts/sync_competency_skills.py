"""Синхронизация competency_skills с KRM.

БД объявлена единственным источником истины для анализа, но связи
«компетенция -> навык» копились инкрементально: сид добавляет новые и не
удаляет устаревшие. В результате в БД лежало 11 238 связей от старых версий
KRM, тогда как актуальный файл даёт другой набор, и анализ из БД расходился
с анализом из файла по 96 кодам из 137.

Скрипт пересобирает плоские связи (ksa_type='flat') для направления и года
строго по KRM: удаляет прежние и вставляет текущие. Навыки, которых нет в
справочнике skills, создаются (source='rpd_skills').

Использование:
    python scripts/sync_competency_skills.py --direction 09.03.02 --year 2025
    python scripts/sync_competency_skills.py --direction 09.03.02 --year 2025 --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg
try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"
DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env


def krm_path(direction: str, year: int) -> Path:
    year_file = REFERENCE_DIR / f"krm_disciplines_{direction}_{year}.json"
    if year_file.exists():
        return year_file
    return REFERENCE_DIR / f"krm_disciplines_{direction}.json"


def load_krm(direction: str, year: int) -> dict[str, dict[str, list[str]]]:
    """{discipline: {code: [skill, ...]}} из KRM года."""
    path = krm_path(direction, year)
    data = json.loads(path.read_text(encoding="utf-8"))
    sub = data.get(direction) or next(iter(data.values()))
    out: dict[str, dict[str, list[str]]] = {}
    for name, body in (sub.get("disciplines") or {}).items():
        skills = body.get("skills") or {}
        if skills:
            out[name] = {code: list(items or []) for code, items in skills.items()}
    return out


def sync(direction: str, year: int, dsn: str, apply: bool) -> int:
    krm = load_krm(direction, year)
    wanted = sum(len(items) for codes in krm.values() for items in codes.values())
    print(f"KRM {direction} {year}: дисциплин {len(krm)}, связей в KRM {wanted}")

    with psycopg.connect(dsn, connect_timeout=15) as conn:
        courses = conn.execute(
            """
            SELECT d.name, c.id::text, c.code
              FROM disciplines d
              JOIN directions dr ON dr.id = d.direction_id
              JOIN competencies c ON c.discipline_id = d.id
             WHERE dr.code = %s AND d.year = %s
            """,
            (direction, year),
        ).fetchall()
        current = conn.execute(
            """
            SELECT count(*)
              FROM competency_skills cs
              JOIN competencies c ON c.id = cs.competency_id
              JOIN disciplines d ON d.id = c.discipline_id
              JOIN directions dr ON dr.id = d.direction_id
             WHERE dr.code = %s AND d.year = %s AND cs.ksa_type = 'flat'
            """,
            (direction, year),
        ).fetchone()[0]
        print(f"в БД сейчас: компетенций {len(courses)}, плоских связей {current}")

        if not apply:
            print("(dry-run; добавьте --apply)")
            return 0

        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM competency_skills cs
                 USING competencies c, disciplines d, directions dr
                 WHERE cs.competency_id = c.id
                   AND c.discipline_id = d.id
                   AND d.direction_id = dr.id
                   AND dr.code = %s AND d.year = %s
                   AND cs.ksa_type = 'flat'
                """,
                (direction, year),
            )
            removed = cur.rowcount
            by_key = {(name, code): comp_id for name, comp_id, code in courses}

            inserted = 0
            for disc, codes in krm.items():
                for code, items in codes.items():
                    comp_id = by_key.get((disc, code))
                    if not comp_id:
                        continue
                    for skill_name in items:
                        name = str(skill_name).strip()
                        if not name:
                            continue
                        row = cur.execute(
                            "SELECT id::text FROM skills WHERE lower(name) = lower(%s)"
                            " ORDER BY id LIMIT 1",
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
                        # match_type ограничен CHECK-ом таблицы:
                        # exact|fuzzy|stem|substring|semantic|explicit.
                        # Связь из KRM утверждена явно (не выведена матчером),
                        # поэтому 'explicit'. Раньше здесь стоял 'krm_sync' -
                        # он не проходил CHECK, и скрипт падал целиком.
                        cur.execute(
                            "INSERT INTO competency_skills"
                            " (competency_id, skill_id, ksa_type, source_text, match_type)"
                            " VALUES (%s, %s, 'flat', %s, 'explicit')"
                            " ON CONFLICT DO NOTHING",
                            (comp_id, skill_id, name),
                        )
                        # rowcount, а не ++: ON CONFLICT DO NOTHING может пропустить
                        # дубликат, и счётчик врал бы в плюс.
                        inserted += max(cur.rowcount, 0)
        conn.commit()
        print(f"удалено связей {removed}, вставлено {inserted}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", default="09.03.02")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--dsn", default="")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)
    return sync(args.direction, args.year, args.dsn, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
