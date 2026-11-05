"""Snapshot market skill demand into market_skill_mappings, one row per period.

market_skill_mappings had a single period and no writer in the codebase, so
market trend analysis could not work from it. This script rebuilds the table
from vacancies.parsed_skills, grouped by the month a vacancy was published.

For each period it stores:
  frequency  raw number of vacancies mentioning the skill
  weight     frequency / total vacancies in that period (comparable across months)

skill_id is resolved against skills by normalized name and left NULL when the
skill is not in the curated table (the row is still useful for trends).

Usage:
    python scripts/snapshot_market_skills.py --dry-run
    python scripts/snapshot_market_skills.py
    python scripts/snapshot_market_skills.py --period 2026-10-01 --replace
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import psycopg

DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env

UNIQUE_INDEX = """
create unique index if not exists uq_msm_period_skill
    on market_skill_mappings (period, market_skill_name)
"""

AGGREGATE = """
with per_vacancy as (
    select v.id,
           date_trunc('month', coalesce(v.published_at, v.created_at))::date as period,
           lower(trim(skill)) as skill
    from vacancies v,
         jsonb_array_elements_text(v.parsed_skills) as skill
    where v.parsed_skills is not null
      and jsonb_array_length(v.parsed_skills) > 0
      and coalesce(v.published_at, v.created_at) is not null
      and trim(skill) <> ''
),
totals as (
    select period, count(distinct id) as n_vac
    from per_vacancy group by period
),
freqs as (
    select period, skill, count(distinct id) as frequency
    from per_vacancy group by period, skill
)
select f.period, f.skill, f.frequency, t.n_vac,
       f.frequency::double precision / nullif(t.n_vac, 0) as weight
from freqs f join totals t using (period)
try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402
order by f.period, f.frequency desc
"""

UPSERT = """
insert into market_skill_mappings
    (id, skill_id, market_skill_name, frequency, weight, period, source, created_at)
values (gen_random_uuid(), %s, %s, %s, %s, %s, 'vacancies_snapshot', now())
on conflict (period, market_skill_name) do update
   set frequency = excluded.frequency,
       weight = excluded.weight,
       skill_id = coalesce(excluded.skill_id, market_skill_mappings.skill_id),
       source = excluded.source
"""


def resolve_skill_ids(conn) -> dict[str, str]:
    rows = conn.execute(
        "select id::text, lower(trim(name)) from skills"
    ).fetchall()
    return {name: skill_id for skill_id, name in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--period", type=date.fromisoformat, default=None)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    with psycopg.connect(args.dsn, connect_timeout=20) as conn:
        rows = conn.execute(AGGREGATE).fetchall()
        skill_ids = resolve_skill_ids(conn)
        print(f"aggregated rows: {len(rows)}  skills in taxonomy: {len(skill_ids)}")

        by_period: dict[date, list[tuple]] = {}
        for period, skill, frequency, n_vac, weight in rows:
            if args.period and period != args.period:
                continue
            if frequency < args.min_frequency:
                continue
            by_period.setdefault(period, []).append((skill, frequency, weight, n_vac))

        total = 0
        unmatched = 0
        for period in sorted(by_period):
            items = by_period[period]
            resolved = 0
            print(f"  {period}: {len(items)} skills, {items[0][3]} vacancies")
            if args.dry_run:
                total += len(items)
                continue

            conn.execute(UNIQUE_INDEX)
            with conn.cursor() as cur:
                for skill, frequency, weight, _ in items:
                    skill_id = skill_ids.get(skill)
                    if skill_id:
                        resolved += 1
                    else:
                        unmatched += 1
                    cur.execute(
                        UPSERT,
                        (skill_id, skill, frequency, round(float(weight), 6), period),
                    )
            conn.commit()
            total += len(items)
            print(f"      upserted {len(items)} rows, skill_id resolved {resolved}")

        print(f"\ntotal rows written: {total}" + (" (dry run)" if args.dry_run else ""))
        if not args.dry_run:
            stats = conn.execute(
                "select count(distinct period), min(period), max(period), count(*) "
                "from market_skill_mappings"
            ).fetchone()
            print(f"table now: periods={stats[0]} range={stats[1]}..{stats[2]} rows={stats[3]}")
            print(f"rows without a taxonomy match: {unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
