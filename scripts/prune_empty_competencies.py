"""Remove competency shells that carry no information.

A competency is a shell when it has no name, no description and no linked
skills. They come from parsing runs that created parent rows but never filled
them, and they only pollute listings and coverage denominators.

Children are reparented to the shell's parent before the shell is deleted, so
no branch is orphaned.

Usage:
    python scripts/prune_empty_competencies.py --dry-run
    python scripts/prune_empty_competencies.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg

DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env
ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "data" / "reports" / "empty_competencies_backup.json"

SELECT_SHELLS = """
select c.id, c.code, c.parent_id, c.category
from competencies c
try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402
where coalesce(trim(c.name), '') = ''
  and coalesce(trim(c.description), '') = ''
  and not exists (select 1 from competency_skills cs where cs.competency_id = c.id)
order by c.code nulls last
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    with psycopg.connect(args.dsn, connect_timeout=15) as conn:
        shells = conn.execute(SELECT_SHELLS).fetchall()
        print(f"empty competency shells: {len(shells)}")
        for row in shells:
            print(f"  id={row[0]} code={row[1]!r} parent={row[2]} category={row[3]!r}")

        if not shells:
            return 0

        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        BACKUP.write_text(
            json.dumps(
                [
                    {"id": str(r[0]), "code": r[1], "parent_id": str(r[2]) if r[2] else None,
                     "category": r[3]}
                    for r in shells
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"backup: {BACKUP}")

        if args.dry_run:
            print("dry run: nothing deleted")
            return 0

        reparented = 0
        with conn.cursor() as cur:
            for comp_id, _code, parent_id, _category in shells:
                cur.execute(
                    "update competencies set parent_id = %s where parent_id = %s",
                    (parent_id, comp_id),
                )
                reparented += cur.rowcount
                cur.execute("delete from competencies where id = %s", (comp_id,))
        conn.commit()

        print(f"deleted {len(shells)} shells, reparented {reparented} children")
        left = conn.execute(
            "select count(*) from competencies where coalesce(trim(name),'') = ''"
        ).fetchone()[0]
        print(f"remaining nameless competencies: {left}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
