"""Дополнить KRM-файл дисциплинами, которые есть в БД, но не заведены в KRM.

KRM-файл (`data/reference/krm_disciplines_<код>.json`) — метаданные дисциплин
для учительского анализа: перечень компетенций, привязанные навыки и KSA-тексты.
Он отстаёт от БД: часть дисциплин (в основном AI/ML-блок набора 2025) есть в
базе, но в файле отсутствует, из-за чего анализ идёт без полного профиля.

Скрипт строит для таких дисциплин запись из БД в том же формате:

    {"competencies": [code, ...],
     "skills": {code: [навык, ...]},
     "ksa":    {code: {"knowledge": [...], "abilities": [...], "skills": [...]}}}

Существующие записи не перетираются без --overwrite.

    python scripts/sync_krm_from_db.py --dry-run
    python scripts/sync_krm_from_db.py
    python scripts/sync_krm_from_db.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import psycopg

try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "data" / "reference"
DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env

DISCIPLINE_CODES = """
select c.code, c.name from competencies c
where c.discipline_id = (select id from disciplines where name = %s limit 1)
  and c.code is not null
order by c.code
"""

SKILLS = """
select c.code, s.name
from competencies c
join competency_skills cs on cs.competency_id = c.id
join skills s on s.id = cs.skill_id
where c.discipline_id = (select id from disciplines where name = %s limit 1)
  and c.code is not null
order by c.code, s.name
"""

KSA = """
select c.code, k.ksa_type::text, k.cleaned_text
from competencies c
join ksa_entries k on k.competency_id = c.id
where c.discipline_id = (select id from disciplines where name = %s limit 1)
  and c.code is not null
  and coalesce(nullif(trim(k.cleaned_text), ''), '') <> ''
order by c.code, k.ksa_type, k.cleaned_text
"""


def build_entry(conn, discipline: str) -> dict | None:
    codes = conn.execute(DISCIPLINE_CODES, (discipline,)).fetchall()
    if not codes:
        return None

    skills: dict[str, list[str]] = {}
    for code, name in conn.execute(SKILLS, (discipline,)):
        skills.setdefault(code, []).append(name)

    ksa: dict[str, dict[str, list[str]]] = {}
    for code, ksa_type, text in conn.execute(KSA, (discipline,)):
        ksa.setdefault(code, {}).setdefault(ksa_type, []).append(text)

    entry: dict = {"competencies": [c for c, _ in codes], "skills": skills}
    if ksa:
        entry["ksa"] = ksa
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="09.03.02")
    parser.add_argument("--dsn", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="перестроить и уже существующие записи")
    args = parser.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    path = REFERENCE / f"krm_disciplines_{args.direction}.json"
    if not path.exists():
        raise SystemExit(f"нет KRM-файла: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    node = doc.setdefault(args.direction, {})
    discs: dict = node.setdefault("disciplines", {})

    with psycopg.connect(args.dsn, connect_timeout=15) as conn:
        db_names = [
            r[0] for r in conn.execute(
                """select disc.name from disciplines disc
                   join directions d on d.id = disc.direction_id
                   where d.code = %s order by disc.name""",
                (args.direction,),
            )
        ]
        targets = [
            n for n in db_names
            if args.overwrite or n not in discs
        ]
        print(f"в БД {len(db_names)}, в KRM {len(discs)}, к добавлению {len(targets)}")

        added, skipped = [], []
        for name in targets:
            entry = build_entry(conn, name)
            if not entry:
                skipped.append((name, "нет компетенций"))
                continue
            discs[name] = entry
            added.append((name, len(entry["competencies"]),
                          len(entry.get("ksa", {}))))
            print(f"  + {name[:60]:60} компетенций={len(entry['competencies'])} ksa={len(entry.get('ksa', {}))}")

    if skipped:
        print("\nпропущено:")
        for name, why in skipped:
            print(f"  - {name[:60]:60} {why}")

    if args.dry_run:
        print("\ndry run: файл не изменён")
        return 0

    backup = path.with_suffix(
        path.suffix + f".bak-{datetime.now():%Y%m%d-%H%M%S}"
    )
    shutil.copy2(path, backup)
    node["disciplines"] = dict(sorted(discs.items()))
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nдобавлено {len(added)}, всего в KRM {len(discs)}")
    print(f"бэкап: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
