"""Сверка дисциплин направления с официальным перечнем образовательной программы.

Источник истины — страница программы на сайте вуза
(`data/reference/official_curriculum_<code>.json`). БД и KRM-файлы наполняются
распознаванием РПД, поэтому имена там расходятся: сокращения, потерянные слова,
латинские дубли. Скрипт показывает, чего не хватает и что лишнее.

    python scripts/audit_curriculum_coverage.py
    python scripts/audit_curriculum_coverage.py --direction 09.03.02 --json report.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "data" / "reference"
DEFAULT_DSN = "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies"

# Дисциплины, которых нет в перечне по смыслу: практики, физкультура, ИЯ и т.п.
# Их отсутствие в БД — не дефект данных, а осознанное сужение анализа.
NON_ANALYTICAL_MARKERS = (
    "иностранный язык",
    "академический курс иностранного",
    "физическ",  # ловит "физическая культура" и "...физической культуре"
    "практика",
    "подготовка к процедуре защиты",
    "безопасность жизнедеятельности",
    "история россии",
    "основы российской государственности",
    "философия",
    "логика и теория аргументации",
    "письмо и мышление",
    "воображение, изображение",
    "принципы естественнонаучного познания",
    "цифровая культура",
    "общественн",
    "основы проектной деятельности",
    "психологи",
    "стрессоустойчив",
    "эмоциональный интеллект",
    "экономико-правов",
    "экономическ",
    "экономика инженерии",
    "маркетинг",
    "междисциплинарный проект",
)


# БД хранит сокращения, официальный перечень — полные названия
# ("Генеративный ИИ и МО" против "Генеративный искусственный интеллект и
# машинное обучение"). Раскрываем ТОЛЬКО заглавные токены: строчное "по" в
# "Практикум по математике" — предлог, его трогать нельзя.
ABBREV = {
    "ИИ": "искусственный интеллект",
    "ПО": "программное обеспечение",
    "МО": "машинное обучение",
    "ИС": "информационные системы",
    "БД": "базы данных",
    "ПК": "программный комплекс",
    "АСУ": "автоматизированные системы управления",
}


def expand_abbrev(name: str) -> str:
    out = []
    for token in (name or "").split():
        bare = token.strip(".,;:()\u00ab\u00bb\"'").upper()
        if bare in ABBREV:
            out.append(ABBREV[bare])
        else:
            out.append(token)
    return " ".join(out)


def norm(name: str) -> str:
    expanded = expand_abbrev(name or "")
    return "".join(ch for ch in expanded.lower() if ch.isalnum())


def strip_parens(name: str) -> str:
    """'Databases and DBMS (Базы данных и СУБД)' -> 'Базы данных и СУБД'."""
    if "(" in name and name.rstrip().endswith(")"):
        inner = name[name.rindex("(") + 1: name.rstrip().rfind(")")].strip()
        if inner:
            return inner
    return name


def load_official(direction: str, year: int) -> dict:
    path = REFERENCE / f"official_curriculum_{direction}_{year}.json"
    if not path.exists():
        path = REFERENCE / f"official_curriculum_{direction}.json"
    if not path.exists():
        raise SystemExit(f"нет официального перечня: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_db_disciplines(dsn: str, direction: str) -> list[str]:
    try:
        with psycopg.connect(dsn, connect_timeout=10) as conn:
            rows = conn.execute(
                """
                SELECT disc.name FROM disciplines disc
                JOIN directions d ON d.id = disc.direction_id
                WHERE d.code = %s ORDER BY disc.name
                """,
                (direction,),
            ).fetchall()
    except Exception as exc:  # локальная БД может быть недоступна
        print(f"БД недоступна ({exc}); сверяю только перечень с KRM-файлом", file=sys.stderr)
        return []
    return [r[0] for r in rows]


def load_krm_disciplines(direction: str) -> list[str]:
    path = REFERENCE / f"krm_disciplines_{direction}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    # Формат: {"09.03.02": {"direction_name": ..., "disciplines": {...|<list>}}}
    node = data.get(direction) if isinstance(data, dict) else None
    if isinstance(node, dict):
        discs = node.get("disciplines")
        if isinstance(discs, dict):
            return [d.get("name", k) if isinstance(d, dict) else k for k, d in discs.items()]
        if isinstance(discs, list):
            return [d.get("name") if isinstance(d, dict) else d for d in discs if d]
        return list(node.keys())
    if isinstance(data, dict):
        return list(data.keys())
    return []


def match(official: list[str], actual: list[str]) -> tuple[dict, list[str], list[str]]:
    actual_norm = {norm(a): a for a in actual}
    pairs: dict[str, str] = {}
    missing: list[str] = []
    used: set[str] = set()

    for off in official:
        candidates = [off, strip_parens(off)]
        hit = None
        for cand in candidates:
            key = norm(cand)
            if key in actual_norm:
                hit = key
                break
        if hit is None:
            keys = list(actual_norm)
            close = difflib.get_close_matches(norm(strip_parens(off)), keys, n=1, cutoff=0.86)
            hit = close[0] if close else None
        if hit:
            pairs[off] = actual_norm[hit]
            used.add(hit)
        else:
            missing.append(off)

    extra = [v for k, v in actual_norm.items() if k not in used]
    return pairs, missing, extra


def is_analytical(name: str) -> bool:
    low = name.lower()
    return not any(m in low for m in NON_ANALYTICAL_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="09.03.02")
    parser.add_argument("--year", type=int, default=2025,
                        help="год набора: файл official_curriculum_<код>_<год>.json")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    official_doc = load_official(args.direction, args.year)
    official = official_doc["disciplines"]
    db_names = load_db_disciplines(args.dsn, args.direction)
    krm_names = load_krm_disciplines(args.direction)

    print(f"{official_doc['direction_code']} {official_doc['direction_name']} "
          f"(набор {official_doc.get('year')})")
    print(f"источник: {official_doc['source_url']}")
    print(f"официальный перечень: {len(official)}")
    print(f"БД: {len(db_names)}  KRM-файл: {len(krm_names)}\n")

    report: dict = {"direction": args.direction, "source": official_doc["source_url"]}

    for label, actual in (("БД", db_names), ("KRM", krm_names)):
        if not actual:
            continue
        pairs, missing, extra = match(official, actual)
        gap = [m for m in missing if is_analytical(m)]
        print(f"--- {label}: сопоставлено {len(pairs)} из {len(official)}")
        print(f"    нет в {label}, но есть в программе: {len(missing)} "
              f"(из них профильных: {len(gap)})")
        for m in gap:
            print(f"      · {m}")
        if extra:
            print(f"    есть в {label}, нет в официальном перечне: {len(extra)}")
            for e in extra:
                print(f"      ? {e}")
        overlap = len(pairs) / max(len(official), 1)
        if overlap < 0.5:
            print(f"    !! пересечение {overlap:.0%}: похоже, перечень и {label} — "
                  f"разные версии учебного плана, а не дефект данных")
        report[label] = {
            "overlap": round(overlap, 3),
            "matched": len(pairs),
            "missing_total": len(missing),
            "missing_analytical": gap,
            "extra": extra,
            "pairs": pairs,
        }

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nотчёт: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
