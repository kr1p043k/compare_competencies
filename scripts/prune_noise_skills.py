"""Удаление «навыков»-мусора, попавшего из PDF.

Парсер иногда сохранял как навык абзац, строку расписания или фразу из
методички: «1 семестра изучения дисциплины.», «2 – – 1 семинар алгоритмов…»,
«зачет с оценкой». Такие записи не являются навыками: они не сопоставляются с
рынком, раздувают знаменатель покрытия и искажают рекомендации.

Критерии (консервативные, чтобы не удалить настоящий навык):
  - длина > MAX_LEN (абзац вместо термина);
  - несколько предложений: ". " встречается >= 2 раз;
  - служебные слова документа (семестр/зачёт/лекция/отчёт/балл) И длина > 40;
  - строка расписания: начинается с числа/тире и содержит "–".

Правки применяются и к БД (связи competency_skills), и к KRM-файлу года, иначе
источники снова разойдутся.

Использование:
    python scripts/prune_noise_skills.py --direction 09.03.02 --year 2025
    python scripts/prune_noise_skills.py --direction 09.03.02 --year 2025 --apply
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

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"
DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env

MAX_LEN = 120
MIN_LEN_DOC_WORDS = 40
DOC_WORDS = re.compile(
    r"(семестр|зач[её]т|экзамен|лекци|практическ|отч[её]т|балл|аттестац)",
    re.IGNORECASE,
)
SCHEDULE = re.compile(r"^\s*\d+\s*[–—-]")
SENTENCE = re.compile(r"\.\s")


def noise_reason(name: str) -> str | None:
    """Почему строку считаем мусором. None - похоже на настоящий навык."""
    if not name:
        return "пусто"
    text = name.strip()
    if len(text) > MAX_LEN:
        return f"абзац ({len(text)} символов)"
    if len(SENTENCE.findall(text)) >= 2:
        return "несколько предложений"
    if len(text) > MIN_LEN_DOC_WORDS and DOC_WORDS.search(text):
        return "фрагмент документа"
    if SCHEDULE.match(text) and "–" in text:
        return "строка расписания"
    return None


def krm_path(direction: str, year: int) -> Path:
    year_file = REFERENCE_DIR / f"krm_disciplines_{direction}_{year}.json"
    return year_file if year_file.exists() else REFERENCE_DIR / f"krm_disciplines_{direction}.json"


def run(direction: str, year: int, dsn: str, apply: bool) -> int:
    import psycopg

    conn = psycopg.connect(dsn, connect_timeout=15)
    try:
        rows = conn.execute(
            """SELECT DISTINCT s.id::text, s.name
                 FROM skills s
                 JOIN competency_skills cs ON cs.skill_id = s.id
                 JOIN competencies c ON c.id = cs.competency_id
                 JOIN disciplines d ON d.id = c.discipline_id
                 JOIN directions dr ON dr.id = d.direction_id
                WHERE dr.code = %s AND d.year = %s""",
            (direction, year),
        ).fetchall()
    except Exception:
        conn.close()
        raise

    flagged = [(sid, name, noise_reason(name)) for sid, name in rows]
    flagged = [f for f in flagged if f[2]]
    print(f"{direction} {year}: навыков в связях {len(rows)}, мусорных {len(flagged)}")
    for _sid, name, reason in flagged[:15]:
        print(f"   [{reason}] {name[:70]}")

    # KRM: те же критерии, чтобы файл и БД не разошлись.
    path = krm_path(direction, year)
    data = json.loads(path.read_text(encoding="utf-8"))
    sub = data.get(direction) or next(iter(data.values()))
    krm_discs = sub.get("disciplines", {})
    krm_hits = 0
    for body in krm_discs.values():
        skills = body.get("skills") or {}
        for code, items in list(skills.items()):
            kept = [s for s in items if not noise_reason(s)]
            krm_hits += len(items) - len(kept)
            skills[code] = kept
    print(f"KRM {path.name}: связей к удалению {krm_hits}")

    if not apply:
        conn.close()
        print("\n(dry-run; добавьте --apply)")
        return 0

    try:
        with conn.cursor() as cur:
            removed = 0
            for sid, _name, _reason in flagged:
                cur.execute(
                    """DELETE FROM competency_skills
                        WHERE skill_id = %s
                          AND competency_id IN (
                            SELECT c.id FROM competencies c
                            JOIN disciplines d ON d.id = c.discipline_id
                            JOIN directions dr ON dr.id = d.direction_id
                            WHERE dr.code = %s AND d.year = %s)""",
                    (sid, direction, year),
                )
                removed += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nудалено связей в БД: {removed}; в KRM: {krm_hits}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", default="09.03.02")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--dsn", default="")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)
    return run(args.direction, args.year, args.dsn, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
