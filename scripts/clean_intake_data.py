"""Очистка набора (интейка) от «двуязычного мусора» в названиях дисциплин.

Проблема: часть дисциплин попала в БД и KRM с названием вида
``Algorithmization and Programming (Алгоритмизация и программирование)`` —
это английский заголовок из PDF рядом с русским переводом. Такой ключ
дублирует реальный курс ``Алгоритмизация и программирование``, имеет покрытие 0
и должен быть слит/переименован.

Режимы:
    python scripts/clean_intake_data.py --direction 09.03.02 --year 2024   # dry-run
    python scripts/clean_intake_data.py --direction 09.03.02 --year 2024 --apply

Правила:
  * «Двуязычный мусор» — имя вида ``^(.+?)\\s*\\((.+)\\)$``, где внешняя часть
    латинская (содержит латиницу и не содержит кириллицы), а скобочная часть
    кириллическая. Кириллическая часть — это корректное имя.
  * Если дисциплина с таким кириллическим именем уже существует (в БД или KRM) —
    MERGE: переносим компетенции/навыки/KSA из мусорной в целевую, удаляем мусорную.
  * Иначе если кириллическое имя есть в официальном перечне — RENAME (просто
    переименовать).
  * Иначе — SKIP (не трогаем).
  * Guard: переименование, которое столкнулось бы с уже существующей ДРУГОЙ
    дисциплиной (не целью слияния), пропускается.
  * Дисциплину с компетенциями не удаляем, пока её содержимое не перенесено в цель.

Перед любой записью: бэкап KRM-файла в ``<имя>.backup.json`` и SQL-дамп
затронутых строк в ``data/result/cleanup/<dir>_<year>_<timestamp>.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Collection
from datetime import datetime
from pathlib import Path
from uuid import UUID

import psycopg
try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"
RESULT_DIR = ROOT / "data" / "result" / "cleanup"
DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env
DEFAULT_YEAR = 2025  # год, для которого KRM-файл называется без суффикса года

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
# Имя-«мусор»: внешняя часть + скобочная группа (перевод).
BILINGUAL_JUNK_RE = re.compile(r"^(.+?)\s*\((.+)\)$")

PLAN_MERGE = "merge"
PLAN_RENAME = "rename"
PLAN_SKIP = "skip"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # пыхать на нетерминальных потоках не нужно
        pass


# ─── pure helpers (unit-tested, no I/O) ─────────────────────────────────────

def detect_bilingual_junk(name: str) -> str | None:
    """Возвращает кириллическую «внутреннюю» часть, если имя — двуязычный мусор.

    ``Algorithmization and Programming (Алгоритмизация и программирование)``
    -> ``Алгоритмизация и программирование``. ``Базы данных и СУБД`` -> None.
    """
    s = (name or "").strip()
    m = BILINGUAL_JUNK_RE.match(s)
    if not m:
        return None
    outer = m.group(1).strip()
    inner = m.group(2).strip()
    if not outer or not inner:
        return None
    # Внешняя часть латинская: есть латиница и нет кириллицы.
    if not LATIN_RE.search(outer) or CYRILLIC_RE.search(outer):
        return None
    # Скобочная часть кириллическая.
    if not CYRILLIC_RE.search(inner):
        return None
    return inner


def _append_unique(target: list[str], items: Collection[str]) -> list[str]:
    for item in items:
        if item not in target:
            target.append(item)
    return target


def merge_entries(target: dict, alias: dict) -> dict:
    """Слить ``alias`` в ``target`` без дублей, сохранив структуру.

    ``competencies`` — список кодов; ``skills`` — ``{код: [тексты]}``;
    ``ksa`` — ``{код: {тип: [тексты]}}`` (остаётся dict-of-dicts, не сглаживаем).
    Мутирует ``target`` и возвращает его.
    """
    target["competencies"] = _append_unique(
        list(target.get("competencies") or []), alias.get("competencies") or []
    )

    cur_skills = target.setdefault("skills", {})
    if not isinstance(cur_skills, dict):
        cur_skills = {}
        target["skills"] = cur_skills
    for code, items in (alias.get("skills") or {}).items():
        if not isinstance(items, list):
            continue
        lst = list(cur_skills.get(code) or [])
        merged = _append_unique(lst, items)
        if merged or code in cur_skills:
            cur_skills[code] = merged

    cur_ksa = target.setdefault("ksa", {})
    if not isinstance(cur_ksa, dict):
        cur_ksa = {}
        target["ksa"] = cur_ksa
    for code, types in (alias.get("ksa") or {}).items():
        if not isinstance(types, dict):
            continue
        tgt_types = cur_ksa.setdefault(code, {})
        if not isinstance(tgt_types, dict):
            tgt_types = {}
            cur_ksa[code] = tgt_types
        for kt, texts in types.items():
            if not isinstance(texts, list):
                continue
            lst = list(tgt_types.get(kt) or [])
            merged = _append_unique(lst, texts)
            if merged or kt in tgt_types:
                tgt_types[kt] = merged
    return target


def plan_action(
    inner: str,
    *,
    existing_names: Collection[str],
    official_names: Collection[str],
    merge_target: str | None,
) -> str:
    """Решить, что делать с мусорным именем, чья кириллическая часть — ``inner``.

    * PLAN_MERGE — ``inner`` уже существует как дисциплина и является целью
      слияния (переносим содержимое и удаляем мусор);
    * PLAN_RENAME — такого имени нет, но оно есть в официальном перечне
      (просто переименовываем);
    * PLAN_SKIP — иначе, включая случай, когда переименование столкнулось бы с
      уже существующей ДРУГОЙ дисциплиной (не целью слияния).
    """
    if merge_target is not None and merge_target == inner and inner in existing_names:
        return PLAN_MERGE
    if inner in existing_names:
        # Имя занято другой дисциплиной, а цели слияния нет -> переименование
        # создало бы дубль (direction, name, year). Пропускаем.
        return PLAN_SKIP
    if inner in official_names:
        return PLAN_RENAME
    return PLAN_SKIP


# ─── loading ────────────────────────────────────────────────────────────────

def krm_path_for(direction: str, year: int) -> Path:
    if year == DEFAULT_YEAR:
        return REFERENCE_DIR / f"krm_disciplines_{direction}.json"
    return REFERENCE_DIR / f"krm_disciplines_{direction}_{year}.json"


def official_path_for(direction: str, year: int) -> Path:
    return REFERENCE_DIR / f"official_curriculum_{direction}_{year}.json"


def load_official_names(direction: str, year: int) -> set[str]:
    path = official_path_for(direction, year)
    if not path.exists():
        raise SystemExit(f"нет файла официального перечня: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    disk = doc.get("disciplines", [])
    if not isinstance(disk, list):
        raise SystemExit(f"{path.name}: поле 'disciplines' не список")
    return {d["name"] if isinstance(d, dict) else d for d in disk if d}


def load_krm(direction: str, year: int) -> tuple[Path, dict, dict]:
    path = krm_path_for(direction, year)
    if not path.exists():
        raise SystemExit(f"нет KRM-файла: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise SystemExit(f"{path.name}: корень не объект")
    node = doc.get(direction)
    if not isinstance(node, dict):
        raise SystemExit(f"{path.name}: нет узла {direction!r}")
    disciplines = node.setdefault("disciplines", {})
    if not isinstance(disciplines, dict):
        raise SystemExit(f"{path.name}: 'disciplines' не объект")
    return path, doc, disciplines


# ─── DB operations ──────────────────────────────────────────────────────────

def load_db_state(conn, direction_code: str, year: int) -> list[dict]:
    """Вернуть [{direction_id, disciplines: {name: id}}] для всех direction по коду."""
    dir_ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM directions WHERE code = %s ORDER BY id",
            (direction_code,),
        ).fetchall()
    ]
    state = []
    for dir_id in dir_ids:
        rows = conn.execute(
            "SELECT id, name FROM disciplines WHERE direction_id = %s AND year = %s ORDER BY name",
            (dir_id, year),
        ).fetchall()
        state.append({"direction_id": str(dir_id), "disciplines": {name: str(did) for did, name in rows}})
    return state


def _move_content(conn, junk_comp_id: str, target_comp_id: str) -> tuple[int, int]:
    """Перенести competency_skills и ksa_entries из мусорной компетенции в целевую.

    Дубль (по уникальным индексам) удаляем, а не переносим. Возвращает
    (перенесено_cs, перенесено_ksa).
    """
    moved_cs = moved_ksa = 0

    junk_cs = conn.execute(
        "SELECT id, skill_id, ksa_type, parse_version_id FROM competency_skills WHERE competency_id = %s",
        (junk_comp_id,),
    ).fetchall()
    for cs_id, skill_id, ksa_type, pv in junk_cs:
        dup = conn.execute(
            """SELECT 1 FROM competency_skills
               WHERE competency_id = %s AND skill_id = %s AND ksa_type = %s
                 AND COALESCE(parse_version_id, %s::uuid) = COALESCE(%s, %s::uuid)
               LIMIT 1""",
            (target_comp_id, skill_id, ksa_type, ZERO_UUID, pv, ZERO_UUID),
        ).fetchone()
        if dup:
            conn.execute("DELETE FROM competency_skills WHERE id = %s", (cs_id,))
        else:
            conn.execute("UPDATE competency_skills SET competency_id = %s WHERE id = %s", (target_comp_id, cs_id))
            moved_cs += 1

    junk_ksa = conn.execute(
        "SELECT id, ksa_type::text, original_text, source FROM ksa_entries WHERE competency_id = %s",
        (junk_comp_id,),
    ).fetchall()
    for k_id, ksa_type, orig, src in junk_ksa:
        dup = conn.execute(
            "SELECT 1 FROM ksa_entries WHERE competency_id = %s AND ksa_type = %s AND original_text = %s AND source = %s LIMIT 1",
            (target_comp_id, ksa_type, orig, src),
        ).fetchone()
        if dup:
            conn.execute("DELETE FROM ksa_entries WHERE id = %s", (k_id,))
        else:
            conn.execute("UPDATE ksa_entries SET competency_id = %s WHERE id = %s", (target_comp_id, k_id))
            moved_ksa += 1

    return moved_cs, moved_ksa


def db_merge(conn, junk_disc_id: str, target_disc_id: str, junk_name: str, target_name: str) -> dict:
    """Слить мусорную дисциплину в целевую. Возвращает отчёт об операции."""
    junk_comps = conn.execute(
        "SELECT id, code, parent_id FROM competencies WHERE discipline_id = %s ORDER BY code",
        (junk_disc_id,),
    ).fetchall()
    target_comps = {
        code: str(cid)
        for cid, code in conn.execute(
            "SELECT id, code FROM competencies WHERE discipline_id = %s", (target_disc_id,)
        ).fetchall()
    }

    moved = []
    for jid, code, _parent in junk_comps:
        jid = str(jid)
        tcid = target_comps.get(code)
        if tcid is not None:
            # Код уже есть в цели: переносим содержимое, затем удаляем мусорную компетенцию.
            moved_cs, moved_ksa = _move_content(conn, jid, tcid)
            # Перевешиваем дочерние компетенции (parent_id) на целевую до удаления.
            conn.execute("UPDATE competencies SET parent_id = %s WHERE parent_id = %s", (tcid, jid))
            conn.execute("DELETE FROM competencies WHERE id = %s", (jid,))
            moved.append(
                {
                    "code": code,
                    "junk_competency_id": jid,
                    "target_competency_id": tcid,
                    "moved_competency_skills": moved_cs,
                    "moved_ksa_entries": moved_ksa,
                    "reassigned": False,
                }
            )
        else:
            # Кода в цели нет: переносим компетенцию целиком (просто смена discipline_id).
            conn.execute("UPDATE competencies SET discipline_id = %s WHERE id = %s", (target_disc_id, jid))
            moved.append(
                {
                    "code": code,
                    "junk_competency_id": jid,
                    "target_competency_id": None,
                    "moved_competency_skills": None,
                    "moved_ksa_entries": None,
                    "reassigned": True,
                }
            )

    # Сопутствующие нулевые анализы покрытия удалятся каскадом; фиксируем их количество.
    cascaded = conn.execute(
        "SELECT count(*) FROM coverage_analyses WHERE discipline_id = %s", (junk_disc_id,)
    ).fetchone()[0]

    conn.execute("DELETE FROM disciplines WHERE id = %s", (junk_disc_id,))
    return {
        "store": "db",
        "action": PLAN_MERGE,
        "junk": {"discipline_id": junk_disc_id, "name": junk_name},
        "target": {"discipline_id": target_disc_id, "name": target_name},
        "moved_competencies": moved,
        "cascade_deleted_coverage_analyses": cascaded,
    }


def db_rename(conn, junk_disc_id: str, new_name: str) -> None:
    conn.execute("UPDATE disciplines SET name = %s WHERE id = %s", (new_name, junk_disc_id))


# ─── main ───────────────────────────────────────────────────────────────────

def build_plan(direction: str, year: int, db_state: list[dict], krm_disciplines: dict,
               official_names: set[str]) -> dict:
    """Собрать план операций для DB и KRM (без записи)."""
    plan: dict[str, list[dict]] = {"db": [], "krm": []}

    # KRM: один словарь на направление.
    krm_names = set(krm_disciplines.keys())
    for junk_key in list(krm_disciplines.keys()):
        inner = detect_bilingual_junk(junk_key)
        if inner is None:
            continue
        merge_target = inner if inner in krm_names else None
        action = plan_action(inner, existing_names=krm_names, official_names=official_names,
                             merge_target=merge_target)
        plan["krm"].append({"junk_name": junk_key, "inner": inner, "action": action})

    # DB: по каждой direction-группе отдельно (уникальность на direction_id).
    for group in db_state:
        names = set(group["disciplines"].keys())
        for junk_name, junk_id in group["disciplines"].items():
            inner = detect_bilingual_junk(junk_name)
            if inner is None:
                continue
            merge_target = inner if inner in names else None
            action = plan_action(inner, existing_names=names, official_names=official_names,
                                 merge_target=merge_target)
            plan["db"].append(
                {
                    "junk_name": junk_name,
                    "junk_id": junk_id,
                    "inner": inner,
                    "action": action,
                    "target_id": group["disciplines"].get(inner),
                }
            )
    return plan


def print_plan(direction: str, year: int, plan: dict) -> None:
    print(f"=== plan {direction} / {year} ===")
    for entry in plan["krm"]:
        print(f"  [krm] {entry['action']:<6} {entry['junk_name']!r} -> {entry['inner']!r}")
    for entry in plan["db"]:
        print(f"  [db ] {entry['action']:<6} {entry['junk_name']!r} -> {entry['inner']!r}")
    if not plan["db"] and not plan["krm"]:
        print("  (мусорных имён не найдено)")


def main() -> int:
    _reconfigure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="09.03.02")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--apply", action="store_true", help="записать изменения (по умолчанию dry-run)")
    parser.add_argument("--dsn", default="")
    args = parser.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)

    official_names = load_official_names(args.direction, args.year)
    krm_path, krm_doc, krm_disciplines = load_krm(args.direction, args.year)

    with psycopg.connect(args.dsn, connect_timeout=15) as conn:
        db_state = load_db_state(conn, args.direction, args.year)

    plan = build_plan(args.direction, args.year, db_state, krm_disciplines, official_names)
    print_plan(args.direction, args.year, plan)

    if not args.apply:
        print("\ndry-run: изменений нет (для записи добавьте --apply)")
        return 0

    # ── запись ──────────────────────────────────────────────────────────────
    backup = krm_path.with_name(krm_path.name[:-len(".json")] + ".backup.json")
    shutil.copy2(krm_path, backup)
    print(f"\nбэкап KRM: {backup}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "direction": args.direction,
        "year": args.year,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_operations": [],
        "krm_operations": [],
    }

    # DB в одной транзакции.
    with psycopg.connect(args.dsn, connect_timeout=15) as conn:
        with conn.transaction():
            for entry in plan["db"]:
                if entry["action"] == PLAN_MERGE and entry["target_id"]:
                    op = db_merge(conn, entry["junk_id"], entry["target_id"],
                                  entry["junk_name"], entry["inner"])
                    report["db_operations"].append(op)
                    print(
                        f"DB  merge  {entry['junk_name']!r} -> {entry['inner']!r}  "
                        f"(перенесено компетенций: {len(op['moved_competencies'])})"
                    )
                    for mc in op["moved_competencies"]:
                        if mc["reassigned"]:
                            print(f"     компетенция {mc['code']!r} перенесена целиком (смена discipline_id)")
                        else:
                            print(
                                f"     компетенция {mc['code']!r}: "
                                f"{mc['moved_competency_skills']} competency_skills, "
                                f"{mc['moved_ksa_entries']} ksa перепривязаны"
                            )
                    if op["cascade_deleted_coverage_analyses"]:
                        print(f"     каскадно удалено coverage_analyses: {op['cascade_deleted_coverage_analyses']}")
                    print(f"     удалена дисциплина {entry['junk_id']} {entry['junk_name']!r}")
                elif entry["action"] == PLAN_RENAME:
                    db_rename(conn, entry["junk_id"], entry["inner"])
                    report["db_operations"].append(
                        {
                            "store": "db",
                            "action": PLAN_RENAME,
                            "renamed": {"discipline_id": entry["junk_id"],
                                        "from": entry["junk_name"], "to": entry["inner"]},
                        }
                    )
                    print(f"DB  rename {entry['junk_name']!r} -> {entry['inner']!r}")
                else:
                    print(f"DB  skip   {entry['junk_name']!r} (столкновение/нет цели)")

    # KRM после DB.
    for entry in plan["krm"]:
        junk_key, inner, action = entry["junk_name"], entry["inner"], entry["action"]
        if action == PLAN_MERGE and inner in krm_disciplines:
            merge_entries(krm_disciplines[inner], krm_disciplines.pop(junk_key))
            report["krm_operations"].append(
                {"store": "krm", "action": PLAN_MERGE, "junk_key": junk_key, "target_key": inner}
            )
            print(f"KRM merge  {junk_key!r} -> {inner!r} (ключ мусора удалён)")
        elif action == PLAN_RENAME:
            krm_disciplines[inner] = krm_disciplines.pop(junk_key)
            report["krm_operations"].append(
                {"store": "krm", "action": PLAN_RENAME, "from": junk_key, "to": inner}
            )
            print(f"KRM rename {junk_key!r} -> {inner!r}")
        else:
            print(f"KRM skip   {junk_key!r}")

    # Отсортировать ключи дисциплин и записать KRM.
    krm_node = krm_doc[args.direction]
    krm_node["disciplines"] = dict(sorted(krm_disciplines.items()))
    krm_path.write_text(json.dumps(krm_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dump_path = RESULT_DIR / f"{args.direction}_{args.year}_{stamp}.json"
    dump_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSQL-дамп затронутых строк: {dump_path}")
    print(f"KRM записан: {krm_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
