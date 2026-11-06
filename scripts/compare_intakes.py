"""Сравнение двух лет набора: что изменилось в покрытии дисциплин.

Берёт две готовые сводки анализа (`data/result/teacher/<код>[_/<год>]/_summary.json`),
сопоставляет дисциплины той же нормализацией имён, что и аудит
(`scripts/audit_curriculum_coverage.py`: аббревиатуры ИИ/ПО/МО/ИС, скобочные
латинско-русские дубли, омоглифы), и отдаёт структурированный отчёт:

    added   - дисциплины, появившиеся в новом наборе;
    removed - исчезнувшие из старого;
    kept    - общие, с дельтой покрытия (новое - старое) и флагом значимости.

    python scripts/compare_intakes.py --direction 09.03.02 --old 2024 --new 2025
    python scripts/compare_intakes.py --direction 09.03.02 --old 2024 --new 2025 --json out.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "data" / "result" / "teacher"

# Значимым считаем изменение покрытия от 5 п.п. (в долях единицы).
COVERAGE_DELTA_THRESHOLD = 0.05

_audit_module: Any = None


def _load_audit() -> Any:
    """Импортирует хелперы нормализации из аудита, не дублируя их схему.

    Скрипт - не пакет, поэтому грузим через importlib (как `curriculum_years.py`).
    Модуль аудита при импорте выполняет только `import psycopg` и определения
    функций, побочных эффектов нет.

    Аудит импортирует `_dsn` как модуль верхнего уровня (он рассчитан на запуск
    из своей директории), поэтому на время exec_module кладём `scripts/` первым
    в sys.path. Иначе под uvicorn `from _dsn import ...` падает, а запасной
    `from scripts._dsn import ...` перехватывает чужой пакет `scripts` из
    site-packages - и /teacher/intake-comparison отвечает 500.
    """
    scripts_dir = str(ROOT / "scripts")
    with contextlib.suppress(ValueError):
        sys.path.remove(scripts_dir)
    sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "audit_curriculum_coverage", ROOT / "scripts" / "audit_curriculum_coverage.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(scripts_dir)
    return module


def _audit() -> Any:
    global _audit_module
    if _audit_module is None:
        _audit_module = _load_audit()
    return _audit_module


def norm(name: str) -> str:
    """Нормализация имени дисциплины (реэкспорт из аудита)."""
    return str(_audit().norm(name))


def strip_parens(name: str) -> str:
    """Отбрасывает латинский дубль 'Databases and DBMS (Базы данных и СУБД)'."""
    return str(_audit().strip_parens(name))


def canonical_key(name: str) -> str:
    """Ключ сопоставления: имя без скобочного латинского дубля, нормализованное."""
    return norm(strip_parens(name))


class IntakeNotFoundError(Exception):
    """Сводка по требуемому году набора отсутствует - сравнивать нечего."""

    def __init__(self, label: str, year: int, direction: str) -> None:
        self.label = label
        self.year = year
        self.direction = direction
        super().__init__(
            f"нет сводки по {label} набору ({year}) направления {direction}: "
            f"нет файла {RESULT_DIR / direction / str(year)}/_summary.json "
            f"и {RESULT_DIR / direction}/_summary.json"
        )


def summary_path(direction: str, year: int, base: Path = RESULT_DIR) -> Path | None:
    """Путь к сводке года. Текущий год лежит в корне (`_summary.json`), прежние -
    в подкаталоге `<год>/_summary.json`. Сверяем `year` внутри файла, а не только
    имя каталога, чтобы не спутать наборы.
    """
    subdir = base / direction / str(year) / "_summary.json"
    if subdir.exists():
        return subdir
    root = base / direction / "_summary.json"
    if root.exists():
        try:
            doc = json.loads(root.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if doc.get("year") == year:
            return root
    return None


def load_summary(direction: str, year: int, label: str, base: Path = RESULT_DIR) -> dict[str, Any]:
    path = summary_path(direction, year, base)
    if path is None:
        raise IntakeNotFoundError(label, year, direction)
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _dedup(disciplines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Схлопывает дубли одной дисциплины, оставляя запись с лучшим покрытием.

    Сводка 2024 года содержит сразу 'Algorithmization and Programming
    (Алгоритмизация и программирование)' (заглушка, coverage_ratio=0) и
    'Алгоритмизация и программирование' (реальный анализ) - обе нормализуются
    в один ключ. Coverage=0 - это алиас без анализа, поэтому при коллизии
    предпочитаем запись с большим покрытием.
    """
    index: dict[str, dict[str, Any]] = {}
    for disc in disciplines:
        key = canonical_key(str(disc.get("name") or ""))
        current = index.get(key)
        if current is None or _num(disc.get("coverage_ratio")) > _num(current.get("coverage_ratio")):
            index[key] = disc
    return list(index.values())


def _mean_coverage(disciplines: list[dict[str, Any]]) -> float:
    if not disciplines:
        return 0.0
    return sum(_num(d.get("coverage_ratio")) for d in disciplines) / len(disciplines)


def compare_intakes(
    old_disciplines: list[dict[str, Any]],
    new_disciplines: list[dict[str, Any]],
    avg_old: float | None = None,
    avg_new: float | None = None,
) -> dict[str, Any]:
    """Сопоставляет два набора дисциплин и строит отчёт о различиях.

    Аргументы - списки словарей `{name, coverage_ratio, gaps, coverage_level, ...}`
    (как `disciplines` в `_summary.json`). `avg_old`/`avg_new` переопределяют
    среднее покрытие набора (по умолчанию считается как среднее по спискам).
    """
    old = _dedup(old_disciplines)
    new = _dedup(new_disciplines)
    audit = _audit()

    old_names = [str(d.get("name") or "") for d in old]
    new_names = [str(d.get("name") or "") for d in new]
    # Сверхку латинский дубль живёт в старом наборе, поэтому `official=old`:
    # strip_parens применяется к нужной стороне (как в аудите).
    pairs, missing, extra = audit.match(old_names, new_names)

    old_by_name = {str(d.get("name") or ""): d for d in old}
    new_by_name = {str(d.get("name") or ""): d for d in new}

    def _card(name: str, disc: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": name,
            "coverage_ratio": disc.get("coverage_ratio"),
            "coverage_level": disc.get("coverage_level"),
            "gaps": disc.get("gaps"),
        }

    added = [_card(n, new_by_name[n]) for n in extra]
    removed = [_card(n, old_by_name[n]) for n in missing]

    kept: list[dict[str, Any]] = []
    improved = 0
    worsened = 0
    unchanged = 0
    for old_name, new_name in pairs.items():
        old_disc = old_by_name[old_name]
        new_disc = new_by_name[new_name]
        cov_old = _num(old_disc.get("coverage_ratio"))
        cov_new = _num(new_disc.get("coverage_ratio"))
        delta = cov_new - cov_old
        gaps_old = _int(old_disc.get("gaps"))
        gaps_new = _int(new_disc.get("gaps"))
        if delta > 0:
            improved += 1
        elif delta < 0:
            worsened += 1
        else:
            unchanged += 1
        kept.append({
            "name": new_name,
            "old_name": old_name if old_name != new_name else None,
            "coverage_ratio_old": round(cov_old, 4),
            "coverage_ratio_new": round(cov_new, 4),
            "coverage_delta": round(delta, 4),
            "gaps_old": gaps_old,
            "gaps_new": gaps_new,
            "gaps_delta": gaps_new - gaps_old,
            "significant_change": abs(delta) >= COVERAGE_DELTA_THRESHOLD,
        })

    if avg_old is None:
        avg_old = _mean_coverage(old)
    if avg_new is None:
        avg_new = _mean_coverage(new)

    return {
        "added": added,
        "removed": removed,
        "kept": kept,
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "kept": len(kept),
            "avg_coverage_old": round(avg_old, 4),
            "avg_coverage_new": round(avg_new, 4),
            "avg_coverage_delta": round(avg_new - avg_old, 4),
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
        },
    }


def build_intake_report(
    direction: str,
    old_year: int,
    new_year: int,
    base: Path = RESULT_DIR,
) -> dict[str, Any]:
    """Загружает сводки двух лет и собирает полный отчёт сравнения."""
    old_doc = load_summary(direction, old_year, "старый", base)
    new_doc = load_summary(direction, new_year, "новый", base)

    report = compare_intakes(
        list(old_doc.get("disciplines", [])),
        list(new_doc.get("disciplines", [])),
        avg_old=_num(old_doc.get("average_coverage")),
        avg_new=_num(new_doc.get("average_coverage")),
    )
    report["direction"] = direction
    report["old_year"] = old_year
    report["new_year"] = new_year
    report["direction_name"] = new_doc.get("direction_name")
    return report


def _fmt_ratio(value: Any) -> str:
    return f"{value:.0%}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "—"


def _print_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    name = report.get("direction_name") or report.get("direction")
    print(f"{name} ({report['direction']})")
    print(f"Сравнение наборов: {report['old_year']} -> {report['new_year']}\n")
    print(f"Итог: добавлено {s['added']}, убрано {s['removed']}, сохранено {s['kept']}")
    print(
        f"Среднее покрытие: {s['avg_coverage_old']:.4f} ({report['old_year']}) "
        f"-> {s['avg_coverage_new']:.4f} ({report['new_year']}), "
        f"дельта {s['avg_coverage_delta']:+.4f}"
    )
    print(
        f"Рост покрытия: {s['improved']}, падение: {s['worsened']}, "
        f"без изменений: {s['unchanged']}"
    )

    print(f"\n--- Добавлены в {report['new_year']} ({s['added']}) ---")
    for d in report["added"]:
        print(f"  · {d['name']} (покрытие {_fmt_ratio(d.get('coverage_ratio'))})")

    print(f"\n--- Убраны из плана ({s['removed']}) ---")
    for d in report["removed"]:
        print(f"  · {d['name']} (покрытие {_fmt_ratio(d.get('coverage_ratio'))})")

    print(f"\n--- Изменилось покрытие ({s['kept']}) ---")
    for d in report["kept"]:
        flag = " [!]" if d.get("significant_change") else ""
        print(
            f"  · {d['name']}: {d['coverage_ratio_old']:.0%} -> {d['coverage_ratio_new']:.0%} "
            f"(дельта {d['coverage_delta']:+.4f}){flag}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="09.03.02")
    parser.add_argument("--old", type=int, default=2024, help="год старого набора")
    parser.add_argument("--new", type=int, default=2025, help="год нового набора")
    parser.add_argument("--json", dest="json_path", default=None, help="сохранить отчёт в JSON")
    args = parser.parse_args(argv)

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        report = build_intake_report(args.direction, args.old, args.new)
    except IntakeNotFoundError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2

    _print_report(report)
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nотчёт: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
