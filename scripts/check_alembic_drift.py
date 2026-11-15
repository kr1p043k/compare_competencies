"""Гейт дрейфа alembic (#6.3.2): ORM-модели <-> схема миграций на чистой БД.

По смыслу это `alembic check` (тот же код autogenerate, только diffs забираются
структурно, а не из текста исключения). Прогон на пустой базе после
`alembic upgrade heads` показывает не только свежие расхождения, но и легаси:
миграции создают таблицы/индексы/колонки (auth-сервисы, _totp, app_settings,
кэши), которых уже нет в Base.metadata, поэтому голый `alembic check` красный
всегда и ничего нового не ловит.

Скрипт превращает diff-операции в стабильные сигнатуры
(`kind:table[.column|:index|:constraint]`) и сверяет с закоммиченным baseline
(scripts/alembic_drift_baseline.json) - ratchet в одну строгую сторону:

- сигнатур больше baseline -> exit 1: новый дрейф, CI падает;
- сигнатур меньше baseline -> предупреждение, exit 0: легаси починили, baseline
  стоит пересобрать, но гейт на этом не краснеет;
- набор совпал -> exit 0.

Любой сбой autogenerate (нет DATABASE_URL/SECRET_KEY, база недоступна) или
неизвестный тип diff-операции -> exit 2 (fail-closed): молча проглотить
проверку хуже, чем упасть.

Запуск (то же окружение, что у job migrations в CI):
    DATABASE_URL=... SECRET_KEY=... python scripts/check_alembic_drift.py
Пересборка baseline (на свежей базе после `python -m alembic upgrade heads`):
    python scripts/check_alembic_drift.py --update-baseline

В CI шаг идёт сразу после "Upgrade from scratch" и использует service-контейнер
postgres того же job; локально нужна живая база с pgvector.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # env.py alembic импортирует src.* при прогоне
if hasattr(sys.stdout, "reconfigure"):  # cp1251-консоль Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASELINE_PATH = ROOT / "scripts" / "alembic_drift_baseline.json"

KNOWN_KINDS: frozenset[str] = frozenset(
    {
        "add_table",
        "remove_table",
        "add_column",
        "remove_column",
        "modify_type",
        "modify_nullable",
        "modify_server_default",
        "modify_comment",
        "add_index",
        "remove_index",
        "add_constraint",
        "remove_constraint",
        "add_fk",
        "remove_fk",
    }
)


class DriftConfigError(RuntimeError):
    """Не удалось получить diffs: нет DATABASE_URL/SECRET_KEY, база недоступна."""


class DriftParseError(RuntimeError):
    """Формат diffs не распознан - гейт не должен молчать."""


class Comparison(NamedTuple):
    new: dict[str, int]
    resolved: dict[str, int]
    total: int


def _flatten(items: list[Any]) -> Iterator[Any]:
    """autogenerate кладёт modify_*/nullable-операции во вложенные списки."""
    for element in items:
        if isinstance(element, list):
            yield from _flatten(element)
        else:
            yield element


def _constraint_signature(kind: str, obj: Any) -> str:
    table = obj.table.name
    name = getattr(obj, "name", None)
    if not name:
        cols = ",".join(sorted(col.name for col in obj.columns))
        return f"{kind}:{table}[{cols}]"
    return f"{kind}:{table}:{name}"


def signature_for(diff: Any) -> str:
    """Стабильная текстовая сигнатура одной diff-операции alembic autogenerate."""
    if not isinstance(diff, (tuple, list)) or not diff:
        raise DriftParseError(f"Неожиданный элемент diffs: {diff!r}")
    kind = diff[0]
    if kind not in KNOWN_KINDS:
        raise DriftParseError(
            f"Неизвестная diff-операция '{kind}'. Alembic научился новому "
            "дрейфу - расширьте signature_for в scripts/check_alembic_drift.py."
        )
    if kind in {"add_table", "remove_table"}:
        return f"{kind}:{diff[1].name}"
    if kind in {"add_column", "remove_column"}:
        # (kind, schema, table_name, Column)
        return f"{kind}:{diff[2]}.{diff[3].name}"
    if kind.startswith("modify_"):
        # (kind, schema, table_name, column_name, opts, old, new)
        return f"{kind}:{diff[2]}.{diff[3]}"
    if kind in {"add_index", "remove_index"}:
        return f"{kind}:{diff[1].table.name}:{diff[1].name}"
    if kind in {"add_constraint", "remove_constraint"}:
        return _constraint_signature(kind, diff[1])
    # add_fk / remove_fk: у FK из моделей имя может отсутствовать (name=None),
    # тогда сигнатура собирается по локальным колонкам (они стабильны).
    fk = diff[1]
    table = fk.table.name
    name = getattr(fk, "name", None)
    if not name:
        cols = ",".join(sorted(col.name for col in fk.columns))
        return f"{kind}:{table}[{cols}]"
    return f"{kind}:{table}:{name}"


def collect_signatures(database_url: str | None) -> Counter[str]:
    """Те же шаги, что делает `alembic check`, но diffs берутся структурно."""
    if database_url:
        os.environ["DATABASE_URL"] = database_url
    if not os.environ.get("DATABASE_URL"):
        raise DriftConfigError(
            "DATABASE_URL не задан. Экспортируйте его (или передайте "
            "--database-url); alembic/env.py строит engine из src.config."
        )

    from alembic.script import ScriptDirectory

    from alembic import autogenerate, environment
    from alembic import config as alembic_config

    cfg = alembic_config.Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    script = ScriptDirectory.from_config(cfg)
    command_args: dict[str, Any] = dict(
        message=None,
        autogenerate=True,
        sql=False,
        head="head",
        splice=False,
        branch_label=None,
        version_path=None,
        rev_id=None,
        depends_on=None,
    )
    revision_context = autogenerate.RevisionContext(cfg, script, command_args)

    def retrieve_migrations(rev: Any, context: Any) -> list[Any]:
        revision_context.run_autogenerate(rev, context)
        return []

    try:
        with environment.EnvironmentContext(
            cfg,
            script,
            fn=retrieve_migrations,
            as_sql=False,
            template_args=revision_context.template_args,
            revision_context=revision_context,
        ):
            script.run_env()
    except Exception as exc:
        # env.py поднимает что угодно: нет БД, нет SECRET_KEY, битый DSN.
        # Для гейта это одинаково "проверка не состоялась" -> exit 2.
        detail = str(exc).strip() or exc.__class__.__name__
        raise DriftConfigError(
            f"autogenerate не выполнен ({exc.__class__.__name__}): {detail[:300]}. "
            "Проверьте доступность БД и SECRET_KEY в окружении."
        ) from exc

    diffs: list[Any] = []
    for upgrade_ops in revision_context.generated_revisions[-1].upgrade_ops_list:
        diffs.extend(upgrade_ops.as_diffs())

    return Counter(signature_for(diff) for diff in _flatten(diffs))


def load_baseline(path: Path) -> Counter[str]:
    if not path.exists():
        raise DriftConfigError(
            f"Baseline не найден: {path}. Соберите его: "
            "python scripts/check_alembic_drift.py --update-baseline"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriftConfigError(f"Baseline повреждён ({path}): {exc}") from exc
    raw = data.get("signatures")
    if not isinstance(raw, dict):
        raise DriftConfigError(f"В baseline нет словаря 'signatures': {path}")
    return Counter({str(key): int(count) for key, count in raw.items()})


def compare(current: Counter[str], baseline: Counter[str]) -> Comparison:
    new = {sig: count - baseline.get(sig, 0) for sig, count in current.items() if count > baseline.get(sig, 0)}
    resolved = {
        sig: base_count - current.get(sig, 0)
        for sig, base_count in baseline.items()
        if current.get(sig, 0) < base_count
    }
    return Comparison(new=new, resolved=resolved, total=sum(current.values()))


def alembic_head() -> str:
    """HEAD ревизия для шапки baseline (без подключения к БД)."""
    proc = subprocess.run(
        [sys.executable, "-m", "alembic", "heads"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if proc.returncode != 0:
        return "unknown"
    lines = (proc.stdout or "").strip().splitlines()
    return lines[0].split(" ")[0] if lines else "unknown"


def write_baseline(path: Path, signatures: Counter[str]) -> None:
    payload = {
        "generated_by": "python scripts/check_alembic_drift.py --update-baseline",
        "alembic_head": alembic_head(),
        "note": (
            "Легаси-набор diffs alembic-check на чистой базе после upgrade heads. "
            "CI падает на новых сигнатурах сверх этого списка; исчезнувшие - "
            "только предупреждение (пересобрать baseline)."
        ),
        "signatures": dict(sorted(signatures.items())),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alembic models<->DB drift gate with ratchet baseline (#6.3.2)")
    parser.add_argument("--database-url", default=None, help="DATABASE_URL для autogenerate-проверки")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="перезаписать baseline текущим набором diffs (на свежей базе после upgrade heads)",
    )
    parser.add_argument("--show-signatures", action="store_true", help="напечатать текущие сигнатуры")
    args = parser.parse_args(argv)

    try:
        current = collect_signatures(args.database_url)
    except (DriftConfigError, DriftParseError) as exc:
        print(f"[alembic-drift] ОШИБКА: {exc}")
        return 2

    if args.show_signatures:
        for sig, count in sorted(current.items()):
            print(f"{sig} x{count}" if count > 1 else sig)

    if args.update_baseline:
        write_baseline(BASELINE_PATH, current)
        print(f"[alembic-drift] baseline перезаписан: {BASELINE_PATH} ({sum(current.values())} ops)")
        return 0

    try:
        baseline = load_baseline(BASELINE_PATH)
    except DriftConfigError as exc:
        print(f"[alembic-drift] ОШИБКА: {exc}")
        return 2

    result = compare(current, baseline)

    if result.new:
        print(f"[alembic-drift] РЕАЛЬНЫЙ ДРЕЙФ: {len(result.new)} новых операций сверх baseline:")
        for sig, count in sorted(result.new.items())[:60]:
            print(f"  + {sig}" + (f" (x{count})" if count > 1 else ""))
        print(
            "Расхождение моделей и миграций, которого не было в baseline.\n"
            "Заведите ревизию `alembic revision --autogenerate` и/или синхронизируйте\n"
            "модели; если удаление из моделей осознанное - пересоберите baseline\n"
            "(`--update-baseline`) и закоммитьте его."
        )
        return 1

    if result.resolved:
        print(f"[alembic-drift] WARNING: {len(result.resolved)} операций baseline больше не дрейфуют:")
        for sig, count in sorted(result.resolved.items())[:20]:
            print(f"  - {sig}" + (f" (x{count})" if count > 1 else ""))
        print("Легаси починили: пересоберите baseline (--update-baseline) и закоммитьте его.")

    print(f"[alembic-drift] OK: дрейф не превышает baseline ({result.total} ops против {sum(baseline.values())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
