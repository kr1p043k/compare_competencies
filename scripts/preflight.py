"""Проверка готовности перед релизом.

Запускается вручную или из CI перед деплоем. Ничего не меняет: только читает
конфигурацию, состояние БД и файловой системы и печатает отчёт.
Код возврата 1, если есть блокирующие проблемы.

Использование:
    python scripts/preflight.py            # локально
    python scripts/preflight.py --strict   # в CI: предупреждения тоже блокируют
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OK = "OK  "
WARN = "WARN"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []


def check(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


def check_env_file() -> None:
    env = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env.exists():
        check(FAIL, ".env", "файл отсутствует — скопируйте .env.example и заполните")
        return
    check(OK, ".env", "найден")

    if not example.exists():
        check(WARN, ".env.example", "нет шаблона для новых разработчиков")
        return

    def keys(path: Path) -> set[str]:
        out = set()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                out.add(line.split("=", 1)[0].strip())
        return out

    env_keys, example_keys = keys(env), keys(example)
    missing = sorted(example_keys - env_keys)
    extra = sorted(env_keys - example_keys)
    if missing:
        check(WARN, ".env", f"ключей из шаблона нет в .env: {', '.join(missing[:6])}")
    else:
        check(OK, ".env", "все ключи шаблона заданы")
    if extra:
        check(WARN, ".env.example", f"в .env есть ключи вне шаблона: {', '.join(extra[:6])}")


def check_secret() -> None:
    """Ключ подписи токенов: длина и отсутствие заглушек."""
    try:
        from src import config

        secret = getattr(config.settings, "SECRET_KEY", None)
        key = secret.get_secret_value() if secret is not None else ""
    except Exception as exc:  # noqa: BLE001 — проверка не должна падать сама
        check(WARN, "SECRET_KEY", f"не удалось прочитать конфиг: {type(exc).__name__}")
        return

    if not key:
        if (ROOT / ".env").exists():
            check(FAIL, "SECRET_KEY", "пустой при наличии .env — задайте openssl rand -hex 32")
        else:
            check(FAIL, "SECRET_KEY", "не задан и нет .env")
        return

    placeholders = {
        "change-me", "changeme", "change_me", "secret",
        "change-me-to-random-string", "change_me_please",
    }
    if key.lower() in placeholders:
        check(FAIL, "SECRET_KEY", "заглушка из примера конфигурации")
    elif len(key) < 32:
        check(FAIL, "SECRET_KEY", f"короче 32 символов (сейчас {len(key)})")
    else:
        check(OK, "SECRET_KEY", f"длина {len(key)}")


def check_cors() -> None:
    try:
        from src import config

        origins = config.ALLOWED_ORIGINS
    except Exception:  # noqa: BLE001
        check(WARN, "ALLOWED_ORIGINS", "не удалось прочитать конфиг")
        return
    if origins.strip() == "*":
        check(FAIL, "ALLOWED_ORIGINS", "wildcard: в проде сервер не стартует (гард в create_app)")
    else:
        check(OK, "ALLOWED_ORIGINS", f"{len(origins.split(','))} домен(ов)")


def check_ports() -> None:
    """Публикация инфраструктурных портов наружу — риск, а не блокер."""
    compose = ROOT / "docker-compose.yml"
    if not compose.exists():
        check(WARN, "docker-compose.yml", "файл не найден")
        return
    text = compose.read_text(encoding="utf-8", errors="ignore")
    risky = []
    for port in ("5432:5432", "5678:5678", "9090:9090", "3001:3000"):
        if f'"{port}"' in text:
            risky.append(port.split(":")[0])
    if risky:
        check(WARN, "compose ports", f"опубликованы на 0.0.0.0: {', '.join(risky)}")
    else:
        check(OK, "compose ports", "инфраструктурные порты не опубликованы")


def check_migrations() -> None:
    """Одна голова и непустой downgrade — самое частое место поломки деплоя."""
    import ast

    versions = ROOT / "alembic" / "versions"
    if not versions.exists():
        check(WARN, "alembic", "каталог versions не найден")
        return

    heads: list[str] = []
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        cfg = Config(str(ROOT / "alembic.ini")) if (ROOT / "alembic.ini").exists() else Config()
        cfg.set_main_option("script_location", str(ROOT / "alembic"))
        heads = list(ScriptDirectory.from_config(cfg).get_heads())
    except Exception as exc:  # noqa: BLE001 — проверка не должна падать сама
        check(WARN, "alembic heads", f"не удалось разобрать миграции: {type(exc).__name__}")
    else:
        if len(heads) == 1:
            check(OK, "alembic heads", f"одна голова: {heads[0]}")
        elif not heads:
            check(FAIL, "alembic heads", "не найдено ни одной головы")
        else:
            check(FAIL, "alembic heads", f"несколько голов: {', '.join(heads)}")

    empty = []
    for path in versions.glob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            empty.append(path.name + " (не парсится)")
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "downgrade":
                body = [n for n in node.body if not isinstance(n, ast.Expr)
                        or not isinstance(getattr(n, "value", None), ast.Constant)]
                if not body or all(isinstance(n, ast.Pass) for n in body):
                    empty.append(path.name)

    if empty:
        check(WARN, "alembic downgrade", f"пустой откат: {', '.join(empty[:4])}")
    else:
        check(OK, "alembic downgrade", "у всех миграций есть откат")


def check_disk() -> None:
    total, used, free = shutil.disk_usage(ROOT)
    pct = free / total * 100
    status = OK if pct >= 15 else WARN if pct >= 7 else FAIL
    check(status, "диск", f"свободно {free / 1e9:.1f} GB ({pct:.0f}%)")


def check_frontend() -> None:
    pkg = ROOT / "frontend" / "package.json"
    if not pkg.exists():
        check(WARN, "frontend", "package.json не найден")
        return
    node_modules = ROOT / "frontend" / "node_modules"
    if not node_modules.exists():
        check(WARN, "frontend", "node_modules отсутствует: выполните npm ci")
    else:
        check(OK, "frontend", "зависимости установлены")


def check_git_dirty() -> None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001
        check(WARN, "git", "не удалось проверить состояние")
        return
    changed = [line for line in out.splitlines() if line.strip()]
    if changed:
        check(WARN, "git", f"незакоммиченных файлов: {len(changed)}")
    else:
        check(OK, "git", "дерево чистое")


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка готовности к релизу")
    parser.add_argument("--strict", action="store_true", help="предупреждения тоже блокируют")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    check_env_file()
    check_secret()
    check_cors()
    check_ports()
    check_migrations()
    check_disk()
    check_frontend()
    check_git_dirty()

    width = max(len(name) for _, name, _ in results)
    for status, name, detail in results:
        print(f"[{status}] {name:<{width}}  {detail}")

    fails = sum(1 for s, _, _ in results if s == FAIL)
    warns = sum(1 for s, _, _ in results if s == WARN)
    print(f"\nИтого: {fails} блокирующих, {warns} предупреждений")

    if fails:
        return 1
    if args.strict and warns:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
