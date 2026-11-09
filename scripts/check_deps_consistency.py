"""Проверка согласованности зависимостей: pyproject <-> requirements.

Docker и CI ставят пакеты через `pip install -r requirements*.txt`. Если
зависимость объявлена только в pyproject, на проде она окажется отсутствующей
(или подтянется транзитивно случайной версией). Реальный случай: `scipy` был в
pyproject и в коде (`src/predictors/ltr_recommendation_engine.py`), но не в
requirements.txt.

Проверяем опасное направление:
  * каждая runtime-зависимость из pyproject [project].dependencies должна быть
    в requirements.txt;
  * каждая dev-зависимость из pyproject [project.optional-dependencies].dev -
    в requirements-dev.txt.

Обратное направление (пин в requirements, которого нет в pyproject) допустимо:
так фиксируют транзитивные пакеты.

Exit 0 - согласовано, exit 1 - список отсутствующих.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _norm(name: str) -> str:
    return re.split(r"[<>=!~\[\s]", name, maxsplit=1)[0].strip().lower().replace("_", "-")


def _req_names(path: Path) -> set[str]:
    names: set[str] = set()
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        names.add(_norm(line))
    return names


def main() -> int:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    runtime = {_norm(d) for d in project.get("dependencies", [])}
    dev = {_norm(d) for d in project.get("optional-dependencies", {}).get("dev", [])}

    req = _req_names(ROOT / "requirements.txt")
    req_dev = _req_names(ROOT / "requirements-dev.txt") | req

    missing_runtime = sorted(runtime - req)
    missing_dev = sorted(dev - req_dev)

    problems = 0
    if missing_runtime:
        problems += len(missing_runtime)
        print("runtime-зависимости из pyproject отсутствуют в requirements.txt:")
        for n in missing_runtime:
            print("   ", n)
    if missing_dev:
        problems += len(missing_dev)
        print("dev-зависимости из pyproject отсутствуют в requirements-dev.txt:")
        for n in missing_dev:
            print("   ", n)

    if problems:
        print(f"\nИТОГ: {problems} расхождений. Docker/CI ставят requirements*.txt.")
        return 1

    print(
        f"зависимости согласованы: runtime={len(runtime)}, dev={len(dev)}, "
        f"requirements={len(req)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
