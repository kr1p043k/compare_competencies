"""Запуск backend из worktree, где нет своего .env.

Подхватывает переменные из канонической копии проекта и стартует uvicorn.
Секреты не печатаются.

Использование: python tools/run_dev_backend.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_SOURCE = Path(os.environ.get("CC_ENV_FILE", r"C:\forCode\compare_online\.env"))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        count += 1
    return count


print(f"env vars loaded: {load_env(ENV_SOURCE)}", file=sys.stderr)

import uvicorn  # noqa: E402

uvicorn.run("src.api_pkg:app", host="127.0.0.1", port=8000, log_level="info")
