"""Гейт дрейфа openapi.json (пункт 64: fail-closed CI-проверки).

CI пересобирает схему и падает, если закоммиченный openapi.json устарел
(.github/workflows/ci.yml, шаг "OpenAPI drift check"). Этот тест повторяет
проверку в pytest, чтобы расхождение ловилось локально и на гейте pytest.

ВАЖНО про версии: схема FastAPI зависит от версии fastapi/pydantic/starlette
(например, форма поля upload меняется между выпусками). Закоммиченный
openapi.json сгенерирован на версиях из requirements.txt. Если локально
установлены другие версии, тест делает skip - иначе он ловил бы дрейф
окружения, а не дрейф схемы. В CI окружение ставится из requirements.txt,
поэтому там проверка выполняется по-настоящему.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OPENAPI_PATH = ROOT / "openapi.json"
REQUIREMENTS = ROOT / "requirements.txt"

#: Пакеты, версии которых определяют форму схемы.
_SCHEMA_PACKAGES = ("fastapi", "starlette", "pydantic")


def _pinned_versions() -> dict[str, str]:
    """Версии пакетов, закреплённые в requirements.txt."""
    pins: dict[str, str] = {}
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, ver = line.partition("==")
        name = name.split("[")[0].strip().lower().replace("_", "-")
        ver = ver.split(";")[0].strip()
        pins[name] = ver
    return pins


def _installed_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _SCHEMA_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "?"
    return out


def _versions_match() -> bool:
    pins = _pinned_versions()
    installed = _installed_versions()
    return all(pins.get(name, installed[name]) == installed[name] for name in _SCHEMA_PACKAGES)


def _generated_text() -> str:
    # Тот же способ сборки, что в scripts/export_openapi.py::_write_schema.
    from src.api_pkg import create_app

    schema = create_app().openapi()
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@pytest.mark.skipif(
    not _versions_match(),
    reason="Версии fastapi/starlette/pydantic расходятся с requirements.txt; "
    "схема зависит от версий. Проверку выполняет CI с пинами.",
)
def test_openapi_json_is_in_sync_with_app():
    """Схема в коде совпадает с закоммиченным openapi.json.

    Если упало - схема изменилась, но артефакт не обновили. Починить:
        python scripts/export_openapi.py
    """
    committed = OPENAPI_PATH.read_text(encoding="utf-8")
    generated = _generated_text()

    if committed != generated:
        c_lines = committed.splitlines()
        g_lines = generated.splitlines()
        first = next(
            (i for i, (a, b) in enumerate(zip(c_lines, g_lines, strict=False)) if a != b),
            min(len(c_lines), len(g_lines)),
        )
        for i in range(max(0, first - 2), min(len(c_lines), first + 3)):
            print("committed:", c_lines[i] if i < len(c_lines) else "<eof>")
            print("generated:", g_lines[i] if i < len(g_lines) else "<eof>")
        raise AssertionError(
            "openapi.json устарел (первое расхождение на строке "
            f"{first + 1}). Запусти: python scripts/export_openapi.py"
        )
