"""Тесты проверки дрейфа версий зависимостей.

Регрессия из жизни: в requirements стоял `aiohttp==3.12.14`, в окружении
оказался 3.14.4, и 15 тестов упали на несовместимости мок-библиотеки. Ошибка
выглядела как баг парсера. Проверка должна ловить такое расхождение сразу.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "check_dependency_drift.py"
    spec = importlib.util.spec_from_file_location("check_dependency_drift", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_dependency_drift"] = module
    spec.loader.exec_module(module)
    return module


drift = _load()


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_parses_hard_pins(tmp_path):
    f = _write(tmp_path, "requirements.txt", "aiohttp==3.12.14\nhttpx==0.28.1\n")
    pins = drift.parse_pins([f])
    assert pins == {"aiohttp": "3.12.14", "httpx": "0.28.1"}


def test_ignores_ranges_and_comments(tmp_path):
    f = _write(
        tmp_path,
        "requirements.txt",
        "# comment\ntorch>=2.5.0\nnumpy~=1.26\nrequests==2.32.3  # trailing\n",
    )
    pins = drift.parse_pins([f])
    assert pins == {"requests": "2.32.3"}


def test_normalises_package_names(tmp_path):
    f = _write(tmp_path, "requirements.txt", "Pillow==10.0.0\nscikit_learn==1.4.0\n")
    pins = drift.parse_pins([f])
    assert "pillow" in pins and "scikit-learn" in pins


def test_missing_file_is_skipped(tmp_path):
    assert drift.parse_pins([tmp_path / "nope.txt"]) == {}


def test_detects_version_mismatch(tmp_path, monkeypatch):
    f = _write(tmp_path, "requirements.txt", "definitely-not-installed-pkg==0.1\n")
    monkeypatch.setattr(
        drift, "installed_version", lambda name: "9.9.9" if name == "definitely-not-installed-pkg" else None
    )
    problems = drift.check([f])
    assert len(problems) == 1
    assert problems[0]["kind"] == "версия отличается"
    assert problems[0]["actual"] == "9.9.9"


def test_detects_missing_package(tmp_path, monkeypatch):
    f = _write(tmp_path, "requirements.txt", "ghost-pkg==1.0\n")
    monkeypatch.setattr(drift, "installed_version", lambda name: None)
    problems = drift.check([f])
    assert problems and problems[0]["kind"] == "не установлен"


def test_reports_nothing_when_versions_match(tmp_path, monkeypatch):
    f = _write(tmp_path, "requirements.txt", "solo-pkg==2.0\n")
    monkeypatch.setattr(drift, "installed_version", lambda name: "2.0")
    assert drift.check([f]) == []
