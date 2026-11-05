"""Тесты аудита тихих исключений.

Регрессия по смыслу: битый JSON в discipline_scope возвращал пустой словарь
без единой строки в логе, и все ручные исключения дисциплин молча пропадали —
анализ считался по более широкому набору, чем настроил преподаватель.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "audit_silent_excepts.py"
    spec = importlib.util.spec_from_file_location("audit_silent_excepts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_silent_excepts"] = module
    spec.loader.exec_module(module)
    return module


audit = _load()


def _scan(tmp_path: Path, code: str) -> list[dict]:
    f = tmp_path / "sample.py"
    f.write_text(textwrap.dedent(code), encoding="utf-8")
    # scan() вычисляет путь относительно ROOT, поэтому подменяем его
    original = audit.ROOT
    audit.ROOT = tmp_path
    try:
        return audit.scan(f)
    finally:
        audit.ROOT = original


def test_detects_pass(tmp_path):
    found = _scan(tmp_path, """
        try:
            x = 1
        except Exception:
            pass
    """)
    assert len(found) == 1
    assert found[0]["kind"] == "pass"


def test_detects_return_none(tmp_path):
    found = _scan(tmp_path, """
        def f():
            try:
                return 1
            except ValueError:
                return None
    """)
    assert [f["kind"] for f in found] == ["return None"]


def test_detects_return_empty_collection(tmp_path):
    found = _scan(tmp_path, """
        def f():
            try:
                return [1]
            except KeyError:
                return []
            except TypeError:
                return {}
    """)
    assert sorted(f["kind"] for f in found) == ["return empty", "return empty"]


def test_detects_continue(tmp_path):
    found = _scan(tmp_path, """
        for item in range(3):
            try:
                int(item)
            except ValueError:
                continue
    """)
    assert [f["kind"] for f in found] == ["continue"]


def test_logged_handler_is_not_flagged(tmp_path):
    found = _scan(tmp_path, """
        def f():
            try:
                return 1
            except ValueError as exc:
                logger.warning("failed", error=str(exc))
                return None
    """)
    assert found == []


def test_handler_with_extra_statements_is_not_flagged(tmp_path):
    # Несколько операторов — уже не «проглотил и забыл».
    found = _scan(tmp_path, """
        def f():
            try:
                return 1
            except ValueError:
                counter += 1
                pass
    """)
    assert found == []


def test_critical_path_detection():
    assert audit.is_critical("src/pipeline/runner.py")
    assert audit.is_critical("src/analyzers/skill_matcher.py")
    assert not audit.is_critical("src/utils.py")
