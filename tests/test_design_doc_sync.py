"""Гейт синхронности дизайн-системы и её документации.

Каждый модуль в frontend/src/design (кроме тестов) должен быть упомянут в
docs/DESIGN-SYSTEM.md. Иначе новый примитив появится в коде, но не в доках, и
команда снова начнёт плодить локальные копии.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DESIGN_DIR = REPO / "frontend" / "src" / "design"
DOC = REPO / "docs" / "DESIGN-SYSTEM.md"


def test_every_design_module_is_documented() -> None:
    doc = DOC.read_text(encoding="utf-8")
    modules = sorted(
        p.name
        for p in DESIGN_DIR.iterdir()
        if p.is_file()
        and p.suffix in {".ts", ".tsx"}
        and not p.name.endswith(".test.ts")
        and not p.name.endswith(".test.tsx")
    )
    missing = [name for name in modules if name not in doc]
    assert not missing, (
        "Модули src/design отсутствуют в docs/DESIGN-SYSTEM.md: "
        + ", ".join(missing)
    )
