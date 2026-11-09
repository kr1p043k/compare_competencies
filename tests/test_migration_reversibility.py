"""Статическая проверка миграций alembic.

Живой downgrade на БД НЕ запускаем (общая БД, другие сессии): проверяем
структуру файлов - у каждой ревизии есть revision/down_revision и обе функции.
Ловит «забыл downgrade()» и битые ревизии до прогона.

Плюс отдельный тест единственной головы (``alembic heads`` == 1) уже есть в
tests/test_retention_cli.py; здесь его не дублируем.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

VERSIONS = Path(__file__).resolve().parent.parent / "alembic" / "versions"
FILES = sorted(p for p in VERSIONS.glob("*.py") if p.name != "__init__.py")


def _module_assign_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _func_names(tree: ast.Module) -> set[str]:
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_versions_dir_not_empty():
    assert FILES, "каталог alembic/versions пуст"


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_migration_declares_revision_and_downgrade(path: Path):
    # read_bytes: ast.parse сам снимает UTF-8 BOM (str с BOM падает на U+FEFF).
    tree = ast.parse(path.read_bytes(), filename=str(path))
    assigns = _module_assign_names(tree)
    funcs = _func_names(tree)
    assert "revision" in assigns, f"{path.name}: нет revision"
    assert "down_revision" in assigns, f"{path.name}: нет down_revision"
    assert "upgrade" in funcs, f"{path.name}: нет upgrade()"
    assert "downgrade" in funcs, f"{path.name}: нет downgrade()"
