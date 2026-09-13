"""Unit (v41): scope overrides precedence + loader resilience."""
import pytest

from src.teacher_scope import (
    effective_excluded,
    effective_in_scope,
    load_scope_overrides,
    scope_source,
)


def test_methodology_baseline():
    assert effective_in_scope("Базы данных и СУБД", {}) is True
    assert effective_in_scope("Философия", {}) is False
    assert effective_in_scope("Иностранный язык (англ. яз., уровень А2)", {}) is False
    assert scope_source("Философия", {}) == "methodology"
    assert scope_source("Базы данных и СУБД", {}) == "default"


def test_override_excludes_core():
    over = {"Базы данных и СУБД": False}
    assert effective_in_scope("Базы данных и СУБД", over) is False
    assert scope_source("Базы данных и СУБД", over) == "custom"


def test_override_reincludes_methodology():
    over = {"Философия": True}
    assert effective_in_scope("Философия", over) is True
    assert "Философия" not in effective_excluded(["Философия", "Базы данных и СУБД"], over)


async def test_loader_missing_table_returns_empty():
    class BadPool:
        async def fetch(self, *a):
            raise RuntimeError('no such table')
    assert await load_scope_overrides(BadPool(), "09.03.02") == {}


async def test_loader_maps_rows():
    class FakePool:
        async def fetch(self, *a):
            return [{"discipline_name": "X", "included": False}]
    assert await load_scope_overrides(FakePool(), "09.03.02") == {"X": False}
