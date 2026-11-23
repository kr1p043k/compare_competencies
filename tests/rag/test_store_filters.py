"""Тесты генерации фильтров поиска (без БД): плейсхолдеры и параметры."""

from __future__ import annotations

from src.rag.store import RagStore


def test_base_filters_expands_scopes_and_sources():
    where, params = RagStore()._base_filters(
        scopes=["public", "internal"],
        sources=["rpd", "krm"],
        direction="09.03.02",
        year=2025,
    )
    assert "d.acl_scope IN (:scope0, :scope1)" in where
    assert "d.source IN (:src0, :src1)" in where
    assert "d.direction = :direction" in where
    assert "d.year = :year" in where
    assert "d.generation = s.active_generation" in where
    assert params == {
        "scope0": "public",
        "scope1": "internal",
        "src0": "rpd",
        "src1": "krm",
        "direction": "09.03.02",
        "year": 2025,
    }


def test_base_filters_without_sources_has_no_source_clause():
    where, params = RagStore()._base_filters(
        scopes=["internal"], sources=None, direction=None, year=None
    )
    assert "d.source" not in where
    assert "direction" not in params
    assert "year" not in params
    assert params == {"scope0": "internal"}


def test_base_filters_defaults_scope_to_internal():
    where, params = RagStore()._base_filters(
        scopes=[], sources=None, direction=None, year=None
    )
    assert "d.acl_scope IN (:scope0)" in where
    assert params["scope0"] == "internal"


def test_base_filters_never_uses_expanding_placeholders():
    where, _ = RagStore()._base_filters(
        scopes=["public"], sources=["rpd"], direction=None, year=None
    )
    assert "IN :" not in where
    assert "IN (:" in where
