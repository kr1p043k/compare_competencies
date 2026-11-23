"""Тесты роль-зависимого смешивания рыночных и научных весов."""

from __future__ import annotations

import json

import pytest

from src import config
from src.analyzers.skills import science_weights
from src.analyzers.skills.science_weights import RoleSourceMix


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def files(tmp_path):
    article = _write(
        tmp_path / "article_weights.json",
        {
            "_meta": {"methodology_version": "v1"},
            "roles": {
                "Data Scientist": {"python": 10, "pytorch": 8, "sql": 4},
                "Backend Developer": {"python": 9, "sql": 6},
            },
            "global": {"python": 19, "sql": 10, "pytorch": 8},
        },
    )
    mix = _write(
        tmp_path / "role_source_mix.json",
        {
            "default": 0.0,
            "roles": {"data science": 0.5, "backend developer": 0.05},
        },
    )
    return article, mix


def test_from_files_normalizes_per_role(files):
    article, mix = files
    m = RoleSourceMix.from_files(article, mix)
    assert m.active
    assert m.article_weights["data scientist"] == {"python": 1.0, "pytorch": 0.8, "sql": 0.4}
    assert m.article_weights["backend developer"] == {"python": 1.0, "sql": pytest.approx(0.6667, abs=1e-3)}
    assert m.global_weights["python"] == 1.0


def test_lambda_resolution_and_substring_roles(files):
    article, mix = files
    m = RoleSourceMix.from_files(article, mix)
    assert m.lambda_for("Data Scientist") == 0.5
    assert m.lambda_for("📊 Data Science и ML") == 0.5
    assert m.lambda_for("Backend Developer") == 0.05
    assert m.lambda_for("Unknown Role") == 0.0
    assert m.lambda_for(None) == 0.0


def test_blend_pulls_toward_science_and_keeps_unsupported_skills(files):
    article, mix = files
    m = RoleSourceMix.from_files(article, mix)
    market = {"python": 1.0, "sql": 0.5, "k8s": 0.25}
    out = m.blend(market, "Data Scientist")
    # python: рынок 1.0 и наука 1.0 -> 1.0; sql: рынок 0.5, наука 0.4 -> 0.45
    assert out["python"] == pytest.approx(1.0)
    assert out["sql"] == pytest.approx(0.45)
    # k8s нет в публикациях: научного свидетельства нет, вес рынка сохраняется
    assert out["k8s"] == 0.25


def test_blend_identity_when_lambda_zero(files):
    article, mix = files
    m = RoleSourceMix.from_files(article, mix)
    market = {"python": 1.0, "sql": 0.5}
    out = m.blend(market, "Frontend Developer")
    assert out == market
    assert out is not market


def test_blend_by_level_returns_input_when_inactive(tmp_path):
    article = _write(tmp_path / "a.json", {"roles": {"x": {"python": 1}}})
    mix = _write(tmp_path / "m.json", {"default": 0.0, "roles": {}})
    m = RoleSourceMix.from_files(article, mix)
    assert not m.active
    levels = {"junior": {"python": 0.5}}
    assert m.blend_by_level(levels, "Data Scientist") is levels


def test_missing_files_give_inactive_mix(tmp_path):
    m = RoleSourceMix.from_files(tmp_path / "nope.json", tmp_path / "nope2.json")
    assert not m.active
    assert m.blend({"python": 1.0}, "Data Scientist") == {"python": 1.0}


def test_bad_lambda_and_negative_weights_are_sanitized(tmp_path):
    article = _write(tmp_path / "a.json", {"roles": {"r": {"python": 5, "bad": -3}}})
    mix = _write(tmp_path / "m.json", {"default": "nope", "roles": {"r": "oops"}})
    m = RoleSourceMix.from_files(article, mix)
    assert m.lambda_for("r") == 0.0
    assert "bad" not in m.article_weights["r"]
    assert m.default_lambda == 0.0


def test_default_mix_respects_kill_switch(monkeypatch):
    monkeypatch.setattr(config, "ROLE_SOURCE_MIX_ENABLED", False)
    monkeypatch.setattr(science_weights, "_cached", None)
    m = science_weights.default_mix()
    assert not m.active
    assert m.source == "disabled"


def test_default_mix_caches_by_mtime(monkeypatch, files, tmp_path):
    article, mix = files
    monkeypatch.setattr(config, "ARTICLE_WEIGHTS_PATH", article)
    monkeypatch.setattr(config, "ROLE_SOURCE_MIX_PATH", mix)
    monkeypatch.setattr(science_weights, "_cached", None)
    first = science_weights.default_mix()
    second = science_weights.default_mix()
    assert first is second
    assert first.active
