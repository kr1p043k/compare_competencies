"""Coverage for src/services/llm_pricing.py (backlog #49).

Pure in-memory cost accounting: no network, no DB, no LLM calls. The env
override loader ``_load_env_pricing`` is tested via the ``LLM_PRICING`` env
var pointing at tmp_path files, so nothing touches the repo tree.

Layers:
  * ``estimate_cost_usd`` - the only public function (happy / unknown-model /
    free-model / zero-token / negative-token / override-precedence paths).
  * ``_load_env_pricing`` - env-var access, filesystem discovery and JSON
    parsing behind the override mechanism.
"""
from __future__ import annotations

import json

import pytest

from src.services import llm_pricing


def _write_pricing_file(tmp_path, payload) -> str:
    path = tmp_path / "pricing.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestEstimateCostUsd:
    def test_yandexgpt_lite_cost_matches_constants(self):
        """Happy path: cost is list-price input+output per 1M tokens."""
        cost = llm_pricing.estimate_cost_usd("yandexgpt-lite", 1_000_000, 0)
        assert cost == pytest.approx(250 / 90, rel=1e-9)

        cost = llm_pricing.estimate_cost_usd("yandexgpt-lite", 0, 1_000_000)
        assert cost == pytest.approx(750 / 90, rel=1e-9)

        cost = llm_pricing.estimate_cost_usd("yandexgpt-lite", 1_000, 500)
        expected = (1_000 * 250 / 90 + 500 * 750 / 90) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_unknown_model_returns_none(self):
        """Failure path: model outside the priced set returns None."""
        assert llm_pricing.estimate_cost_usd("unknown-model", 1_000, 1_000) is None

    def test_free_models_cost_zero(self):
        """Boundary: self-hosted open-source models are priced at 0.0."""
        for model in ("gpt-oss:120b", "qwen2.5:0.5b"):
            assert llm_pricing.estimate_cost_usd(model, 10_000, 10_000) == 0.0

    def test_zero_and_negative_tokens(self):
        """Boundary: zero tokens are free; negatives are not clamped."""
        assert llm_pricing.estimate_cost_usd("yandexgpt-lite", 0, 0) == 0.0
        expected = round(-100 * 250 / 90 / 1_000_000, 10)
        assert llm_pricing.estimate_cost_usd("yandexgpt-lite", -100, 0) == expected

    def test_env_pricing_overrides_defaults(self, monkeypatch):
        """Environment overrides win over the built-in default prices."""
        monkeypatch.setattr(
            llm_pricing,
            "_env_pricing",
            {"yandexgpt-lite": {"input": 0.001, "output": 0.003}},
        )
        expected = (1_000 * 0.001 + 500 * 0.003) / 1_000_000
        assert llm_pricing.estimate_cost_usd("yandexgpt-lite", 1_000, 500) == pytest.approx(
            expected, rel=1e-9
        )

    def test_defaults_apply_when_override_has_other_model(self, monkeypatch):
        """Boundary: override entries for other models do not hide defaults."""
        monkeypatch.setattr(llm_pricing, "_env_pricing", {"other-model": {"input": 1.0}})
        expected = (1_000 * 250 / 90 + 500 * 750 / 90) / 1_000_000
        assert llm_pricing.estimate_cost_usd("yandexgpt-lite", 1_000, 500) == pytest.approx(
            expected, rel=1e-6
        )


class TestLoadEnvPricing:
    def test_no_env_var_returns_empty(self, monkeypatch):
        monkeypatch.delenv("LLM_PRICING", raising=False)
        assert llm_pricing._load_env_pricing() == {}

    def test_empty_env_var_returns_empty(self, monkeypatch):
        monkeypatch.setenv("LLM_PRICING", "")
        assert llm_pricing._load_env_pricing() == {}

    def test_valid_file_is_loaded(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "LLM_PRICING",
            _write_pricing_file(tmp_path, {"yandexgpt-lite": {"input": 0.001, "output": 0.003}}),
        )
        assert llm_pricing._load_env_pricing() == {
            "yandexgpt-lite": {"input": 0.001, "output": 0.003}
        }

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_PRICING", str(tmp_path / "no-such-file.json"))
        assert llm_pricing._load_env_pricing() == {}

    def test_invalid_json_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_PRICING", _write_pricing_file(tmp_path, "{{{not-json"))
        assert llm_pricing._load_env_pricing() == {}

    def test_non_dict_json_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_PRICING", _write_pricing_file(tmp_path, [1, 2, 3]))
        assert llm_pricing._load_env_pricing() == {}
