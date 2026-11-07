"""Unit tests for LLMClient base_url handling (offline, no openai package).

The openai package is an optional runtime dependency of LLMClient (installed
lazily inside __init__). These tests stub `openai` in sys.modules so that the
`from openai import OpenAI` inside LLMClient.__init__ resolves to a fake class
that records the base_url passed into it. That recorded value is exactly the
URL the real OpenAI client would be pointed at.

Covers:
  (a) a gateway URL already ending in /v1 is NOT doubled (no .../v1/v1)
  (b) an Ollama-style base without /v1 gets the OpenAI-compatible /v1 suffix
  (c) trailing slash before /v1 does not produce a doubled suffix either
  (d) the new LLM_CHAT_* settings are exported with their defaults
"""
from __future__ import annotations

import sys
import types


class _FakeOpenAI:
    """Stand-in for openai.OpenAI; records the kwargs __init__ received."""

    def __init__(self, **kwargs):
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")


class _OpenAIStub(types.ModuleType):
    """Module stand-in exposing only the OpenAI attribute used by LLMClient."""

    def __getattr__(self, name):
        if name == "OpenAI":
            return _FakeOpenAI
        raise AttributeError(name)


def _install_openai_stub(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", _OpenAIStub("openai"))


def _make_client(monkeypatch, base_url: str):
    _install_openai_stub(monkeypatch)
    from src.services.llm_client import LLMClient

    return LLMClient(base_url=base_url)


class TestLLMClientBaseUrl:
    def test_gateway_v1_url_not_doubled(self, monkeypatch):
        client = _make_client(monkeypatch, "http://x/k1/v1")
        assert client._client.base_url == "http://x/k1/v1"
        assert client._client.base_url != "http://x/k1/v1/v1"

    def test_ollama_base_without_v1_gets_v1(self, monkeypatch):
        client = _make_client(monkeypatch, "http://ollama:11434")
        assert client._client.base_url == "http://ollama:11434/v1"

    def test_gateway_v1_trailing_slash_not_doubled(self, monkeypatch):
        client = _make_client(monkeypatch, "http://x/k1/v1/")
        assert client._client.base_url == "http://x/k1/v1"

    def test_llm_chat_config_defaults_exported(self):
        from src import config

        assert config.LLM_CHAT_URL == "http://127.0.0.1:8797/k1/v1"
        assert config.LLM_CHAT_MODEL == "deepseek-ai/DeepSeek-V4.1-Flash"
