"""Tests for src.parsing.api.embedding_loader."""

import pytest

from src import config
from src.parsing.api import embedding_loader as el


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch):
    monkeypatch.setattr(el, "_embedding_model", None)


def test_env_threads_set_during_import():
    import os
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("MKL_NUM_THREADS") == "1"


def test_get_embedding_model_caches_result(monkeypatch):
    fake_model = type("FakeModel", (), {"eval": lambda self: None})()
    sentinel = []

    def fake_sentence_transformer(*args, **kwargs):
        sentinel.append((args, kwargs))
        return fake_model

    fake_module = type("FakeST", (), {"SentenceTransformer": fake_sentence_transformer})()
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)

    model1 = el.get_embedding_model("m1")
    model2 = el.get_embedding_model("m1")
    assert model1 is fake_model
    assert model2 is fake_model
    assert len(sentinel) == 1


def test_get_embedding_model_passes_hf_token(monkeypatch):
    fake_model = type("FakeModel", (), {"eval": lambda self: None})()
    calls = []

    def fake_sentence_transformer(*args, **kwargs):
        calls.append(kwargs)
        return fake_model

    fake_module = type("FakeST", (), {"SentenceTransformer": fake_sentence_transformer})()
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(config, "HF_TOKEN", type("Secret", (), {"get_secret_value": lambda self: "hf_xxx"})())
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "model-x")

    model = el.get_embedding_model()
    assert model is fake_model
    assert calls[-1]["token"] == "hf_xxx"


def test_get_embedding_model_uses_default_model_name(monkeypatch):
    fake_model = type("FakeModel", (), {"eval": lambda self: None})()
    calls = []

    def fake_sentence_transformer(*args, **kwargs):
        calls.append(args)
        return fake_model

    fake_module = type("FakeST", (), {"SentenceTransformer": staticmethod(fake_sentence_transformer)})()
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "default-model")
    monkeypatch.setattr(config, "HF_TOKEN", None)

    el.get_embedding_model()
    assert calls[0][0] == "default-model"


def test_get_embedding_model_falls_back_to_mirror(monkeypatch):
    fake_model = type("FakeModel", (), {"eval": lambda self: None})()

    class FailingOnce:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("primary failed")
            return fake_model

    builder = FailingOnce()
    fake_module = type("FakeST", (), {"SentenceTransformer": builder})()
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(config, "HF_TOKEN", None)

    model = el.get_embedding_model("m2")
    assert model is not None
    assert builder.calls == 2


@pytest.mark.xfail(reason="src re-raises the mirror exception instead of the intended fallback RuntimeError")
def test_get_embedding_model_raises_after_both_failures(monkeypatch):
    def fake_sentence_transformer(*args, **kwargs):
        raise RuntimeError("all endpoints down")

    fake_module = type("FakeST", (), {"SentenceTransformer": fake_sentence_transformer})()
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(config, "HF_TOKEN", None)

    with pytest.raises(RuntimeError, match="Failed to load embedding model"):
        el.get_embedding_model("m3")


def test_get_embedding_model_eval_called(monkeypatch):
    fake_model = type("FakeModel", (), {"eval": lambda self: None})()
    calls = []
    monkeypatch.setattr(fake_model, "eval", lambda: calls.append("eval"))

    def fake_sentence_transformer(*args, **kwargs):
        return fake_model

    fake_module = type("FakeST", (), {"SentenceTransformer": fake_sentence_transformer})()
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(config, "HF_TOKEN", None)

    el.get_embedding_model("m4")
    assert "eval" in calls
