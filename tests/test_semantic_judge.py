"""LLM-судья хвоста: заготовки. Без сети — mock-клиент и недоступность."""

import numpy as np

from src import config
from src.services import semantic_judge as sj


class _MockClient:
    def __init__(self, text="", exc=None):
        self.text = text
        self.exc = exc
        self.calls = 0

    def generate(self, prompt, temperature=None, max_tokens=None):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.text


def test_judge_disabled_skips_without_call(monkeypatch):
    monkeypatch.setattr(config, "LLM_JUDGE_ENABLED", False)
    client = _MockClient('{"0": true}')
    out = sj.judge_pairs([("pytest", "vlan", 0.64)], client=client)
    assert out["skipped"] == "disabled"
    assert out["verdicts"] == {}
    assert client.calls == 0


def test_judge_parses_verdicts(monkeypatch):
    monkeypatch.setattr(config, "LLM_JUDGE_ENABLED", True)
    client = _MockClient('{"0": true, "1": false}')
    out = sj.judge_pairs([("a", "b", 0.65), ("c", "d", 0.66)], client=client)
    assert out["skipped"] is None
    assert out["verdicts"] == {("a", "b"): True, ("c", "d"): False}


def test_judge_garbage_abstains_not_vetoes(monkeypatch):
    monkeypatch.setattr(config, "LLM_JUDGE_ENABLED", True)
    client = _MockClient("не json вообще")
    out = sj.judge_pairs([("a", "b", 0.65)], client=client)
    assert out["verdicts"] == {("a", "b"): None}


def test_judge_unavailable_skips(monkeypatch):
    monkeypatch.setattr(config, "LLM_JUDGE_ENABLED", True)
    client = _MockClient(exc=ConnectionError("down"))
    out = sj.judge_pairs([("a", "b", 0.65)], client=client)
    assert out["verdicts"] == {}
    assert out["skipped"].startswith("unavailable")


class _StubComp:
    def embed_skills(self, skills):
        return np.array([[float(i + 1), 0.0] for i in range(len(skills))])


def test_gray_zone_bounds():
    matches = [
        {"skill": "low", "similarity": 0.59},
        {"skill": "in1", "similarity": 0.60},
        {"skill": "in2", "similarity": 0.749},
        {"skill": "high", "similarity": 0.75},
    ]
    out = sj.gray_pairs_for_judge(_StubComp(), ["q1", "q2"], matches)
    assert {m for _, m, _ in out} == {"in1", "in2"}
