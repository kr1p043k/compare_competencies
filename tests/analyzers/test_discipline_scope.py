"""Discipline scope: explicit mapping, derived neighbours, and the ordering."""

from __future__ import annotations

import numpy as np

from src.analyzers import discipline_scope as scope


class FakeProvider:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = {k: np.asarray(v, dtype=float) for k, v in vectors.items()}

    def encode(self, texts, show_progress_bar=False):
        return np.array([self.vectors.get(t, [0.0, 0.0]) for t in texts])


class FakeMatcher:
    def __init__(self, names, vectors):
        self._market_names = list(names)
        self._embedding_provider = FakeProvider(vectors)
        matrix = np.array([vectors[n] for n in names], dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._market_embeddings = matrix / norms


MATCHER = FakeMatcher(
    ["python", "pytest", "qa", "testing", "frontend", "react"],
    {
        "python": [1.0, 0.0],
        "pytest": [0.98, 0.2],
        "qa": [0.9, 0.4],
        "testing": [0.88, 0.47],
        "frontend": [0.0, 1.0],
        "react": [0.1, 0.99],
        # anchors/keywords resolved through the provider
        "автотесты": [0.95, 0.3],
    },
)


def test_neighbours_returns_close_skills_only():
    got = scope.neighbours(MATCHER, "python", limit=3, min_sim=0.55)
    assert "pytest" in got
    assert "react" not in got


def test_derived_scope_uses_the_anchors():
    resolved = scope.resolve_scope(MATCHER, "disc", ["python"], {})
    assert resolved is not None
    skills, source = resolved
    assert source == "derived"
    assert "pytest" in skills


def test_explicit_mapping_wins_over_derived():
    overrides = {"disc": {"skills": ["react"], "keywords": []}}
    skills, source = scope.resolve_scope(MATCHER, "disc", ["python"], overrides)
    assert source == "explicit"
    assert skills == {"react"}


def test_keywords_expand_through_neighbours():
    overrides = {"disc": {"skills": [], "keywords": ["автотесты"]}}
    skills, source = scope.resolve_scope(MATCHER, "disc", ["python"], overrides)
    assert source == "explicit"
    assert "автотесты" in skills
    assert "pytest" in skills  # semantic neighbour of the keyword


def test_no_anchors_and_no_overrides_returns_none():
    assert scope.resolve_scope(MATCHER, "disc", [], {}) is None


def test_overrides_round_trip(tmp_path):
    path = tmp_path / "scope.json"
    scope.save_overrides({"disc": {"skills": ["Python"], "keywords": ["qa"]}}, path)
    loaded = scope.load_overrides(path)
    assert loaded == {"disc": {"skills": ["Python"], "keywords": ["qa"]}}


def test_missing_or_broken_file_yields_empty(tmp_path):
    assert scope.load_overrides(tmp_path / "nope.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert scope.load_overrides(broken) == {}


def test_describe_reports_source_and_configuration():
    overrides = {"disc": {"skills": ["react"], "keywords": []}}
    info = scope.describe(MATCHER, "disc", ["python"], overrides)
    assert info["source"] == "explicit"
    assert info["configured"] is True
    assert info["skills"] == ["react"]


def test_neighbours_without_embeddings_is_safe():
    class Bare:
        _market_embeddings = None
        _market_names = None
        _embedding_provider = None

    assert scope.neighbours(Bare(), "python") == []
