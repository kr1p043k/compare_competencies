"""Tests for HybridWeightCalculator."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import Ok, Err
from src.parsing.skills.hybrid_weight_calculator import HybridWeightCalculator


class TestHybridWeightCalculator:
    def test_init(self):
        calc = HybridWeightCalculator(MagicMock())
        assert calc is not None

    def test_calculate_bm25_fails(self):
        bm25 = MagicMock()
        bm25.calculate_weights.return_value = Err("error")
        calc = HybridWeightCalculator(bm25)
        result = calc.calculate([])
        assert result.is_err()

    def test_calculate_bm25_empty(self):
        bm25 = MagicMock()
        bm25.calculate_weights.return_value = Ok({})
        calc = HybridWeightCalculator(bm25)
        result = calc.calculate([])
        assert result.is_ok()
        assert result.ok() == {}

    def test_calculate_no_embeddings(self):
        bm25 = MagicMock()
        bm25.calculate_weights.return_value = Ok({"python": 0.5, "sql": 0.3})
        cache = MagicMock()
        cache.model = None
        calc = HybridWeightCalculator(bm25, embedding_cache=cache)
        result = calc.calculate([])
        assert result.is_ok()
        weights = result.ok()
        assert "python" in weights
        assert "sql" in weights

    def test_calculate_too_few_embeddings(self):
        bm25 = MagicMock()
        bm25.calculate_weights.return_value = MagicMock(is_ok=True, ok=lambda: {"python": 0.5})
        cache = MagicMock()
        cache.model = object()
        cache.get_embeddings.return_value = {"python": np.array([0.1, 0.2])}
        calc = HybridWeightCalculator(bm25, embedding_cache=cache)
        result = calc.calculate([])
        assert result.is_ok()

    def test_norm(self):
        weights = {"a": 0.8, "b": 0.2, "c": 0.5}
        normed = HybridWeightCalculator._norm(weights)
        assert abs(normed["a"] - 1.0) < 0.001
        assert abs(normed["b"] - 0.0) < 0.001
        assert abs(normed["c"] - 0.5) < 0.001

    def test_norm_empty(self):
        assert HybridWeightCalculator._norm({}) == {}

    def test_norm_single(self):
        assert HybridWeightCalculator._norm({"a": 1.0}) == {"a": 1.0}
