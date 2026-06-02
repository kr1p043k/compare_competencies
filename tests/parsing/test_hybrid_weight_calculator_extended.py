"""Extended tests for HybridWeightCalculator — covers all uncovered paths."""
import numpy as np
from unittest.mock import patch, PropertyMock

from src import Ok, Err
from src.errors import DomainError
from src.parsing.skills.hybrid_weight_calculator import HybridWeightCalculator


def _calc():
    from src.parsing.skills.bm25_ranker import BM25Ranker
    return HybridWeightCalculator(BM25Ranker())


# ============================================================================
# _norm static helper
# ============================================================================

class TestNorm:
    def test_empty(self):
        assert HybridWeightCalculator._norm({}) == {}

    def test_single_value(self):
        assert HybridWeightCalculator._norm({"a": 5.0}) == {"a": 5.0}

    def test_equal_values_returns_unchanged(self):
        d = {"a": 0.5, "b": 0.5}
        assert HybridWeightCalculator._norm(d) is d

    def test_normal_minmax(self):
        assert HybridWeightCalculator._norm({"a": 0.8, "b": 0.2, "c": 0.5}) == {"a": 1.0, "b": 0.0, "c": 0.5}


# ============================================================================
# calculate — BM25 error / empty
# ============================================================================

class TestCalculateBm25Errors:
    def test_bm25_returns_err(self):
        calc = _calc()
        with patch.object(calc.bm25, "calculate_weights", return_value=Err(DomainError("fail", "detail"))):
            result = calc.calculate([])
        assert result.is_err()
        assert "BM25" in result.err().message

    def test_bm25_returns_empty_ok(self):
        calc = _calc()
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok({})):
            result = calc.calculate([])
        assert result.is_ok()
        assert result.unwrap() == {}


# ============================================================================
# calculate — fallback paths (no full hybrid)
# ============================================================================

class TestCalculateFallback:
    def test_model_unavailable(self):
        calc = _calc()
        bm25 = {"a": 0.8, "b": 0.6}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(bm25)):
            with patch.object(type(calc.cache), "model", new_callable=PropertyMock, return_value=None):
                result = calc.calculate([])
        assert result.unwrap() == {"a": 1.0, "b": 0.0}

    def test_get_embeddings_exception(self):
        calc = _calc()
        bm25 = {"python": 0.8, "sql": 0.6}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(bm25)):
            with patch.object(calc.cache, "get_embeddings", side_effect=RuntimeError("fail")):
                result = calc.calculate([])
        assert result.unwrap() == {"python": 1.0, "sql": 0.0}

    def test_few_embeddings_less_than_10(self):
        calc = _calc()
        n, m = 15, 5
        bm25 = {f"s{i}": 0.5 for i in range(n)}
        mock_embs = {f"s{i}": np.random.rand(384).astype(np.float32) for i in range(m)}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(bm25)):
            with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
                result = calc.calculate([])
        weights = result.unwrap()
        assert len(weights) == n
        # all equal -> _norm returns unchanged
        assert all(v == 0.5 for v in weights.values())


# ============================================================================
# calculate — PCA paths
# ============================================================================

class TestCalculatePca:
    def test_pca_applied(self):
        calc = _calc()
        n = 150
        skills = {f"s{i}": float(i) / n for i in range(n)}
        mock_embs = {f"s{i}": np.random.rand(384).astype(np.float32) for i in range(n)}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(skills)):
            with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
                with patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_ENABLED", True), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_SAMPLES", 100), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_FEATURES", 128), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_TARGET_DIM", 64):
                    result = calc.calculate([])
        assert result.is_ok()
        assert len(result.unwrap()) == n

    def test_pca_not_enough_samples(self):
        calc = _calc()
        n = 50
        skills = {f"s{i}": 0.5 for i in range(n)}
        mock_embs = {f"s{i}": np.random.rand(384).astype(np.float32) for i in range(n)}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(skills)):
            with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
                with patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_ENABLED", True), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_SAMPLES", 100), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_FEATURES", 128):
                    result = calc.calculate([])
        assert result.is_ok()
        assert len(result.unwrap()) == n

    def test_pca_low_features(self):
        calc = _calc()
        n = 150
        skills = {f"s{i}": 0.5 for i in range(n)}
        mock_embs = {f"s{i}": np.random.rand(64).astype(np.float32) for i in range(n)}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(skills)):
            with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
                with patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_ENABLED", True), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_SAMPLES", 100), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_FEATURES", 128):
                    result = calc.calculate([])
        assert result.is_ok()
        assert len(result.unwrap()) == n

    def test_pca_target_dim_matches_features(self):
        calc = _calc()
        n = 150
        skills = {f"s{i}": 0.5 for i in range(n)}
        mock_embs = {f"s{i}": np.random.rand(200).astype(np.float32) for i in range(n)}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(skills)):
            with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
                with patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_ENABLED", True), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_SAMPLES", 100), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_FEATURES", 128), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_TARGET_DIM", 200):
                    result = calc.calculate([])
        assert result.is_ok()
        assert len(result.unwrap()) == n


# ============================================================================
# calculate — full hybrid computation
# ============================================================================

class TestCalculateHybrid:
    def test_full_hybrid(self):
        calc = _calc()
        n = 20
        bm25 = {f"s{i}": float(i) / n for i in range(n)}
        rng = np.random.RandomState(42)
        mock_embs = {f"s{i}": rng.rand(384).astype(np.float32) for i in range(n)}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(bm25)):
            with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
                with patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_ENABLED", False):
                    result = calc.calculate([])
        assert result.is_ok()
        weights = result.unwrap()
        assert len(weights) == n
        assert all(0.0 <= v <= 1.0 for v in weights.values())

    def test_bm25_only_skills_appended(self):
        calc = _calc()
        n, m = 20, 10
        bm25_all = {f"s{i}": 0.5 for i in range(n)}
        mock_embs = {f"s{i}": np.random.rand(384).astype(np.float32) for i in range(m)}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(bm25_all)):
            with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
                with patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_ENABLED", False):
                    result = calc.calculate([])
        assert result.is_ok()
        weights = result.unwrap()
        assert len(weights) == n
        for i in range(m, n):
            assert weights[f"s{i}"] == 0.5

    def test_custom_alpha_beta(self):
        calc = _calc()
        n = 20
        bm25 = {f"s{i}": 0.5 for i in range(n)}
        rng = np.random.RandomState(42)
        mock_embs = {f"s{i}": rng.rand(384).astype(np.float32) for i in range(n)}
        with patch.object(calc.bm25, "calculate_weights", return_value=Ok(bm25)):
            with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
                with patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_ENABLED", False), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.HYBRID_BM25_WEIGHT", 0.5, create=True), \
                     patch("src.parsing.skills.hybrid_weight_calculator.config.HYBRID_SEM_WEIGHT", 0.5, create=True):
                    result = calc.calculate([])
        assert result.is_ok()
        assert len(result.unwrap()) == n
