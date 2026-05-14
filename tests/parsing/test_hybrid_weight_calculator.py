# tests/parsing/test_hybrid_weight_calculator.py
import numpy as np
import pytest
from unittest.mock import patch
from src.parsing.skills.hybrid_weight_calculator import HybridWeightCalculator
from src.parsing.skills.bm25_ranker import BM25Ranker

@pytest.fixture
def calc():
    ranker = BM25Ranker()
    return HybridWeightCalculator(ranker)

def test_hybrid_weights_empty_bm25(calc):
    with patch.object(calc.bm25, "calculate_weights", return_value={}):
        weights = calc.calculate([])
    assert weights == {}

def test_hybrid_weights_few_skills(calc):
    bm25 = {"a": 0.8, "b": 0.6}
    with patch.object(calc.bm25, "calculate_weights", return_value=bm25):
        weights = calc.calculate([])
    assert weights == {"a": 1.0, "b": 0.0}

def test_hybrid_weights_embedding_unavailable(calc):
    bm25 = {"a": 0.8, "b": 0.6}
    with patch.object(calc.bm25, "calculate_weights", return_value=bm25):
        with patch.object(calc.cache, '_model', None):  # убираем модель
            weights = calc.calculate([{"description": "test"}])
    assert weights == {"a": 1.0, "b": 0.0}

def test_hybrid_weights_embedding_exception(calc):
    """Если получение эмбеддингов вызывает исключение, fallback на BM25."""
    bm25 = {"a": 0.8, "b": 0.6}
    with patch.object(calc.bm25, "calculate_weights", return_value=bm25):
        with patch.object(calc.cache, "get_embeddings", side_effect=Exception("fail")):
            weights = calc.calculate([{"description": "test"}])
        assert weights == {"a": 1.0, "b": 0.0}

def test_hybrid_weights_with_pca(calc):
    """При большом количестве навыков и включенном PCA."""
    skills = {f"skill_{i}": 0.5 for i in range(150)}
    with patch.object(calc.bm25, "calculate_weights", return_value=skills):
        mock_embs = {f"skill_{i}": np.random.rand(384).astype(np.float32) for i in range(150)}
        with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
            # Включаем PCA
            with patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_ENABLED", True), \
                 patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_SAMPLES", 100), \
                 patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_MIN_FEATURES", 128), \
                 patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_TARGET_DIM", 64):
                weights = calc.calculate([{"description": "test"}])
            assert len(weights) > 0

def test_hybrid_weights_embedding_error(calc):
    bm25 = {"python": 0.8, "sql": 0.6}
    with patch.object(calc.bm25, "calculate_weights", return_value=bm25):
        with patch.object(calc.cache, "get_embeddings", side_effect=RuntimeError("fail")):
            weights = calc.calculate([])
    assert "python" in weights

def test_hybrid_weights_model_unavailable(calc):
    """Модель эмбеддингов недоступна -> чистый BM25 с minmax."""
    bm25 = {"a": 0.8, "b": 0.6}
    with patch.object(calc.bm25, "calculate_weights", return_value=bm25):
        # патчим model так, чтобы обращение к свойству model возвращало None
        with patch("src.parsing.skills.hybrid_weight_calculator.SkillEmbeddingCache.model", None):
            weights = calc.calculate([])
    assert weights == {"a": 1.0, "b": 0.0}

def test_hybrid_weights_embedding_exception_fallback(calc):
    bm25 = {"python": 0.9, "sql": 0.7}
    with patch.object(calc.bm25, "calculate_weights", return_value=bm25):
        with patch.object(calc.cache, "get_embeddings", side_effect=RuntimeError("fail")):
            weights = calc.calculate([])
    assert "python" in weights

def test_hybrid_weights_pca_not_enabled(calc):
    skills = {f"skill_{i}": 0.5 for i in range(50)}
    with patch.object(calc.bm25, "calculate_weights", return_value=skills):
        mock_embs = {f"skill_{i}": np.random.rand(384).astype(np.float32) for i in range(50)}
        with patch.object(calc.cache, "get_embeddings", return_value=mock_embs):
            with patch("src.parsing.skills.hybrid_weight_calculator.config.PCA_ENABLED", False):
                weights = calc.calculate([{"description": "test"}])
                assert len(weights) == 50
