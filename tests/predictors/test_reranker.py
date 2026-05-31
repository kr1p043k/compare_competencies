from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import Err, Ok
from src.errors import DomainError
from src.predictors.reranker import RerankerResult, CrossEncoderReranker


class TestRerankerResult:
    def test_init(self):
        r = RerankerResult(query="q", documents=["a", "b"], scores=[0.9, 0.1], ranked_indices=[0, 1])
        assert r.query == "q"
        assert r.documents == ["a", "b"]

    def test_top_k(self):
        r = RerankerResult(query="q", documents=["a", "b", "c"], scores=[0.3, 0.9, 0.6], ranked_indices=[1, 2, 0])
        top = r.top_k(2)
        assert top == [("b", 0.9), ("c", 0.6)]

    def test_top_k_default(self):
        docs = [f"d{i}" for i in range(20)]
        r = RerankerResult(query="q", documents=docs, scores=list(range(20, 0, -1)), ranked_indices=list(range(20)))
        top = r.top_k()
        assert len(top) == 10


class TestCrossEncoderRerankerInit:
    def test_default_params(self):
        r = CrossEncoderReranker()
        assert r.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert r.max_length == 512
        assert r.batch_size == 32
        assert r._model is None

    def test_custom_params(self):
        r = CrossEncoderReranker(model_name="custom", max_length=128, batch_size=16)
        assert r.model_name == "custom"

    def test_name(self):
        r = CrossEncoderReranker(model_name="my-model")
        assert "my-model" in r.name


class TestCrossEncoderRerankerLazyLoad:
    def test_already_loaded(self):
        r = CrossEncoderReranker()
        r._model = MagicMock()
        result = r._lazy_load()
        assert result.is_ok()

    @patch("sentence_transformers.CrossEncoder")
    def test_load_success(self, MockCE):
        r = CrossEncoderReranker()
        result = r._lazy_load()
        assert result.is_ok()
        assert r._model is not None


class TestCrossEncoderRerankerRerank:
    def test_empty_documents(self):
        r = CrossEncoderReranker()
        r._model = MagicMock()
        result = r.rerank("q", [])
        assert result.is_ok()
        assert result.ok().documents == []

    def test_lazy_load_failure(self):
        r = CrossEncoderReranker()
        r._lazy_load = MagicMock(return_value=Err(DomainError("model not available")))
        result = r.rerank("q", ["doc1"])
        assert result.is_err()

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_success(self, MockCE):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.5])
        MockCE.return_value = mock_model
        r = CrossEncoderReranker()
        r._lazy_load()
        result = r.rerank("python developer", ["java dev", "python expert", "c++ guru"])
        assert result.is_ok()
        rr = result.ok()
        assert len(rr.documents) == 3
        assert rr.ranked_indices[0] == 0

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_with_top_k(self, MockCE):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        MockCE.return_value = mock_model
        r = CrossEncoderReranker()
        r._lazy_load()
        result = r.rerank("query", ["a", "b", "c", "d", "e"], top_k=2)
        assert result.is_ok()
        assert len(result.ok().ranked_indices) == 2

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_nested_scores(self, MockCE):
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.9], [0.1], [0.5]]
        MockCE.return_value = mock_model
        r = CrossEncoderReranker()
        r._lazy_load()
        result = r.rerank("q", ["a", "b", "c"])
        assert result.is_ok()
        assert result.ok().scores[0] == 0.9

    def test_rerank_exception(self):
        r = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("OOM")
        r._model = mock_model
        result = r.rerank("q", ["a"])
        assert result.is_err()
