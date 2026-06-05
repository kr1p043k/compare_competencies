"""Tests for FAISS vector search."""

import numpy as np
import pytest

from src.vector_search.faiss_index import FaissIndex


class TestFaissIndex:
    def test_build_and_search(self):
        embeddings = {
            "python": np.random.rand(128).astype(np.float32),
            "sql": np.random.rand(128).astype(np.float32),
            "docker": np.random.rand(128).astype(np.float32),
        }
        idx = FaissIndex(dim=128)
        result = idx.build(embeddings)
        assert result.is_ok()
        assert idx.is_built

        query = np.random.rand(128).astype(np.float32)
        search_result = idx.search(query, top_k=2)
        assert search_result.is_ok()
        results = search_result.ok()
        assert len(results) == 2
        for skill, score in results:
            assert skill in embeddings
            assert 0.0 <= score <= 1.0

    def test_build_empty_embeddings(self):
        idx = FaissIndex()
        result = idx.build({})
        assert result.is_err()

    def test_search_before_build(self):
        idx = FaissIndex()
        result = idx.search(np.random.rand(128).astype(np.float32))
        assert result.is_err()

    def test_save_and_load(self):
        import tempfile, os
        embeddings = {
            "python": np.random.rand(128).astype(np.float32),
            "sql": np.random.rand(128).astype(np.float32),
        }
        idx = FaissIndex(dim=128)
        idx.build(embeddings)
        d = tempfile.mkdtemp()
        path = os.path.join(d, "faiss.index")
        save_result = idx.save(path)
        assert save_result.is_ok()
        assert os.path.exists(path)

        loaded = FaissIndex()
        load_result = loaded.load(path, list(embeddings.keys()))
        assert load_result.is_ok()
        assert loaded.is_built
        assert loaded.dim == 128

    def test_save_no_index(self):
        idx = FaissIndex()
        result = idx.save("/tmp/nonexistent.index")
        assert result.is_err()

    def test_hnsw_index(self):
        embeddings = {f"skill_{i}": np.random.rand(64).astype(np.float32) for i in range(20)}
        idx = FaissIndex(dim=64, index_type="hnsw")
        result = idx.build(embeddings)
        assert result.is_ok()

        query = np.random.rand(64).astype(np.float32)
        search_result = idx.search(query, top_k=5)
        assert search_result.is_ok()
        assert len(search_result.ok()) == 5
