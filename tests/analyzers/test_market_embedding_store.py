"""Общий стор маркет-эмбеддингов: хэш данных, subset-reuse, delta-merge."""

from unittest.mock import MagicMock

import numpy as np

from src.analyzers.comparison.market_embedding_store import (
    market_data_hash,
    merge_matrix,
    split_cached,
)


def test_data_hash_stable_and_order_insensitive():
    assert market_data_hash(["b", "a"]) == market_data_hash(["a", "b"])
    assert market_data_hash(["a"]) != market_data_hash(["b"])


def test_split_cached_subset_and_missing():
    have, mat, missing = split_cached(
        ["a", "b", "c"], np.eye(3, dtype=np.float32), ["b", "d"])
    assert have == ["b"] and missing == ["d"]
    np.testing.assert_array_equal(mat, np.array([[0, 1, 0]], dtype=np.float32))


def test_merge_matrix_order():
    order, mat = merge_matrix(
        ["b"], np.array([[0, 1]], dtype=np.float32),
        ["d"], np.array([[0, 0]], dtype=np.float32))
    assert order == ["b", "d"] and mat.shape == (2, 2)


def _mock_provider(dim=4):
    prov = MagicMock()
    prov.encode.side_effect = lambda names, **kw: np.stack([
        np.full(dim, float(i + 1), dtype=np.float32) for i, _ in enumerate(names)])
    return prov


def test_matcher_delta_merge_encodes_only_missing(tmp_path, monkeypatch):
    from src.analyzers import skill_matcher as sm
    from src.analyzers.skill_matcher import SkillMatcher

    monkeypatch.setattr(sm, "MARKET_CACHE_MIN_SKILLS", 0)
    monkeypatch.setattr("src.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    prov = _mock_provider()
    m = SkillMatcher({}, embedding_provider=prov)
    assert m.set_market({"alpha": 5, "beta": 3}).is_ok()
    assert prov.encode.call_count == 1
    # Тот же маркет + один новый: кодируется только новый.
    assert m.set_market({"alpha": 5, "beta": 3, "gamma": 1}).is_ok()
    last_call = prov.encode.call_args_list[-1]
    assert list(last_call.args[0]) == ["gamma"]
    assert m._market_names == ["alpha", "beta", "gamma"]
    assert m._market_embeddings.shape == (3, 4)


def test_matcher_junk_names_not_indexed(tmp_path, monkeypatch):
    """Призраки ('я') чистятся из маркета и НЕ попадают в семантический
    индекс (2026-10-12: set_market кодировал сырые ключи)."""
    from src.analyzers import skill_matcher as sm
    from src.analyzers.skill_matcher import SkillMatcher

    monkeypatch.setattr(sm, "MARKET_CACHE_MIN_SKILLS", 0)
    monkeypatch.setattr("src.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    prov = _mock_provider()
    m = SkillMatcher({}, embedding_provider=prov)
    assert m.set_market({"я": 99, "python": 10}).is_ok()
    assert list(m.market_skills) == ["python"]
    assert list(m._market_names) == ["python"]


def test_comparator_reuses_matcher_file_delta(tmp_path):
    from src.analyzers.comparison.embedding_comparator import EmbeddingComparator

    comp = EmbeddingComparator(cache_dir=str(tmp_path))
    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 4
    fake_model.encode.side_effect = lambda skills, **kw: np.stack([
        np.full(4, float(abs(hash(s)) % 97) + 1.0) for s in skills])
    comp.model = fake_model
    comp.provider = fake_model
    comp.build_market_index(["python", "java"], level="all")
    n_calls = fake_model.encode.call_count
    # Расширение набора: докодируется только java2... точнее только новое.
    comp2 = EmbeddingComparator(cache_dir=str(tmp_path))
    comp2.model = fake_model
    comp2.provider = fake_model
    comp2.build_market_index(["python", "java", "sql"], level="all")
    only_new = [c.args[0] for c in fake_model.encode.call_args_list[n_calls:]]
    assert only_new == [["sql"]], only_new
    assert comp2.market_skills == ["python", "java", "sql"]
