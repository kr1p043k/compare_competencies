import numpy as np
import json
from unittest.mock import patch, MagicMock
from src.parsing.skills.skill_embedding_cache import SkillEmbeddingCache

def test_cache_compute_new_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    cache._model = MagicMock()
    cache._model.encode.return_value = np.random.rand(2, 384)
    with patch.object(cache, "_model_version", return_value="test"):
        embs = cache.get_embeddings(["python", "sql"])
    assert len(embs) == 2
    assert "python" in embs
    cache._model.encode.assert_called_once()

def test_cache_compute_new(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    cache._model = MagicMock()
    cache._model.encode.return_value = np.random.rand(2, 384)
    with patch.object(cache, "_model_version", return_value="test"):
        embs = cache.get_embeddings(["python", "sql"])
    assert len(embs) == 2
    assert "python" in embs

def test_cache_load_existing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    # Предварительно создаём кэш
    cache_path = tmp_path / "skill_embeddings.json"
    data = {"model_version": "test", "embeddings": {"python": [0.1, 0.2]}}
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    cache._model = MagicMock()
    with patch.object(cache, "_model_version", return_value="test"):
        embs = cache.get_embeddings(["python"])
        assert "python" in embs
        cache._model.encode.assert_not_called()

def test_cache_version_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    # Создаём старый кэш с другой версией модели
    cache_path = tmp_path / "skill_embeddings.json"
    data = {"model_version": "old_version", "embeddings": {"python": [0.1, 0.2]}}
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    cache._model = MagicMock()
    cache._model.encode.return_value = np.array([[0.3, 0.4]])
    with patch.object(cache, "_model_version", return_value="new_version"):
        embs = cache.get_embeddings(["python"])
        # Должен пересчитать, так как версии не совпадают
        cache._model.encode.assert_called_once()
        assert "python" in embs

def test_cache_version_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    cache._model = MagicMock()
    cache._model.encode.return_value = np.random.rand(1, 384)

    cache_path = tmp_path / "skill_embeddings.json"
    cache_path.write_text('{"model_version": "old", "embeddings": {"python": [0.1]}}', encoding="utf-8")

    with patch.object(cache, "_model_version", return_value="new"):
        embs = cache.get_embeddings(["python"])
    assert "python" in embs
    cache._model.encode.assert_called_once()

def test_cache_load_corrupted_json(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    cache._model = MagicMock()
    cache._model.encode.return_value = np.random.rand(1, 384)

    cache_path = tmp_path / "skill_embeddings.json"
    cache_path.write_text("{corrupted", encoding="utf-8")

    with patch.object(cache, "_model_version", return_value="test"):
        embs = cache.get_embeddings(["python"])
    assert "python" in embs
    cache._model.encode.assert_called_once()

def test_cache_missing_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    cache._model = MagicMock()
    cache._model.encode.return_value = np.random.rand(1, 384)
    with patch.object(cache, "_model_version", return_value="test"):
        embs = cache.get_embeddings(["python"])
    assert "python" in embs
    cache._model.encode.assert_called_once()

def test_cache_load_version_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    cache._model = MagicMock()
    cache._model.encode.return_value = np.random.rand(1, 384)
    cache_path = tmp_path / "skill_embeddings.json"
    cache_path.write_text('{"model_version":"old","embeddings":{"python":[0.1]}}', encoding="utf-8")
    with patch.object(cache, "_model_version", return_value="new"):
        embs = cache.get_embeddings(["python"])
    assert "python" in embs
    cache._model.encode.assert_called_once()

def test_cache_model_property_loads_model(tmp_path, monkeypatch):
    """Строки 33-34: свойство model загружает модель"""
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    assert cache._model is None
    # Вызов свойства
    model = cache.model
    assert model is not None
    # Повторный вызов не создаёт новую
    model2 = cache.model
    assert model is model2

def test_cache_model_version_determination():
    """Строка 38: определение версии модели при отсутствии sentence_transformers"""
    cache = SkillEmbeddingCache()
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        version = cache._model_version()
        assert "unknown" in version

def test_cache_model_property_loads_model(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.skills.skill_embedding_cache.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache = SkillEmbeddingCache()
    assert cache._model is None
    model = cache.model
    assert model is not None
    model2 = cache.model
    assert model is model2
