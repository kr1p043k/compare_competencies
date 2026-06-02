"""Тесты ModelRegistry."""
from pathlib import Path
from src.ml.registry import ModelRegistry, ModelRecord


class TestModelRegistry:
    def setup_method(self):
        self.registry = ModelRegistry()

    def test_register_and_get_latest(self):
        rec = ModelRecord(name="reranker", version="1.0", path=Path("/models/reranker.bin"))
        self.registry.register(rec)

        result = self.registry.get("reranker")
        assert result.is_ok()
        assert result.unwrap().version == "1.0"

    def test_get_by_version(self):
        self.registry.register(ModelRecord(name="reranker", version="1.0", path=Path("/m1.bin")))
        self.registry.register(ModelRecord(name="reranker", version="2.0", path=Path("/m2.bin")))

        result = self.registry.get("reranker", version="1.0")
        assert result.is_ok()
        assert result.unwrap().version == "1.0"

    def test_get_latest_returns_last(self):
        self.registry.register(ModelRecord(name="reranker", version="1.0", path=Path("/m1.bin")))
        self.registry.register(ModelRecord(name="reranker", version="2.0", path=Path("/m2.bin")))

        result = self.registry.get("reranker")
        assert result.unwrap().version == "2.0"

    def test_get_not_found(self):
        result = self.registry.get("nonexistent")
        assert result.is_err()

    def test_list_models(self):
        self.registry.register(ModelRecord(name="a", version="1", path=Path("/a")))
        self.registry.register(ModelRecord(name="b", version="1", path=Path("/b")))
        assert set(self.registry.list_models()) == {"a", "b"}

    def test_list_versions(self):
        self.registry.register(ModelRecord(name="x", version="1.0", path=Path("/x")))
        self.registry.register(ModelRecord(name="x", version="1.1", path=Path("/x")))
        assert self.registry.list_versions("x") == ["1.0", "1.1"]

    def test_count(self):
        self.registry.register(ModelRecord(name="a", version="1", path=Path("/a")))
        self.registry.register(ModelRecord(name="a", version="2", path=Path("/a")))
        self.registry.register(ModelRecord(name="b", version="1", path=Path("/b")))
        assert self.registry.count == 3
