"""Тесты dataset versioning."""
from pathlib import Path
from src.pipeline.versioning import compute_file_hash, version_dataset


class TestVersioning:
    def test_compute_file_hash(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"a": 1}', encoding="utf-8")
        h = compute_file_hash(f)
        assert len(h) == 16
        assert isinstance(h, str)

    def test_different_files_different_hash(self, tmp_path):
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text("hello", encoding="utf-8")
        f2.write_text("world", encoding="utf-8")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_version_dataset(self, tmp_path):
        f = tmp_path / "vacancies.json"
        f.write_text('[{"id": 1}, {"id": 2}]', encoding="utf-8")
        result = version_dataset(f, name="vacancies", rows=2)
        assert result.is_ok()
        v = result.unwrap()
        assert v.name == "vacancies"
        assert v.rows == 2
        assert v.hash is not None

    def test_version_nonexistent_file(self, tmp_path):
        result = version_dataset(tmp_path / "nope.json")
        assert result.is_err()

    def test_version_default_name(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("x", encoding="utf-8")
        result = version_dataset(f)
        assert result.is_ok()
        assert result.unwrap().name == "data"
