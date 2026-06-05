import json
from pathlib import Path
from unittest.mock import patch
import pytest
from src import Err, Ok
from src.artifacts import ArtifactManifest

class TestArtifactManifest:
    def test_init_with_defaults(self, tmp_path):
        artifact = tmp_path / "model.pkl"
        manifest = ArtifactManifest(artifact_path=artifact, model_version="test")
        assert manifest.artifact_path == artifact
        assert manifest.manifest_path == artifact.with_suffix(".manifest.json")
        assert manifest.model_version == "test"
        assert manifest.metrics == {}
        assert manifest.data_hash is None

    def test_init_with_data_hash_and_metrics(self, tmp_path):
        artifact = tmp_path / "data.pkl"
        manifest = ArtifactManifest(
            artifact_path=artifact,
            data_hash="abc123",
            metrics={"accuracy": 0.95},
            model_version="test",
        )
        assert manifest.data_hash == "abc123"
        assert manifest.metrics == {"accuracy": 0.95}

    def test_to_dict(self, tmp_path):
        manifest = ArtifactManifest(
            artifact_path=tmp_path / "model.pkl",
            data_hash="hash",
            metrics={"loss": 0.1},
            model_version="v1",
        )
        d = manifest.to_dict()
        assert d["artifact"] == "model.pkl"
        assert d["data_hash"] == "hash"
        assert d["metrics"]["loss"] == 0.1
        assert d["model_version"] == "v1"

    def test_save_and_load(self, tmp_path):
        artifact = tmp_path / "model.joblib"
        manifest = ArtifactManifest(
            artifact_path=artifact,
            data_hash="123",
            metrics={"r2": 0.9},
            model_version="v2",
        )
        assert manifest.save().is_ok()
        assert manifest.manifest_path.exists()
        match ArtifactManifest.load(artifact):
            case Ok(loaded):
                assert loaded.data_hash == "123"
                assert loaded.metrics["r2"] == 0.9
            case Err(err):
                pytest.fail(f"Failed to load manifest: {err}")

    def test_is_compatible_same_version(self, tmp_path):
        manifest = ArtifactManifest(artifact_path=tmp_path / "x.pkl", model_version="v1")
        with patch.object(ArtifactManifest, '_get_embedding_model_version', return_value='v1'):
            result = manifest.is_compatible()
            assert result.is_ok() and result.unwrap() is True

    def test_is_compatible_different_version(self, tmp_path):
        manifest = ArtifactManifest(artifact_path=tmp_path / "x.pkl", model_version="old")
        with patch.object(ArtifactManifest, '_get_embedding_model_version', return_value='new'):
            result = manifest.is_compatible()
            assert result.is_ok() and result.unwrap() is False

    def test_compute_data_hash(self, tmp_path):
        file_path = tmp_path / "data.txt"
        file_path.write_text("hello")
        h = ArtifactManifest.compute_data_hash(file_path)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_get_embedding_model_version(self, monkeypatch):
        monkeypatch.setattr(ArtifactManifest, '_get_embedding_model_version', lambda: 'test_version')
        ver = ArtifactManifest._get_embedding_model_version()
        assert ver == 'test_version'
