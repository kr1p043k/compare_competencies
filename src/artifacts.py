"""
Универсальный манифест для артефактов:
- сохраняет метаинформацию рядом с файлом модели / эмбеддингов / кластеров
- проверяет актуальность при загрузке (data_hash, model_version)

Формат имени файла:
  some_model.joblib           → some_model.manifest.json
  market_embeddings_middle.pkl → market_embeddings_middle.manifest.json
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src import config
from src.utils import atomic_write_json

logger = structlog.get_logger(__name__)


class ArtifactManifest:
    """
    Сопровождает артефакт JSON‑файлом с тем же именем, но с расширением .manifest.json.
    """

    def __init__(
        self,
        artifact_path: Path,
        data_hash: str | None = None,
        model_version: str | None = None,
        metrics: dict[str, float] | None = None,
    ):
        self.artifact_path = Path(artifact_path)
        self.manifest_path = self.artifact_path.with_suffix(".manifest.json")
        self.data_hash = data_hash
        self.model_version = model_version or self._get_embedding_model_version()
        self.metrics = metrics or {}
        self.created_at = datetime.now(UTC).isoformat()

    # ------------------------------------------------------------------
    @staticmethod
    def _get_embedding_model_version() -> str:
        """Возвращает строку, идентифицирующую текущую модель эмбеддингов."""
        try:
            import sentence_transformers

            lib_ver = sentence_transformers.__version__
        except ImportError:
            lib_ver = "unknown"
        return f"{config.EMBEDDING_MODEL}_st{lib_ver}"

    @staticmethod
    def compute_data_hash(file_path: Path) -> str:
        """SHA256‑хеш содержимого файла (для вакансий)."""
        import hashlib

        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": str(self.artifact_path.name),
            "data_hash": self.data_hash,
            "model_version": self.model_version,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }

    def save(self) -> None:
        """Сохраняет манифест рядом с артефактом."""
        atomic_write_json(self.to_dict(), self.manifest_path)
        logger.info("artifact_manifest_saved", path=str(self.manifest_path))

    @classmethod
    def load(cls, artifact_path: Path) -> "ArtifactManifest":
        """Загружает манифест из .manifest.json."""
        manifest_path = artifact_path.with_suffix(".manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            artifact_path=artifact_path,
            data_hash=data.get("data_hash"),
            model_version=data.get("model_version"),
            metrics=data.get("metrics"),
        )

    def is_compatible(self) -> bool:
        """Проверяет, совпадает ли версия модели эмбеддингов с текущей."""
        current = self._get_embedding_model_version()
        if self.model_version != current:
            logger.warning(
                "manifest_model_mismatch",
                manifest_version=self.model_version,
                current_version=current,
            )
            return False
        return True
