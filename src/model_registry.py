"""Model Registry — version tracking for LTR, embeddings, clusters."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src import Err, Ok, Result
from src.errors import DomainError

logger = structlog.get_logger(__name__)


class ModelRegistry:
    def __init__(self, registry_dir: str | Path | None = None):
        from src import config

        self._dir = Path(registry_dir) if registry_dir else (config.MODELS_DIR / "registry")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "registry_index.json"
        self._index: dict[str, list[dict]] = {}
        self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning("registry_index_load_failed", error=str(e))
                self._index = {}

    def _save_index(self):
        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("registry_index_save_failed", error=str(e))

    def register(
        self,
        model_type: str,
        artifact_path: str | Path,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> Result[str, DomainError]:
        try:
            path = Path(artifact_path)
            if not path.exists():
                return Err(DomainError(message=f"Artifact not found: {path}"))

            entry = {
                "version": str(len(self._index.get(model_type, [])) + 1),
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "metrics": metrics or {},
                "tags": tags or {},
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }
            self._index.setdefault(model_type, []).append(entry)
            self._save_index()
            logger.info("model_registered", model_type=model_type, version=entry["version"])
            return Ok(entry["version"])
        except Exception as e:
            return Err(DomainError(message=str(e), detail="ModelRegistry.register"))

    def list_versions(self, model_type: str) -> list[dict]:
        return self._index.get(model_type, [])

    def latest(self, model_type: str) -> dict | None:
        versions = self._index.get(model_type, [])
        return versions[-1] if versions else None

    def get(self, model_type: str, version: str) -> dict | None:
        for v in self._index.get(model_type, []):
            if v["version"] == version:
                return v
        return None

    def prune(self, model_type: str, keep_last: int = 3) -> int:
        versions = self._index.get(model_type, [])
        if len(versions) <= keep_last:
            return 0
        to_remove = versions[:-keep_last]
        for entry in to_remove:
            try:
                Path(entry["path"]).unlink(missing_ok=True)
            except Exception as e:
                logger.warning("prune_remove_failed", path=entry["path"], error=str(e))
        self._index[model_type] = versions[-keep_last:]
        self._save_index()
        removed = len(to_remove)
        logger.info("model_registry_pruned", model_type=model_type, removed=removed)
        return removed

    def register_ltr(self, metrics: dict[str, Any] | None = None) -> Result[str, DomainError]:
        from src import config

        path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
        return self.register("ltr", path, metrics=metrics)

    def register_embeddings(self, level: str = "middle") -> Result[str, DomainError]:
        from src import config

        path = config.EMBEDDINGS_CACHE_DIR / f"market_embeddings_{level}.joblib"
        return self.register("embeddings", path, tags={"level": level})

    def register_clusters(self, level: str = "middle") -> Result[str, DomainError]:
        from src import config

        path = config.VACANCY_CLUSTERS_CACHE_DIR / f"vacancy_clusters_{level}.pkl"
        return self.register("clusters", path, tags={"level": level})
