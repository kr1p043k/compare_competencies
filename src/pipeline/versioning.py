"""Dataset versioning — хеширование и метаданные для датасетов."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

from src import Err, Ok, Result
from src.errors import DomainError

logger = structlog.get_logger(__name__)


@dataclass
class DatasetVersion:
    name: str
    hash: str
    rows: int
    path: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hash": self.hash,
            "rows": self.rows,
            "path": self.path,
            "created_at": self.created_at,
        }


def compute_file_hash(path: Path) -> str:
    import hashlib
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def version_dataset(
    path: Path,
    name: str | None = None,
    rows: int | None = None,
) -> Result[DatasetVersion, DomainError]:
    if not path.exists():
        return Err(DomainError(message=f"dataset not found: {path}"))
    try:
        file_hash = compute_file_hash(path)
        version = DatasetVersion(
            name=name or path.stem,
            hash=file_hash,
            rows=rows or 0,
            path=str(path),
            created_at=datetime.now(UTC).isoformat(),
        )
        logger.info("dataset_versioned", name=version.name, hash=file_hash, rows=rows)
        return Ok(version)
    except Exception as e:
        return Err(DomainError(message=f"versioning failed: {e}"))
