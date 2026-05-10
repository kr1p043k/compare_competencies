import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from src.config import COMPETENCY_MAPPING_FILE, LOG_FILE


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Возвращает логгер с указанным именем (устаревшая, используйте structlog)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_competency_mapping():
    with open(COMPETENCY_MAPPING_FILE, encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(data: Any, filepath: Path) -> None:
    """
    Атомарная запись JSON: пишет во временный файл, затем переименовывает.
    Защищает от битых файлов при падении процесса во время записи.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, filepath)  # атомарно на POSIX, почти атомарно на Windows
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_read_json(filepath: Path) -> Any:
    """
    Безопасное чтение JSON: если файл битый — возвращает None.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
