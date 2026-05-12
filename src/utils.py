import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import structlog

from src.config import COMPETENCY_MAPPING_FILE, LOG_FILE

logger = structlog.get_logger(__name__)


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


def safe_read_json(filepath: Path):
    """Безопасно читает JSON-файл с проверкой размера, кодировки и структуры."""
    if not filepath.exists():
        return None
    if filepath.stat().st_size == 0:
        logger.error("empty_json_file", path=str(filepath))
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("invalid_json_structure", path=str(filepath), type=type(data).__name__)
            return None
        return data
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("json_read_error", path=str(filepath), error=str(e))
        return None
    except Exception as e:
        logger.error("json_read_unexpected_error", path=str(filepath), error=str(e))
        return None


def safe_read_competency_json(filepath: Path) -> list[str]:
    """
    Безопасно читает JSON-файл компетенций студента.
    Ожидает ключ 'компетенции', 'навыки' или 'codes'.
    Возвращает список строк или пустой список при ошибке.
    """
    if not filepath.exists():
        return []
    if filepath.stat().st_size == 0:
        logger.error("empty_competency_file", path=str(filepath))
        return []
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        codes = data.get("компетенции") or data.get("навыки") or data.get("codes") or []
        if not isinstance(codes, list):
            logger.error("invalid_competency_structure", path=str(filepath))
            return []
        return codes
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("competency_json_read_error", path=str(filepath), error=str(e))
        return []
    except Exception as e:
        logger.error("competency_read_unexpected_error", path=str(filepath), error=str(e))
        return []
