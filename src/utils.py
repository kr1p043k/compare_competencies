import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src import Err, Ok, Result
from src.config import (
    BASE_DIR,
    COMPETENCY_MAPPING_FILE,
    LOG_FILE,
)
from src.errors import DomainError

logger = structlog.get_logger(__name__)


def skill_words(name: str) -> set[str]:
    return set(name.lower().replace("-", " ").split())


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


def load_competency_mapping() -> dict[str, list[str]]:
    if not COMPETENCY_MAPPING_FILE.exists():
        logger.warning("competency_mapping_file_not_found", path=str(COMPETENCY_MAPPING_FILE))
        return {}
    try:
        with open(COMPETENCY_MAPPING_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error("competency_mapping_load_failed", path=str(COMPETENCY_MAPPING_FILE), error=str(e))
        return {}


def load_competency_mapping_result() -> Result[dict[str, list[str]], DomainError]:
    if not COMPETENCY_MAPPING_FILE.exists():
        return Err(DomainError(message="Competency mapping file not found", detail=str(COMPETENCY_MAPPING_FILE)))
    try:
        with open(COMPETENCY_MAPPING_FILE, encoding="utf-8") as f:
            return Ok(json.load(f))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return Err(DomainError(message="Failed to load competency mapping", detail=str(e)))


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
        if not isinstance(data, (list, dict)):
            logger.error("invalid_json_structure", path=str(filepath), type=type(data).__name__)
            return None
        return data
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("json_read_error", path=str(filepath), error=str(e))
        return None
    except Exception as e:
        logger.error("json_read_unexpected_error", path=str(filepath), error=str(e))
        return None


def safe_read_json_result(filepath: Path) -> Result[list, DomainError]:
    if not filepath.exists():
        return Err(DomainError(message="File not found", detail=str(filepath)))
    if filepath.stat().st_size == 0:
        return Err(DomainError(message="Empty JSON file", detail=str(filepath)))
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return Err(DomainError(message="Invalid JSON structure, expected list", detail=f"got {type(data).__name__}"))
        return Ok(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return Err(DomainError(message="JSON read error", detail=str(e)))
    except Exception as e:
        return Err(DomainError(message="Unexpected JSON read error", detail=str(e)))


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


def safe_read_competency_json_result(filepath: Path) -> Result[list[str], DomainError]:
    if not filepath.exists():
        return Err(DomainError(message="Competency file not found", detail=str(filepath)))
    if filepath.stat().st_size == 0:
        return Err(DomainError(message="Empty competency file", detail=str(filepath)))
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        codes = data.get("компетенции") or data.get("навыки") or data.get("codes") or []
        if not isinstance(codes, list):
            return Err(DomainError(message="Invalid competency structure", detail=str(filepath)))
        return Ok(codes)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return Err(DomainError(message="Competency JSON read error", detail=str(e)))
    except Exception as e:
        return Err(DomainError(message="Unexpected competency read error", detail=str(e)))


def validate_safe_path(user_path: str | Path, base_dir: Path | None = None) -> Path:
    if base_dir is None:
        base_dir = BASE_DIR
    resolved = (base_dir / user_path).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ValueError(f"Путь '{user_path}' выходит за пределы разрешённой директории")
    return resolved


def validate_safe_path_result(user_path: str | Path, base_dir: Path | None = None) -> Result[Path, DomainError]:
    if base_dir is None:
        base_dir = BASE_DIR
    try:
        resolved = (base_dir / user_path).resolve()
        if not str(resolved).startswith(str(base_dir.resolve())):
            return Err(DomainError(message=f"Path '{user_path}' outside allowed directory"))
        return Ok(resolved)
    except Exception as e:
        return Err(DomainError(message="Path validation failed", detail=str(e)))


def extract_date_from_filename(filepath: Path) -> datetime | None:
    """Извлекает дату из имени freq_*.json.
    
    Поддерживает форматы:
      freq_2026-04-15.json         -> 2026-04-15
      freq_2026-04-15-120000.json  -> 2026-04-15 12:00:00
      freq_2026-04-15 120000.json  -> 2026-04-15 12:00:00
    Возвращает None если дату не удалось распарсить.
    """
    stem = filepath.stem.replace("freq_", "")
    formats = ["%Y-%m-%d-%H%M%S", "%Y-%m-%d", "%Y-%m-%d %H%M%S"]
    for fmt in formats:
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    return None


def build_inverted_skill_index(mapping: dict[str, list[str]]) -> dict[str, set[str]]:
    """Строит inverted index: skill_keyword -> set(competency_codes).
    
    Из mapping {competency_code: [keyword1, keyword2, ...]} делает
    {keyword1: {comp_code}, keyword2: {comp_code}, ...}
    """
    idx: dict[str, set[str]] = defaultdict(set)
    for comp_code, keywords in mapping.items():
        for kw in keywords:
            kw_norm = kw.lower().strip()
            idx[kw_norm].add(comp_code)
    return dict(idx)


def load_inverted_skill_index() -> dict[str, set[str]]:
    """Загружает оба mapping-файла и возвращает объединённый inverted index."""
    from src.config import (
        KRM_MAPPING_PATH,
    )
    idx: dict[str, set[str]] = defaultdict(set)

    for path in [COMPETENCY_MAPPING_FILE, KRM_MAPPING_PATH]:
        try:
            if path and path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                for comp_code, keywords in data.items():
                    for kw in keywords:
                        kw_norm = kw.lower().strip()
                        idx[kw_norm].add(comp_code)
        except Exception:
            logger.warning("inverted_index_load_failed", path=str(path))
    return dict(idx)


def _market_freq_lookup(
    skill_name: str,
    freq_map: dict[str, int],
    inverted_index: dict[str, set[str]] | None = None,
    mapping: dict[str, list[str]] | None = None,
    _norm_cache: dict[str, str] | None = None,
) -> int:
    """4-level lookup chain for skill name -> market frequency.

    Level 1: direct key lookup
    Level 2: SkillNormalizer.normalize() -> lookup
    Level 3: fuzzy match (rapidfuzz WRatio >= 85)
    Level 4: bridging through competency_mapping inverted index
    """
    # Level 1
    v = freq_map.get(skill_name, 0) or 0
    if v:
        return v

    # Level 2
    from src.parsing.skills.skill_normalizer import SkillNormalizer
    match SkillNormalizer.normalize(skill_name):
        case Ok(norm) if norm != skill_name:
            v = freq_map.get(norm, 0) or 0
            if v:
                return v
            skill_name = norm
        case _:
            pass

    # Level 3
    try:
        from rapidfuzz import process as rp_process
        from rapidfuzz import fuzz as rp_fuzz
        matches = rp_process.extract(skill_name, list(freq_map.keys()), scorer=rp_fuzz.WRatio, limit=1)
        if matches and matches[0][1] >= 85:
            return freq_map[matches[0][0]]
    except ImportError:
        pass

    # Level 4: bridging through mapping inverted index
    if inverted_index and mapping and skill_name.lower().strip() in inverted_index:
        comp_codes = inverted_index[skill_name.lower().strip()]
        for comp_code in comp_codes:
            keywords = mapping.get(comp_code, [])
            for kw in keywords:
                v = freq_map.get(kw, 0) or 0
                if v:
                    return v
                # try normalized
                match SkillNormalizer.normalize(kw):
                    case Ok(norm):
                        v = freq_map.get(norm, 0) or 0
                        if v:
                            return v

    return 0

