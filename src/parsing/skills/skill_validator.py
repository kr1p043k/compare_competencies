"""
Типизированный валидатор навыков.
Отвечает ТОЛЬКО за валидацию, не за извлечение!
Исправленная версия с улучшенной фильтрацией и логированием.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from src import Result, Ok, Err
from src import config
from src.errors import DomainError
from src.parsing.utils import load_it_skills

logger = structlog.get_logger(__name__)


class ValidationReason(Enum):
    """Причины отклонения навыка"""

    TOO_SHORT = "Слишком короткий (<3 символов)"
    TOO_LONG = "Слишком длинный (>4 слов или >50 символов)"
    IN_BLACKLIST = "В чёрном списке"
    LOW_CONFIDENCE = "Низкая уверенность"
    NOT_IN_WHITELIST = "Нет в белом списке"
    EMPTY = "Пустой"
    ONLY_DIGITS = "Только цифры"
    ONLY_SPECIAL = "Только спецсимволы"
    GENERIC_WORD = "Общее слово (frontend/backend и т.д.)"
    VALID = "Валиден"


@dataclass
class ValidationResult:
    """Результат валидации навыка"""

    skill: str
    is_valid: bool
    reasons: list[ValidationReason] = None
    confidence: float = 0.0
    normalized_skill: str = ""

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []

    def __repr__(self):
        if self.is_valid:
            return f"✓ '{self.skill}' (confidence: {self.confidence:.2f})"
        else:
            reasons_str = "; ".join(r.value for r in self.reasons)
            return f"✗ '{self.skill}': {reasons_str}"


class SkillValidator:
    """
    Валидатор для проверки извлечённых навыков.
    Улучшенная версия с расширенными проверками и настраиваемыми порогами.
    """

    # Паттерны для проверки
    DIGITS_PATTERN = re.compile(r"^\d+$")
    SPECIAL_PATTERN = re.compile(r"^[^\w\s]+$")

    def __init__(
        self,
        blacklist: set[str] | None = None,
        whitelist: set[str] | None = None,
        min_length: int = 3,
        max_length: int = 50,
        max_words: int = 6,
        min_confidence: float = 0.5,
        remove_generic: bool = True,
    ):
        """
        Args:
            blacklist: Набор запрещённых терминов (если None — загружается из data/reference/skill_blacklist.json)
            whitelist: Набор разрешённых навыков (если None, используется load_it_skills())
            min_length: Минимальная длина навыка (символов)
            max_length: Максимальная длина навыка (символов)
            max_words: Максимальное количество слов в навыке
            min_confidence: Минимальная уверенность для валидации
            remove_generic: Удалять ли generic слова
        """
        # Загружаем чёрный список
        if blacklist is not None:
            self.blacklist = blacklist
        else:
            blacklist_path = config.SKILL_BLACKLIST_PATH
            if blacklist_path.exists():
                self.blacklist = set(json.loads(blacklist_path.read_text(encoding="utf-8")))
                logger.info("blacklist_loaded", count=len(self.blacklist))
            else:
                logger.warning("blacklist_file_not_found", path=str(blacklist_path))
                self.blacklist = set()

        # Загружаем generic-слова
        if remove_generic:
            generic_path = config.GENERIC_WORDS_PATH
            if generic_path.exists():
                self.generic_words = set(json.loads(generic_path.read_text(encoding="utf-8")))
                logger.info("generic_words_loaded", count=len(self.generic_words))
            else:
                logger.warning("generic_words_file_not_found", path=str(generic_path))
                self.generic_words = set()
        else:
            self.generic_words = set()

        # Загружаем filler-слова
        filler_path = config.FILLER_WORDS_PATH
        if filler_path.exists():
            self.filler_words = set(json.loads(filler_path.read_text(encoding="utf-8")))
            logger.info("filler_words_loaded", count=len(self.filler_words))
        else:
            logger.warning("filler_words_file_not_found", path=str(filler_path))
            self.filler_words = set()

        # Белый список
        self.whitelist = whitelist if whitelist is not None else load_it_skills()
        self.min_length = min_length
        self.max_length = max_length
        self.max_words = max_words
        self.min_confidence = min_confidence

        # Статистика
        self.stats = {"total": 0, "valid": 0, "rejected": 0, "reasons": Counter()}

    def validate(self, skill: str, confidence: float = 1.0) -> Result[ValidationResult, DomainError]:
        """
        Валидирует навык с расширенными проверками

        Args:
            skill: Навык для проверки
            confidence: Уверенность в извлечении (0-1)

        Returns:
            ValidationResult с результатами
        """
        try:
            self.stats["total"] += 1
            reasons = []

            if not skill or not skill.strip():
                self.stats["rejected"] += 1
                self.stats["reasons"]["empty"] += 1
                return Ok(ValidationResult(skill=skill, is_valid=False, reasons=[ValidationReason.EMPTY], confidence=0.0))

            skill = skill.strip()
            skill_lower = skill.lower()

            if confidence < self.min_confidence:
                reasons.append(ValidationReason.LOW_CONFIDENCE)

            if len(skill) < self.min_length:
                reasons.append(ValidationReason.TOO_SHORT)

            if len(skill) > self.max_length:
                reasons.append(ValidationReason.TOO_LONG)

            words = skill_lower.split()
            if len(words) > self.max_words:
                reasons.append(ValidationReason.TOO_LONG)

            if self.DIGITS_PATTERN.match(skill):
                reasons.append(ValidationReason.ONLY_DIGITS)

            if self.SPECIAL_PATTERN.match(skill):
                reasons.append(ValidationReason.ONLY_SPECIAL)

            if skill_lower in self.generic_words:
                reasons.append(ValidationReason.GENERIC_WORD)

            for bad in self.blacklist:
                if bad in skill_lower:
                    reasons.append(ValidationReason.IN_BLACKLIST)
                    break

            if all(word in self.filler_words for word in words):
                reasons.append(ValidationReason.IN_BLACKLIST)

            if self.whitelist is not None:
                skill_normalized = skill_lower.replace(" ", "").replace("-", "").replace("_", "")
                if skill_normalized not in self.whitelist:
                    found_in_whitelist = any(
                        skill_normalized in wl.replace(" ", "").replace("-", "").replace("_", "")
                        or wl.replace(" ", "").replace("-", "").replace("_", "") in skill_normalized
                        for wl in self.whitelist
                    )
                    if not found_in_whitelist:
                        reasons.append(ValidationReason.NOT_IN_WHITELIST)

            if not any(c.isalpha() for c in skill):
                reasons.append(ValidationReason.ONLY_DIGITS)

            is_valid = len(reasons) == 0

            confidence_score = min(1.0, confidence + 0.1) if is_valid else max(0.1, 1.0 - len(reasons) * 0.2)

            if is_valid:
                self.stats["valid"] += 1
            else:
                self.stats["rejected"] += 1
                for reason in reasons:
                    self.stats["reasons"][reason.value] += 1

            return Ok(ValidationResult(skill=skill, is_valid=is_valid, reasons=reasons, confidence=confidence_score))
        except Exception as e:
            return Err(DomainError(message=str(e), detail=f"validate({skill})"))

    def validate_batch(self, skills: list[str], confidences: list[float] | None = None) -> tuple:
        """
        Валидирует массив навыков

        Args:
            skills: Список навыков
            confidences: Список уверенностей (опционально)

        Returns:
            (valid_skills, results)
        """
        if confidences is None:
            confidences = [1.0] * len(skills)

        valid_skills = []
        results = []

        for skill, conf in zip(skills, confidences, strict=False):
            result = self.validate(skill, conf).ok()
            results.append(result)
            if result.is_valid:
                valid_skills.append(result.skill)

        return valid_skills, results

    def get_rejection_report(self, results: list[ValidationResult]) -> dict[str, Any]:
        """
        Возвращает отчёт об отклонениях
        """
        rejected = [r for r in results if not r.is_valid]

        rejection_counts = Counter()
        for result in rejected:
            for reason in result.reasons:
                rejection_counts[reason.value] += 1

        return {
            "total_validated": len(results),
            "valid": len(results) - len(rejected),
            "rejected": len(rejected),
            "rejection_reasons": dict(rejection_counts),
            "rejection_rate": len(rejected) / len(results) if results else 0,
        }

    def reset_stats(self):
        """Сбрасывает статистику"""
        self.stats = {"total": 0, "valid": 0, "rejected": 0, "reasons": Counter()}

    def get_stats(self) -> dict[str, Any]:
        """Возвращает текущую статистику"""
        return self.stats
