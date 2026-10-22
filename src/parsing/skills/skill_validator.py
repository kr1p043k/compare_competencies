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

from src import Err, Ok, Result, config
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
                return Ok(ValidationResult(
                    skill=skill, is_valid=False, reasons=[ValidationReason.EMPTY], confidence=0.0
                ))

            skill = skill.strip()
            skill_lower = skill.lower()

            # Явный whitelist точнее эвристик: точное совпадение бьет
            # blacklist-подстроки ('автоматизация' в 'автоматизация тестирования')
            # и generic-ярлыки. Структурные проверки (длина/слова/цифры) действуют.
            # P-length: 2-символьные whitelist-навыки (qa/ui/ux/go/1c) минуют
            # TOO_SHORT — явная санкция сильнее эвристики длины.
            # 1-символьные ('r') всё равно режем: ненадёжный токен.
            skill_nospace = skill_lower.replace(" ", "").replace("-", "").replace("_", "")
            wl_norm = {
                w.lower().replace(" ", "").replace("-", "").replace("_", "")
                for w in (self.whitelist or set())
            }
            whitelisted_exact = bool(skill_nospace) and skill_nospace in wl_norm

            if confidence < self.min_confidence:
                reasons.append(ValidationReason.LOW_CONFIDENCE)

            if len(skill) < self.min_length and not (
                whitelisted_exact and len(skill) >= 2
            ):
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

            if not whitelisted_exact and skill_lower in self.generic_words:
                reasons.append(ValidationReason.GENERIC_WORD)

            if not whitelisted_exact:
                for bad in self.blacklist:
                    bad_low = str(bad).lower()
                    if bad_low.startswith("="):
                        # Точное совпадение (=nat режет только "nat",
                        # но щадит react native / hibernate)
                        if skill_lower == bad_low[1:]:
                            reasons.append(ValidationReason.IN_BLACKLIST)
                            break
                    elif bad_low in skill_lower:
                        reasons.append(ValidationReason.IN_BLACKLIST)
                        break

            if all(word in self.filler_words for word in words):
                reasons.append(ValidationReason.IN_BLACKLIST)

            if self.whitelist is not None:
                # Точки тоже нормализуем с обеих сторон ('nodejs' vs 'node.js').
                def _wl_norm(w: str) -> str:
                    return w.lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
                skill_normalized = _wl_norm(skill_lower)
                wl_norm = {_wl_norm(w) for w in self.whitelist}
                if skill_normalized not in wl_norm:
                    # Проза с вкраплением навыка — не навык ('i love python...').
                    # Короткие кандидаты (<=3 слов): сабстринг в обе стороны +
                    # слово целиком (дешёво и покрывает легитимные 'machine
                    # learning', 'nodejs'). Длинные: доля слов, покрытых
                    # фразами whitelist (>=0.5), иначе проза валидируется.
                    words_norm = [_wl_norm(w) for w in words]
                    if len(words) <= 3:
                        found_in_whitelist = any(
                            skill_normalized in wl or wl in skill_normalized
                            for wl in wl_norm
                        ) or any(w in wl_norm for w in words_norm)
                    else:
                        # Кэш фраз ради скорости (validate вызывается тысячами).
                        if self.__dict__.get("_wl_phrases") is None:
                            try:
                                from src.text.ru_morph import lemma as _lemma
                            except Exception:
                                _lemma = lambda w: w  # noqa: E731
                            self.__dict__["_wl_lemmatize"] = _lemma
                            # Фразы — по словам ( _wl_norm целиком склеил бы
                            # 'автоматизация тестирования' в один токен).
                            self.__dict__["_wl_phrases"] = [
                                tuple(_lemma(_wl_norm(w)) for w in str(x).split())
                                for x in (self.whitelist or set())
                            ]
                        _lemma = self.__dict__["_wl_lemmatize"]
                        words_lem = [_lemma(w) for w in words_norm]
                        covered = [False] * len(words_lem)
                        for ph in self.__dict__["_wl_phrases"]:
                            if not ph:
                                continue
                            for i in range(len(words_lem) - len(ph) + 1):
                                if tuple(words_lem[i:i + len(ph)]) == ph:
                                    for j in range(i, i + len(ph)):
                                        covered[j] = True
                        found_in_whitelist = (
                            sum(covered) / len(words_lem) >= 0.5
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
            if result is None:
                continue
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
