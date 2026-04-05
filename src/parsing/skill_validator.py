"""
Типизированный валидатор навыков.
Отвечает ТОЛЬКО за валидацию, не за извлечение!
"""

from dataclasses import dataclass
from typing import List, Set, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationReason(Enum):
    """Причины отклонения навыка"""
    TOO_SHORT = "Слишком короткий"
    TOO_LONG = "Слишком длинный (>4 слов)"
    IN_BLACKLIST = "В чёрном списке"
    NOT_IN_WHITELIST = "Нет в белом списке"
    EMPTY = "Пустой"
    VALID = "Валиден"


@dataclass
class ValidationResult:
    """Результат валидации навыка"""
    skill: str
    is_valid: bool
    reasons: List[ValidationReason] = None
    confidence: float = 0.0
    
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
    """Валидатор для проверки извлечённых навыков"""
    
    # ПОЛНЫЙ чёрный список (мусорные термины)
    DEFAULT_BLACKLIST = {
        # === SOFT SKILLS ===
        "оценка потребностей клиентов", "развитие ключевых клиентов", "проведение презентаций",
        "проведение переговоров", "подготовка коммерческих предложений", "навыки межличностного общения",
        "межличностное общение", "мотивация персонала", "стратегическое мышление", "аналитическое мышление",
        "командная работа", "клиентоориентированность", "ориентация на результат", "продвижение бренда",
        "развитие продаж", "традиционная розница", "собственная розница", "клиентами", "клиентам",
        "инициатива", "харизма", "многозадачность", "бухгалтерская отчетность", "бухгалтерский учет",
        
        # === ЯЗЫКИ И ОБРАЗОВАНИЕ ===
        "английского языка", "русского языка", "высшее образование", "знание английского",
        "английский язык", "русский язык", "язык программирования", "вербальные",
        
        # === ОБЩИЕ ФРАЗЫ (ФИЛЬТРЫ) ===
        "опыт работы", "умение", "желательно", "преимуществом", "плюсом",
        "знание", "владение", "требуется", "должен", "навык", "компетенция",
        
        # === БИЗНЕС-ТЕРМИНЫ ===
        "продажи", "маркетинг", "smm", "управление персоналом", "развитие продаж",
        "анализ продаж", "анализ рынка", "анализ цен", "контроль цен",
        "мониторинг рынка", "мониторинг цен", "построение воронки продаж",
        "управление проектами", "управление продуктом", "управление командой",
        "ведение переговоров", "проведение встреч", "организация встреч",
        
        # === СПЕЦИФИЧНЫЕ МУСОРНЫЕ ФРАЗЫ ===
        "пластичные смазки", "автомобильные перевозки",
        "организация клиентских мероприятий", "аналитика маркетплейсов",
        "контроль и анализ ценообразования","тестирование по"
        
        # === ОДИНОЧНЫЕ СЛОВА - SOFT SKILLS ===
        "лидерство", "лидер", "стратегия", "креативность", "инновация",
        "организованность", "ответственность", "надежность", "честность",
        "влиятельность", "уверенность", "боевой", "амбициозный",
        "дипломатичность", "пунктуальность", "пунктуален",
    }
    
    # Слова-паразиты (если ВСЕ слова из этого набора - отклоняем)
    FILLER_WORDS = {
        "обо", "это", "что", "как", "быть", "иметь", "делать", "сказать",
        "мочь", "должен", "хотеть", "знать", "работа", "работать", "служба",
        "обладать", "справляться", "справляться",
    }
    
    def __init__(
        self,
        blacklist: Optional[Set[str]] = None,
        whitelist: Optional[Set[str]] = None,
        min_length: int = 3,
        max_length: int = 100,
        max_words: int = 4
    ):
        """
        Args:
            blacklist: Набор запрещённых терминов
            whitelist: Набор разрешённых навыков (если None, не используется)
            min_length: Минимальная длина навыка (символов)
            max_length: Максимальная длина навыка (символов)
            max_words: Максимальное количество слов в навыке
        """
        self.blacklist = blacklist or self.DEFAULT_BLACKLIST
        self.whitelist = whitelist
        self.min_length = min_length
        self.max_length = max_length
        self.max_words = max_words
    
    def validate(self, skill: str) -> ValidationResult:
        """Валидирует навык с расширенными проверками"""
        reasons = []
        
        if not skill or not skill.strip():
            return ValidationResult(
                skill=skill,
                is_valid=False,
                reasons=[ValidationReason.EMPTY],
                confidence=0.0
            )
        
        skill = skill.strip()
        skill_lower = skill.lower()
        
        # === ПРОВЕРКА 1: Длина ===
        if len(skill) < self.min_length:
            reasons.append(ValidationReason.TOO_SHORT)
        
        if len(skill) > self.max_length:
            reasons.append(ValidationReason.TOO_LONG)
        
        # === ПРОВЕРКА 2: Количество слов ===
        words = skill_lower.split()
        if len(words) > self.max_words:
            reasons.append(ValidationReason.TOO_LONG)
        
        # === ПРОВЕРКА 3: Чёрный список (КРИТИЧЕСКАЯ) ===
        for bad in self.blacklist:
            if bad in skill_lower:
                reasons.append(ValidationReason.IN_BLACKLIST)
                break
        
        # === ПРОВЕРКА 4: Слова-паразиты ===
        if all(word in self.FILLER_WORDS for word in words):
            reasons.append(ValidationReason.IN_BLACKLIST)
        
        # === ПРОВЕРКА 5: Белый список ===
        if self.whitelist is not None:
            if skill_lower not in self.whitelist:
                found_in_whitelist = any(
                    skill_lower in wl.lower() or wl.lower() in skill_lower
                    for wl in self.whitelist
                )
                if not found_in_whitelist:
                    reasons.append(ValidationReason.NOT_IN_WHITELIST)
        
        is_valid = len(reasons) == 0
        confidence = 0.95 if is_valid else max(0.1, 1.0 - len(reasons) * 0.3)
        
        return ValidationResult(
            skill=skill,
            is_valid=is_valid,
            reasons=reasons,
            confidence=confidence
        )
    
    def validate_batch(self, skills: List[str]) -> tuple:
        """Валидирует массив навыков"""
        valid_skills = []
        results = []
        
        for skill in skills:
            result = self.validate(skill)
            results.append(result)
            if result.is_valid:
                valid_skills.append(result.skill)
        
        return valid_skills, results
    
    def get_rejection_report(self, results: List[ValidationResult]) -> dict:
        """Возвращает отчёт об отклонениях"""
        rejected = [r for r in results if not r.is_valid]
        
        rejection_counts = {}
        for result in rejected:
            for reason in result.reasons:
                key = reason.value
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
        
        return {
            'total_validated': len(results),
            'valid': len(results) - len(rejected),
            'rejected': len(rejected),
            'rejection_reasons': rejection_counts
        }