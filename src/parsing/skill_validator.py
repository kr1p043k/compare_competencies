"""
Типизированный валидатор навыков.
Отвечает ТОЛЬКО за валидацию, не за извлечение!
Исправленная версия с улучшенной фильтрацией и логированием.
"""

from dataclasses import dataclass
from typing import List, Set, Optional, Dict, Any
from enum import Enum
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


class ValidationReason(Enum):
    """Причины отклонения навыка"""
    TOO_SHORT = "Слишком короткий (<3 символов)"
    TOO_LONG = "Слишком длинный (>4 слов или >50 символов)"
    IN_BLACKLIST = "В чёрном списке"
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
    reasons: List[ValidationReason] = None
    confidence: float = 0.0
    normalized_skill: str = ""  # Нормализованная версия
    
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
    
    # ПОЛНЫЙ чёрный список (мусорные термины)
    DEFAULT_BLACKLIST = {
        # === SOFT SKILLS (полностью исключаем) ===
        "оценка потребностей клиентов", "развитие ключевых клиентов", "проведение презентаций",
        "проведение переговоров", "подготовка коммерческих предложений", "навыки межличностного общения",
        "межличностное общение", "мотивация персонала", "стратегическое мышление", "аналитическое мышление",
        "командная работа", "клиентоориентированность", "ориентация на результат", "продвижение бренда",
        "развитие продаж", "традиционная розница", "собственная розница", "клиентами", "клиентам",
        "инициатива", "харизма", "многозадачность", "бухгалтерская отчетность", "бухгалтерский учет",
        "управление временем", "тайм-менеджмент", "стрессоустойчивость", "ответственность", 
        "исполнительность", "дисциплинированность", "пунктуальность", "целеустремленность", "командная работа"
        
        # === ЯЗЫКИ И ОБРАЗОВАНИЕ (исключаем) ===
        "английского языка", "русского языка", "высшее образование", "знание английского",
        "английский язык", "русский язык", "язык программирования", "вербальные", "письменный",
        "разговорный английский", "технический английский", "upper intermediate", "intermediate",
        
        # === ОБЩИЕ ФРАЗЫ (ФИЛЬТРЫ) ===
        "опыт работы", "умение", "желательно", "преимуществом", "плюсом", "плюсом будет",
        "знание", "владение", "требуется", "должен", "навык", "компетенция", "обязан",
        "уметь", "знать", "понимать", "владеть", "иметь",
        
        # === БИЗНЕС-ТЕРМИНЫ (исключаем) ===
        "продажи", "маркетинг", "smm", "управление персоналом", "развитие продаж",
        "анализ продаж", "анализ рынка", "анализ цен", "контроль цен",
        "мониторинг рынка", "мониторинг цен", "построение воронки продаж",
        "управление проектами", "управление продуктом", "управление командой",
        "ведение переговоров", "проведение встреч", "организация встреч",
        "бизнес-процессы", "оптимизация процессов", "автоматизация процессов",
        
        # === СПЕЦИФИЧНЫЕ МУСОРНЫЕ ФРАЗЫ ===
        "пластичные смазки", "автомобильные перевозки",
        "организация клиентских мероприятий", "аналитика маркетплейсов",
        "контроль и анализ ценообразования", "банковских или support"
    }
    
    # GENERIC слова (исключаем всегда)
    GENERIC_WORDS = {
        "frontend", "front-end", "front end", "frontend разработка", "front-end разработка",
        "backend", "back-end", "back end", "backend разработка", "back-end разработка",
        "fullstack", "full-stack", "full stack", "fullstack разработка",
        "разработка", "программирование", "кодинг", "coding",
        "web", "веб", "web разработка", "веб разработка",
        "api", "rest", "soap", "graphql",
        "базы данных", "database",
        "git", "svn", "version control",
        "linux", "windows", "macos",
        "контроль и анализ ценообразования","тестирование по","тестирование по тест",
        "тестирование по сценариям",
        "тестирование по кейсам",
        "тестирование по требованиям", "язык программирования",             "или", "подобных", "подобные", "подобный", "подобное",
    "язык", "языка", "языку", "языком", "языке",
    "языки", "языков", "языкам", "языками", "языках",
    "и т.д.", "и т.п.", "и др.", "и проч.", "etc",
    "опыт", "знание", "умение", "навык", "навыки",
    "работа", "работать", "владение", "понимание",
    "систем", "система", "системы", "архитектур", "архитектура",
    "моделей", "модели", "модель", "production", "prod",
    "банковских", "support", "commerce", "ритейле", "антифрод",
    "участие", "проведение", "организация", "управление",
    "разработка", "программирование", "тестирование",
    "оценивать", "построение", "создание", "внедрение",
    "методов", "методы", "алгоритмов", "алгоритмы",
    "библиотеками", "библиотеки", "фреймворками", "фреймворки",
    "платформами", "инструментами", "инструменты",
    "стеком", "стека", "технологиями", "технологии",
        
        # === ОДИНОЧНЫЕ СЛОВА - SOFT SKILLS ===
        "лидерство", "лидер", "стратегия", "креативность", "инновация",
        "организованность", "ответственность", "надежность", "честность",
        "влиятельность", "уверенность", "боевой", "амбициозный",
        "дипломатичность", "пунктуальность", "пунктуален"
    }
    
    # Слова-паразиты (если ВСЕ слова из этого набора - отклоняем)
    FILLER_WORDS = {
        "обо", "это", "что", "как", "быть", "иметь", "делать", "сказать",
        "мочь", "должен", "хотеть", "знать", "работа", "работать", "служба",
        "обладать", "справляться", "справляться", "являться", "становиться",
    }
    
    # Паттерны для проверки
    DIGITS_PATTERN = re.compile(r'^\d+$')
    SPECIAL_PATTERN = re.compile(r'^[^\w\s]+$')
    
    def __init__(
        self,
        blacklist: Optional[Set[str]] = None,
        whitelist: Optional[Set[str]] = None,
        min_length: int = 3,
        max_length: int = 50,
        max_words: int = 4,
        min_confidence: float = 0.5,
        remove_generic: bool = True
    ):
        """
        Args:
            blacklist: Набор запрещённых терминов
            whitelist: Набор разрешённых навыков (если None, не используется)
            min_length: Минимальная длина навыка (символов)
            max_length: Максимальная длина навыка (символов)
            max_words: Максимальное количество слов в навыке
            min_confidence: Минимальная уверенность для валидации
            remove_generic: Удалять ли generic слова
        """
        self.blacklist = blacklist or self.DEFAULT_BLACKLIST
        self.whitelist = {w.lower().strip() for w in whitelist} if whitelist is not None else None
        self.generic_words = self.GENERIC_WORDS if remove_generic else set()
        self.min_length = min_length
        self.max_length = max_length
        self.max_words = max_words
        self.min_confidence = min_confidence
        
        # Статистика
        self.stats = {
            'total': 0,
            'valid': 0,
            'rejected': 0,
            'reasons': Counter()
        }
    
    def validate(self, skill: str, confidence: float = 1.0) -> ValidationResult:
        """
        Валидирует навык с расширенными проверками
        
        Args:
            skill: Навык для проверки
            confidence: Уверенность в извлечении (0-1)
        
        Returns:
            ValidationResult с результатами
        """
        self.stats['total'] += 1
        reasons = []
        
        if not skill or not skill.strip():
            self.stats['rejected'] += 1
            self.stats['reasons']['empty'] += 1
            return ValidationResult(
                skill=skill,
                is_valid=False,
                reasons=[ValidationReason.EMPTY],
                confidence=0.0
            )
        
        skill = skill.strip()
        skill_lower = skill.lower()
        
        # === ПРОВЕРКА 1: Уверенность ===
        if confidence < self.min_confidence:
            reasons.append(ValidationReason.IN_BLACKLIST)
        
        # === ПРОВЕРКА 2: Длина ===
        if len(skill) < self.min_length:
            reasons.append(ValidationReason.TOO_SHORT)
        
        if len(skill) > self.max_length:
            reasons.append(ValidationReason.TOO_LONG)
        
        # === ПРОВЕРКА 3: Количество слов ===
        words = skill_lower.split()
        if len(words) > self.max_words:
            reasons.append(ValidationReason.TOO_LONG)
        
        # === ПРОВЕРКА 4: Только цифры ===
        if self.DIGITS_PATTERN.match(skill):
            reasons.append(ValidationReason.ONLY_DIGITS)
        
        # === ПРОВЕРКА 5: Только спецсимволы ===
        if self.SPECIAL_PATTERN.match(skill):
            reasons.append(ValidationReason.ONLY_SPECIAL)
        
        # === ПРОВЕРКА 6: Generic слова ===
        if skill_lower in self.generic_words:
            reasons.append(ValidationReason.GENERIC_WORD)
        
        # === ПРОВЕРКА 7: Чёрный список (КРИТИЧЕСКАЯ) ===
        for bad in self.blacklist:
            if bad in skill_lower:
                reasons.append(ValidationReason.IN_BLACKLIST)
                break
        
        # === ПРОВЕРКА 8: Слова-паразиты ===
        if all(word in self.FILLER_WORDS for word in words):
            reasons.append(ValidationReason.IN_BLACKLIST)
        
        # === ПРОВЕРКА 9: Белый список (если задан) ===
        if self.whitelist is not None:
            skill_normalized = skill_lower.replace(' ', '').replace('-', '').replace('_', '')
            if skill_normalized not in self.whitelist:
                found_in_whitelist = any(
                    skill_normalized in wl.replace(' ', '').replace('-', '').replace('_', '') or
                    wl.replace(' ', '').replace('-', '').replace('_', '') in skill_normalized
                    for wl in self.whitelist
                )
                if not found_in_whitelist:
                    reasons.append(ValidationReason.NOT_IN_WHITELIST)
        
        # === ПРОВЕРКА 10: Дополнительная - наличие букв ===
        if not any(c.isalpha() for c in skill):
            reasons.append(ValidationReason.ONLY_DIGITS)
        
        is_valid = len(reasons) == 0
        
        # Вычисляем уверенность
        if is_valid:
            confidence_score = min(1.0, confidence + 0.1)
        else:
            confidence_score = max(0.1, 1.0 - len(reasons) * 0.2)
        
        # Обновляем статистику
        if is_valid:
            self.stats['valid'] += 1
        else:
            self.stats['rejected'] += 1
            for reason in reasons:
                self.stats['reasons'][reason.value] += 1
        
        return ValidationResult(
            skill=skill,
            is_valid=is_valid,
            reasons=reasons,
            confidence=confidence_score
        )
    
    def validate_batch(self, skills: List[str], confidences: Optional[List[float]] = None) -> tuple:
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
        
        for skill, conf in zip(skills, confidences):
            result = self.validate(skill, conf)
            results.append(result)
            if result.is_valid:
                valid_skills.append(result.skill)
        
        return valid_skills, results
    
    def get_rejection_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Возвращает отчёт об отклонениях
        """
        rejected = [r for r in results if not r.is_valid]
        
        rejection_counts = Counter()
        for result in rejected:
            for reason in result.reasons:
                rejection_counts[reason.value] += 1
        
        return {
            'total_validated': len(results),
            'valid': len(results) - len(rejected),
            'rejected': len(rejected),
            'rejection_reasons': dict(rejection_counts),
            'rejection_rate': len(rejected) / len(results) if results else 0
        }
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self.stats = {
            'total': 0,
            'valid': 0,
            'rejected': 0,
            'reasons': Counter()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает текущую статистику"""
        return self.stats