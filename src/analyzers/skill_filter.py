"""
Фильтр навыков: убирает мусор, оставляет только чистые навыки.
Использует competency_frequency как reference.
"""

from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)


class SkillFilter:
    """
    Фильтрует веса навыков, удаляя:
    - ВСЕ bigrams (двухсловные и более)
    - Мусорные комбинации
    - Generic слова (frontend, backend, fullstack)
    - Слова, которые не в reference
    
    Остаёт ТОЛЬКО чистые, атомарные навыки.
    """
    
    # Чистые навыки (reference из competency_frequency)
    REFERENCE_SKILLS = {
        "postgresql", "rest", "api", "java", "redis", "git", "sql",
        "fastapi", "django", "docker", "mysql", "linux", "mongodb",
        "react", "graphql", "typescript", "javascript", "python",
        "kubernetes", "css", "html", "node", "express", "sass",
        "webpack", "vite", "jest", "cypress", "npm", "yarn",
        "go", "rust", "kotlin", "scala", "php", "ruby",
        "flask", "tornado", "aiohttp", "celery", "pytest",
        "postgresql", "mongodb", "redis", "elasticsearch",
        "aws", "azure", "gcp", "heroku", "netlify", "vercel",
        "stripe", "paypal", "twilio", "auth0", "firebase",
        "pytorch", "tensorflow", "keras", "numpy", "pandas",
        "scikit-learn", "xgboost", "catboost",
    }
    
    # === GENERIC слова, которые ВСЕГДА исключаются ===
    GENERIC_WORDS = {
        "frontend", "backend", "fullstack", "api",  # слишком broad
        "core", "net",  # части других слов
        "crm",  # business term, не tech skill
    }
    
    def __init__(self, reference_skills: Set[str] = None):
        """
        Args:
            reference_skills: Набор эталонных навыков
        """
        self.reference_skills = reference_skills or self.REFERENCE_SKILLS
        logger.info(f"SkillFilter инициализирован с {len(self.reference_skills)} reference навыками")

    def filter_weights(self, skill_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Агрессивно фильтрует веса навыков.
        
        Удаляет:
        1. ВСЕ многосложные навыки (bigrams+)
        2. Generic слова
        3. Мусор (короткие, неизвестные)
        """
        if not skill_weights:
            return {}
        
        filtered = {}
        removed_count = 0
        removed_bigrams = 0
        removed_generic = 0
        removed_unknown = 0
        
        logger.info(f"\n🔍 АГРЕССИВНАЯ ФИЛЬТРАЦИЯ НАВЫКОВ:")
        logger.info(f"  - Исходно: {len(skill_weights)} навыков")
        
        for skill, weight in skill_weights.items():
            skill_lower = skill.lower().strip()
            words = skill_lower.split()
            
            # === ПРОВЕРКА 1: Это bigram или более? ===
            if len(words) > 1:
                removed_bigrams += 1
                removed_count += 1
                continue
            
            # === ПРОВЕРКА 2: Это generic слово? ===
            if skill_lower in self.GENERIC_WORDS:
                removed_generic += 1
                removed_count += 1
                continue
            
            # === ПРОВЕРКА 3: Это известный навык? ===
            if skill_lower not in self.reference_skills:
                removed_unknown += 1
                removed_count += 1
                continue
            
            # Все проверки пройдены - добавляем
            filtered[skill_lower] = weight
        
        logger.info(f"  - Удалено bigrams: {removed_bigrams}")
        logger.info(f"  - Удалено generic: {removed_generic}")
        logger.info(f"  - Удалено unknown: {removed_unknown}")
        logger.info(f"  - Всего удалено: {removed_count}")
        logger.info(f"  - ✓ Осталось: {len(filtered)} ЧИСТЫХ навыков")
        
        return filtered

    def merge_with_reference(
        self,
        skill_weights: Dict[str, float],
        competency_freq: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Объединяет очищенные skill_weights с competency_frequency.
        competency_freq имеет приоритет (это самый честный источник).
        """
        if not competency_freq:
            # Если competency_freq нет, просто возвращаем отфильтрованные веса
            return self.filter_weights(skill_weights)
        
        merged = {}
        
        logger.info(f"\n🔗 ОБЪЕДИНЕНИЕ С COMPETENCY_FREQUENCY:")
        logger.info(f"  - competency_freq источник: {len(competency_freq)} навыков")
        
        # === ШАГ 1: Добавляем ВСЁ из competency_freq (это истина) ===
        competency_freq_added = 0
        for skill, count in competency_freq.items():
            skill_clean = skill.lower().strip()
            
            # Исключаем generic слова
            if skill_clean in self.GENERIC_WORDS:
                logger.debug(f"  ⊘ competency_freq: исключён '{skill}' (generic)")
                continue
            
            # Только unigrams!
            words = skill_clean.split()
            if len(words) > 1:
                logger.debug(f"  ⊘ competency_freq: исключён '{skill}' (bigram)")
                continue
            
            # Добавляем с максимальным весом (это честный источник)
            merged[skill_clean] = 1.0
            competency_freq_added += 1
        
        logger.info(f"  - Добавлено из competency_freq: {competency_freq_added}")
        
        # === ШАГ 2: Добавляем дополнительно из skill_weights ===
        skill_weights_added = 0
        for skill, weight in skill_weights.items():
            skill_lower = skill.lower().strip()
            
            # Пропускаем если уже есть
            if skill_lower in merged:
                continue
            
            # Пропускаем generic слова
            if skill_lower in self.GENERIC_WORDS:
                continue
            
            # Только unigrams!
            words = skill_lower.split()
            if len(words) > 1:
                continue
            
            # Только если вес высокий (> 0.1)
            if weight >= 0.10:
                merged[skill_lower] = weight
                skill_weights_added += 1
        
        logger.info(f"  - Добавлено из skill_weights (вес >= 0.10): {skill_weights_added}")
        logger.info(f"  - ✓ ИТОГО: {len(merged)} навыков в финале")
        
        # === ШАГ 3: Нормализуем ===
        if merged:
            max_weight = max(merged.values())
            merged = {k: v / max_weight for k, v in merged.items()}
        
        return merged

    def get_clean_weights(
        self,
        skill_weights: Dict[str, float],
        competency_freq: Dict[str, int] = None,
        use_reference: bool = True
    ) -> Dict[str, float]:
        """
        Получает чистые веса навыков.
        
        Порядок обработки:
        1. Фильтруем TF-IDF веса (удаляем bigrams, generic, unknown)
        2. Объединяем с competency_freq если есть
        3. Нормализуем
        """
        logger.info("\n" + "="*80)
        logger.info("ФИНАЛЬНАЯ ОЧИСТКА НАВЫКОВ")
        logger.info("="*80)
        
        # Шаг 1: Фильтруем
        filtered = self.filter_weights(skill_weights)
        
        # Шаг 2: Объединяем с reference если есть
        if competency_freq and use_reference:
            clean = self.merge_with_reference(filtered, competency_freq)
        else:
            clean = filtered
        
        if not clean:
            logger.error("❌ После фильтрации навыков не осталось!")
            return {}
        
        logger.info(f"\n✅ ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
        logger.info(f"  - Чистых навыков: {len(clean)}")
        
        # Логируем топ-10
        logger.info(f"\n  Топ-10 навыков:")
        top_10 = sorted(clean.items(), key=lambda x: x[1], reverse=True)[:10]
        for rank, (skill, weight) in enumerate(top_10, 1):
            logger.info(f"    {rank:2d}. {skill:20s} - {weight:.4f}")
        
        return clean

    def validate_skills(self, skills: List[str]) -> List[str]:
        """Валидирует список навыков, оставляя только чистые."""
        validated = []
        
        for skill in skills:
            skill_lower = skill.lower().strip()
            words = skill_lower.split()
            
            # Пропускаем многосложные
            if len(words) > 1:
                continue
            
            # Пропускаем generic
            if skill_lower in self.GENERIC_WORDS:
                continue
            
            # Пропускаем неизвестные (если нужна валидация)
            if skill_lower not in self.reference_skills:
                continue
            
            validated.append(skill_lower)
        
        return validated