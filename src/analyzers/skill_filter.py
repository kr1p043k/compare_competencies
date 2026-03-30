"""
Фильтр навыков: убирает мусор, оставляет только чистые навыки.
Использует competency_frequency как reference.
"""
import numpy as np
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
        "frontend", "backend", "fullstack", "api", "core", "net", "crm",
        "английский", "english", "язык", "разработки", "разработка",  # ← добавь
        "и др", "и другие", "знание", "опыт", "умение", "требуется"
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
            #if len(words) > 1:
                #removed_bigrams += 1
                #removed_count += 1
                #continue
            
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
        Объединяет skill_weights (TF-IDF) с competency_frequency.
        
        ЛОГИКА:
        - Если навык есть в competency_freq И в skill_weights → берём TF-IDF вес
        - Если навык есть ТОЛЬКО в competency_freq → вес = нормализованный count
        - Если навык есть ТОЛЬКО в skill_weights → берём TF-IDF вес
        """
        if not competency_freq:
            return self.filter_weights(skill_weights)
        
        merged = {}
        
        logger.info(f"\n🔗 ОБЪЕДИНЕНИЕ С COMPETENCY_FREQUENCY:")
        logger.info(f"  - competency_freq источник: {len(competency_freq)} навыков")
        logger.info(f"  - skill_weights (TF-IDF) источник: {len(skill_weights)} навыков")
        
        # === ШАГ 1: Добавляем навыки из competency_freq ===
        competency_freq_added = 0
        tfidf_weights_used = 0
        count_weights_used = 0
        
        # Найдём максимальный count для нормализации
        max_count = max(competency_freq.values()) if competency_freq.values() else 1
        
        for skill, count in competency_freq.items():
            skill_clean = skill.lower().strip()
            
            # Исключаем ТОЛЬКО generic слова
            if skill_clean in self.GENERIC_WORDS:
                logger.debug(f"  ⊘ исключён '{skill}' (generic)")
                continue
            
            # BIGRAMS теперь разрешены!
            # Пытаемся использовать TF-IDF вес
            if skill_clean in skill_weights:
                # ✅ Используем TF-IDF вес (это честнее, чем count)
                merged[skill_clean] = skill_weights[skill_clean]
                tfidf_weights_used += 1
                logger.debug(f"  ✓ '{skill}' - TF-IDF вес: {skill_weights[skill_clean]:.4f}")
            else:
                # Если нет в TF-IDF, нормализуем count
                # Это может быть редкий навык, который не попал в TF-IDF модель
                weight = float(count) / max_count if max_count > 0 else 0.5
                merged[skill_clean] = weight
                count_weights_used += 1
                logger.debug(f"  ✓ '{skill}' - count вес: {weight:.4f} (count={count})")
            
            competency_freq_added += 1
        
        logger.info(f"  - Добавлено из competency_freq: {competency_freq_added}")
        logger.info(f"    • с TF-IDF весами: {tfidf_weights_used}")
        logger.info(f"    • с count весами: {count_weights_used}")
        
        if not merged:
            logger.warning("  ⚠️  competency_freq после фильтрации пуст!")
            return {}
        
        logger.info(f"  - ✓ ИТОГО: {len(merged)} навыков")
        
        # === ШАГ 2: НЕ нормализуем, если уже используем TF-IDF веса ===
        # Если все веса из TF-IDF - они уже нормализованы
        # Если смешанные (TF-IDF + count) - нормализуем по max
        
        if tfidf_weights_used > 0 and count_weights_used == 0:
            # Все веса из TF-IDF - не трогаем
            logger.info(f"  - Веса уже нормализованы (из TF-IDF)")
            return merged
        else:
            # Есть count веса - нормализуем всё
            max_weight = max(merged.values())
            if max_weight > 0:
                merged = {k: v / max_weight for k, v in merged.items()}
                logger.info(f"  - Веса нормализованы по max={max_weight:.4f}")
            
            return merged

    def get_clean_weights(self, skill_weights_raw: Dict[str, float], 
                          competency_freq: Dict[str, int] = None, 
                          use_reference: bool = True) -> Dict[str, float]:
        """Финальная очистка весов с сохранением реальной важности навыков"""
        logger.info("\n" + "="*80)
        logger.info("ФИНАЛЬНАЯ ОЧИСТКА НАВЫКОВ (улучшенная версия 2026)")
        logger.info("="*80)

        if not skill_weights_raw:
            logger.warning("skill_weights_raw пустой")
            return {}

        # 1. Берём данные из competency_freq как основной источник частот (самый надёжный)
        if competency_freq and len(competency_freq) > 0:
            logger.info(f"Используем реальные частоты из competency_frequency.json ({len(competency_freq)} навыков)")
            raw_freq = {k.lower().strip(): float(v) for k, v in competency_freq.items() if v > 0}
        else:
            # fallback на то, что пришло
            raw_freq = {k.lower().strip(): float(v) for k, v in skill_weights_raw.items() if v > 0}

        # 2. Убираем generic слова
        generic_removed = 0
        for word in list(raw_freq.keys()):
            if word in self.GENERIC_WORDS or len(word.split()) > 3:   # убираем очень длинные фразы
                del raw_freq[word]
                generic_removed += 1

        logger.info(f"- Удалено generic и длинных фраз: {generic_removed}")
        logger.info(f"- Осталось после очистки: {len(raw_freq)} навыков")

        if not raw_freq:
            return {}

        # 3. Нормализация с сохранением пропорций (главное исправление)
        total = sum(raw_freq.values())
        if total > 0:
            # Используем softmax-подобную нормализацию, чтобы частые навыки сильно выделялись
            normalized = {}
            for skill, count in raw_freq.items():
                # Логарифмическая шкала + нормализация (чтобы не было плоских весов)
                weight = np.log1p(count) / np.log1p(total) if total > 1 else count / total
                normalized[skill] = round(float(weight), 4)
        else:
            normalized = {skill: 1.0 for skill in raw_freq}

        logger.info(f"✓ ИТОГО чистых навыков: {len(normalized)}")
        logger.info("ТОП-10 наиболее важных навыков рынка:")
        for skill, w in sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"   {skill:25} → {w:.4f}")

        return normalized

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