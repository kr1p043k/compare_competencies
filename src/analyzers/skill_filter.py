"""
Фильтр навыков: убирает мусор, оставляет только чистые навыки.
Исправленная версия с сохранением пропорций частот.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SkillFilter:
    """
    Фильтрует веса навыков, удаляя мусор и сохраняя реальную важность.
    Исправленная версия с правильной нормализацией и сохранением пропорций.
    """
    
    # Чистые навыки (reference из competency_frequency)
    REFERENCE_SKILLS = {
        # Языки программирования
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "kotlin", "swift",
        "php", "ruby", "scala", "r", "matlab", "perl", "lua", "haskell",
        
        # Фреймворки и библиотеки
        "react", "vue", "angular", "django", "flask", "fastapi", "spring", "spring boot",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "express", "node", "next", "nuxt", "gatsby",
        
        # Базы данных
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite", "cassandra",
        "oracle", "mssql", "dynamodb", "couchdb", "neo4j",
        
        # DevOps и инфраструктура
        "docker", "kubernetes", "k8s", "jenkins", "git", "gitlab", "github", "bitbucket",
        "terraform", "ansible", "prometheus", "grafana", "nginx", "apache",
        
        # Облачные технологии
        "aws", "azure", "gcp", "yandex cloud", "heroku", "digitalocean",
        
        # Data Science и ML
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "big data", "spark", "hadoop", "airflow", "mlflow", "langchain", "mlops",
        
        # Фронтенд
        "html", "css", "sass", "scss", "webpack", "vite", "redux", "mobx", "graphql",
        "rest api", "restful", "axios", "tailwind", "bootstrap", "material ui",
        
        # Тестирование
        "jest", "pytest", "cypress", "playwright", "selenium", "junit", "testng",
        
        # Инструменты
        "figma", "storybook", "eslint", "prettier", "webpack", "babel", "npm", "yarn",
    }
    
    # GENERIC слова, которые ВСЕГДА исключаются
    GENERIC_WORDS = {
        "frontend", "front-end", "front end", "frontend разработка", "front-end разработка",
        "backend", "back-end", "back end", "backend разработка", "back-end разработка",
        "fullstack", "full-stack", "full stack", "fullstack разработка",
        "разработка", "программирование", "кодинг", "coding",
        "web", "веб", "web разработка", "веб разработка",
        "api", "rest", "soap", "graphql",
        "базы данных", "database", "sql", "nosql",
        "git", "svn", "version control",
        "linux", "windows", "macos",
        "английский", "english", "язык", "разработки", "разработка",
        "и др", "и другие", "знание", "опыт", "умение", "требуется",
        "core", "net", "crm", "erp",
    }
    
    def __init__(self, reference_skills: Set[str] = None):
        """
        Args:
            reference_skills: Набор эталонных навыков
        """
        self.reference_skills = reference_skills or self.REFERENCE_SKILLS
        logger.info(f"SkillFilter инициализирован с {len(self.reference_skills)} reference навыками")

    def filter_weights(
        self, 
        skill_weights: Dict[str, float],
        min_weight: float = 0.01
    ) -> Dict[str, float]:
        """
        Агрессивно фильтрует веса навыков.
        
        Удаляет:
        1. Generic слова
        2. Мусор (короткие, неизвестные)
        3. Навыки с очень низким весом
        
        Args:
            skill_weights: Словарь весов навыков
            min_weight: Минимальный вес для включения
        
        Returns:
            Отфильтрованный словарь весов
        """
        if not skill_weights:
            return {}
        
        filtered = {}
        removed_count = 0
        removed_generic = 0
        removed_unknown = 0
        removed_low_weight = 0
        
        logger.info(f"\n🔍 ФИЛЬТРАЦИЯ НАВЫКОВ:")
        logger.info(f"  - Исходно: {len(skill_weights)} навыков")
        
        for skill, weight in skill_weights.items():
            skill_lower = skill.lower().strip()
            
            # === ПРОВЕРКА 1: Generic слово? ===
            if skill_lower in self.GENERIC_WORDS:
                removed_generic += 1
                removed_count += 1
                continue
            
            # === ПРОВЕРКА 2: Слишком маленький вес? ===
            if weight < min_weight:
                removed_low_weight += 1
                removed_count += 1
                continue
            
            # === ПРОВЕРКА 3: Известный навык? ===
            if skill_lower not in self.reference_skills:
                # Проверяем частичное совпадение
                is_known = False
                for ref in self.reference_skills:
                    if ref in skill_lower or skill_lower in ref:
                        is_known = True
                        break
                
                if not is_known:
                    removed_unknown += 1
                    removed_count += 1
                    continue
            
            # Все проверки пройдены - добавляем
            filtered[skill_lower] = weight
        
        logger.info(f"  - Удалено generic: {removed_generic}")
        logger.info(f"  - Удалено unknown: {removed_unknown}")
        logger.info(f"  - Удалено low weight: {removed_low_weight}")
        logger.info(f"  - Всего удалено: {removed_count}")
        logger.info(f"  - ✓ Осталось: {len(filtered)} ЧИСТЫХ навыков")
        
        return filtered

    def normalize_weights(
        self,
        skill_weights: Dict[str, float],
        method: str = 'minmax'
    ) -> Dict[str, float]:
        """
        Нормализует веса с сохранением пропорций.
        
        Args:
            skill_weights: Словарь весов навыков
            method: Метод нормализации ('minmax', 'log', 'softmax')
        
        Returns:
            Нормализованные веса
        """
        if not skill_weights:
            return {}
        
        weights = list(skill_weights.values())
        
        if method == 'minmax':
            min_w = min(weights)
            max_w = max(weights)
            
            if max_w > min_w:
                normalized = {}
                for skill, weight in skill_weights.items():
                    # Нормализуем в диапазон [0.1, 1.0]
                    norm_val = 0.1 + 0.9 * (weight - min_w) / (max_w - min_w)
                    normalized[skill] = round(norm_val, 4)
                return normalized
            else:
                # Все веса одинаковые
                return {skill: 1.0 for skill in skill_weights}
        
        elif method == 'log':
            # Логарифмическая нормализация (для больших разбросов)
            log_weights = {skill: np.log1p(weight) for skill, weight in skill_weights.items()}
            max_log = max(log_weights.values())
            if max_log > 0:
                return {skill: round(w / max_log, 4) for skill, w in log_weights.items()}
            return skill_weights
        
        elif method == 'softmax':
            # Softmax нормализация
            exp_weights = {skill: np.exp(weight) for skill, weight in skill_weights.items()}
            total = sum(exp_weights.values())
            if total > 0:
                return {skill: round(w / total, 4) for skill, w in exp_weights.items()}
            return skill_weights
        
        else:
            logger.warning(f"Неизвестный метод нормализации: {method}")
            return skill_weights

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
        
        Args:
            skill_weights: Веса из TF-IDF (или других методов)
            competency_freq: Частоты из компетенций
        
        Returns:
            Объединённый словарь весов
        """
        if not competency_freq:
            logger.warning("competency_freq пуст, возвращаем skill_weights")
            return self.filter_weights(skill_weights)
        
        merged = {}
        
        logger.info(f"\n🔗 ОБЪЕДИНЕНИЕ С COMPETENCY_FREQUENCY:")
        logger.info(f"  - competency_freq источник: {len(competency_freq)} навыков")
        logger.info(f"  - skill_weights источник: {len(skill_weights)} навыков")
        
        # Найдём максимальный count для нормализации
        max_count = max(competency_freq.values()) if competency_freq.values() else 1
        min_count = min(competency_freq.values()) if competency_freq.values() else 0
        
        tfidf_weights_used = 0
        count_weights_used = 0
        generic_removed = 0
        
        for skill, count in competency_freq.items():
            skill_clean = skill.lower().strip()
            
            # Исключаем generic слова
            if skill_clean in self.GENERIC_WORDS:
                generic_removed += 1
                continue
            
            # Пытаемся использовать TF-IDF вес
            if skill_clean in skill_weights:
                # ✅ Используем TF-IDF вес (это честнее, чем count)
                merged[skill_clean] = skill_weights[skill_clean]
                tfidf_weights_used += 1
            else:
                # Если нет в TF-IDF, нормализуем count
                # Используем min-max нормализацию для сохранения различий
                if max_count > min_count:
                    weight = (count - min_count) / (max_count - min_count)
                else:
                    weight = count / max_count if max_count > 0 else 0.5
                
                # Немного понижаем вес для count-based (так как это менее надёжно)
                weight = weight * 0.8
                merged[skill_clean] = round(weight, 4)
                count_weights_used += 1
        
        logger.info(f"  - Удалено generic слов: {generic_removed}")
        logger.info(f"  - Добавлено из competency_freq: {len(competency_freq) - generic_removed}")
        logger.info(f"    • с TF-IDF весами: {tfidf_weights_used}")
        logger.info(f"    • с count весами: {count_weights_used}")
        
        # Добавляем навыки, которые есть только в skill_weights
        skills_only_in_tfidf = set(skill_weights.keys()) - set(merged.keys())
        for skill in skills_only_in_tfidf:
            skill_clean = skill.lower().strip()
            if skill_clean not in self.GENERIC_WORDS:
                merged[skill_clean] = skill_weights[skill]
                logger.debug(f"  + Добавлен из skill_weights: {skill_clean}")
        
        logger.info(f"  - ✓ ИТОГО: {len(merged)} навыков")
        
        return merged

    def get_clean_weights(
        self,
        skill_weights_raw: Dict[str, float],
        competency_freq: Dict[str, int] = None,
        use_reference: bool = True,
        normalize_method: str = 'minmax'
    ) -> Dict[str, float]:
        """
        Финальная очистка весов с сохранением реальной важности навыков.
        
        Args:
            skill_weights_raw: Сырые веса навыков
            competency_freq: Частоты компетенций (основной источник)
            use_reference: Использовать reference_skills для фильтрации
            normalize_method: Метод нормализации ('minmax', 'log', 'softmax')
        
        Returns:
            Очищенные и нормализованные веса
        """
        logger.info("\n" + "="*80)
        logger.info("ФИНАЛЬНАЯ ОЧИСТКА НАВЫКОВ (сохранение пропорций)")
        logger.info("="*80)

        if not skill_weights_raw:
            logger.warning("skill_weights_raw пустой")
            return {}

        # 1. Берём данные из competency_freq как основной источник частот (самый надёжный)
        if competency_freq and len(competency_freq) > 0:
            logger.info(f"Используем реальные частоты из competency_frequency.json ({len(competency_freq)} навыков)")
            raw_freq = {k.lower().strip(): float(v) for k, v in competency_freq.items() if v > 0}
        else:
            raw_freq = {k.lower().strip(): float(v) for k, v in skill_weights_raw.items() if v > 0}
            logger.info(f"Используем skill_weights_raw как источник ({len(raw_freq)} навыков)")

        # 2. Убираем generic слова и слишком длинные фразы
        generic_removed = 0
        long_removed = 0
        
        for word in list(raw_freq.keys()):
            if word in self.GENERIC_WORDS:
                del raw_freq[word]
                generic_removed += 1
            elif len(word.split()) > 4:  # слишком длинные фразы
                del raw_freq[word]
                long_removed += 1

        logger.info(f"- Удалено generic слов: {generic_removed}")
        logger.info(f"- Удалено длинных фраз (>4 слов): {long_removed}")
        logger.info(f"- Осталось после очистки: {len(raw_freq)} навыков")

        if not raw_freq:
            logger.warning("После очистки не осталось навыков!")
            return {}

        # 3. Фильтрация по reference (если нужно)
        if use_reference and self.reference_skills:
            filtered_by_ref = {}
            for skill, weight in raw_freq.items():
                if skill in self.reference_skills:
                    filtered_by_ref[skill] = weight
                else:
                    # Проверяем частичное совпадение
                    matched = False
                    for ref in self.reference_skills:
                        if ref in skill or skill in ref:
                            matched = True
                            break
                    if matched:
                        filtered_by_ref[skill] = weight * 0.9  # немного снижаем вес
            
            logger.info(f"- После фильтрации по reference: {len(filtered_by_ref)} навыков")
            raw_freq = filtered_by_ref

        if not raw_freq:
            logger.warning("После reference фильтрации не осталось навыков!")
            return {}

        # 4. Нормализация с сохранением пропорций
        normalized = self.normalize_weights(raw_freq, method=normalize_method)

        # 5. Логируем топ-10
        logger.info(f"\n✓ ИТОГО чистых навыков: {len(normalized)}")
        logger.info("ТОП-10 наиболее важных навыков рынка:")
        
        top_skills = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:10]
        for skill, w in top_skills:
            original_weight = raw_freq.get(skill, 0)
            logger.info(f"   {skill:30} → {w:.4f} (исходный вес: {original_weight:.2f})")

        return normalized

    def validate_skills(self, skills: List[str]) -> List[str]:
        """
        Валидирует список навыков, оставляя только чистые.
        
        Args:
            skills: Список навыков для валидации
        
        Returns:
            Список валидных навыков
        """
        validated = []
        
        for skill in skills:
            skill_lower = skill.lower().strip()
            
            # Пропускаем generic
            if skill_lower in self.GENERIC_WORDS:
                continue
            
            # Пропускаем слишком длинные
            if len(skill_lower.split()) > 4:
                continue
            
            # Пропускаем неизвестные (если нужна валидация)
            if skill_lower not in self.reference_skills:
                # Проверяем частичное совпадение
                matched = False
                for ref in self.reference_skills:
                    if ref in skill_lower or skill_lower in ref:
                        matched = True
                        break
                if not matched:
                    continue
            
            validated.append(skill_lower)
        
        return validated

    def get_skill_categories(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Группирует навыки по категориям.
        
        Args:
            skills: Список навыков
        
        Returns:
            Словарь {категория: [список навыков]}
        """
        categories = {
            'programming_languages': [],
            'frameworks': [],
            'databases': [],
            'devops': [],
            'cloud': [],
            'data_science': [],
            'frontend': [],
            'testing': [],
            'tools': [],
            'other': []
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            
            if skill_lower in ['python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'go', 'rust', 'kotlin', 'swift', 'php', 'ruby']:
                categories['programming_languages'].append(skill)
            elif skill_lower in ['react', 'vue', 'angular', 'django', 'flask', 'fastapi', 'spring', 'express', 'next', 'nuxt']:
                categories['frameworks'].append(skill)
            elif skill_lower in ['postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'sqlite']:
                categories['databases'].append(skill)
            elif skill_lower in ['docker', 'kubernetes', 'jenkins', 'git', 'terraform', 'ansible']:
                categories['devops'].append(skill)
            elif skill_lower in ['aws', 'azure', 'gcp', 'yandex cloud']:
                categories['cloud'].append(skill)
            elif skill_lower in ['machine learning', 'deep learning', 'nlp', 'pandas', 'numpy', 'tensorflow', 'pytorch']:
                categories['data_science'].append(skill)
            elif skill_lower in ['html', 'css', 'sass', 'scss', 'webpack', 'redux', 'graphql']:
                categories['frontend'].append(skill)
            elif skill_lower in ['jest', 'pytest', 'cypress', 'playwright', 'selenium']:
                categories['testing'].append(skill)
            elif skill_lower in ['git', 'figma', 'storybook', 'eslint', 'prettier']:
                categories['tools'].append(skill)
            else:
                categories['other'].append(skill)
        
        # Убираем пустые категории
        return {k: v for k, v in categories.items() if v}