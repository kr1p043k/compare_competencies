"""
Фильтр навыков: убирает мусор, оставляет только чистые навыки.
Исправленная версия с сохранением пропорций частот.
"""

import numpy as np
import structlog

from src import Result, Ok, Err
from src.errors import DomainError

logger = structlog.get_logger(__name__)


class SkillFilter:
    """
    Фильтрует веса навыков, удаляя мусор и сохраняя реальную важность.
    Исправленная версия с правильной нормализацией и сохранением пропорций.
    """

    # GENERIC слова, которые ВСЕГДА исключаются
    # Загружаются из внешнего JSON-файла, указанного в GENERIC_WORDS_PATH (config)

    def __init__(self, reference_skills: set[str] = None, generic_words_path: str = None):
        """
        Args:
            reference_skills: Набор эталонных навыков. Если не передан, загружается из it_skills.json.
            generic_words_path: Путь к JSON-файлу со стоп-словами навыков.
        """
        if reference_skills is None:
            from src.parsing.utils import load_it_skills

            reference_skills = load_it_skills()
        self.reference_skills = reference_skills
        self.GENERIC_WORDS = self._load_generic_words(generic_words_path)

    @staticmethod
    def _load_generic_words(path: str = None) -> set[str]:
        from src import config as cfg
        p = path or cfg.GENERIC_WORDS_PATH
        try:
            with open(p, encoding="utf-8") as f:
                import json
                return set(json.load(f))
        except Exception:
            return set()

    def filter_weights(self, skill_weights: dict[str, float], min_weight: float = 0.01) -> Result[dict[str, float], DomainError]:
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
            return Ok({})

        filtered = {}
        removed_count = 0
        removed_generic = 0
        removed_unknown = 0
        removed_low_weight = 0

        logger.info("skill_filtering_started", total_skills=len(skill_weights))

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

        logger.info(
            "skill_filtering_completed",
            removed_generic=removed_generic,
            removed_unknown=removed_unknown,
            removed_low_weight=removed_low_weight,
            total_removed=removed_count,
            remaining=len(filtered),
        )

        return Ok(filtered)

    def normalize_weights(self, skill_weights: dict[str, float], method: str = "minmax") -> Result[dict[str, float], DomainError]:
        """
        Нормализует веса с сохранением пропорций.

        Args:
            skill_weights: Словарь весов навыков
            method: Метод нормализации ('minmax', 'log', 'softmax')

        Returns:
            Нормализованные веса
        """
        if not skill_weights:
            return Ok({})

        weights = list(skill_weights.values())

        if method == "minmax":
            min_w = min(weights)
            max_w = max(weights)

            if max_w > min_w:
                normalized = {}
                for skill, weight in skill_weights.items():
                    # Нормализуем в диапазон [0.1, 1.0]
                    norm_val = 0.1 + 0.9 * (weight - min_w) / (max_w - min_w)
                    normalized[skill] = round(norm_val, 4)
                return Ok(normalized)
            else:
                # Все веса одинаковые
                return Ok({skill: 1.0 for skill in skill_weights})

        elif method == "log":
            # Логарифмическая нормализация (для больших разбросов)
            log_weights = {skill: np.log1p(weight) for skill, weight in skill_weights.items()}
            max_log = max(log_weights.values())
            if max_log > 0:
                return Ok({skill: round(w / max_log, 4) for skill, w in log_weights.items()})
            return Ok(skill_weights)

        elif method == "softmax":
            # Softmax нормализация
            exp_weights = {skill: np.exp(weight) for skill, weight in skill_weights.items()}
            total = sum(exp_weights.values())
            if total > 0:
                return Ok({skill: round(w / total, 4) for skill, w in exp_weights.items()})
            return Ok(skill_weights)

        else:
            logger.warning("unknown_normalization_method", method=method)
            return Ok(skill_weights)

    def merge_with_reference(
        self, skill_weights: dict[str, float], competency_freq: dict[str, int]
    ) -> Result[dict[str, float], DomainError]:
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
            logger.warning("competency_freq_empty")
            return self.filter_weights(skill_weights)

        merged = {}

        logger.info(
            "merging_with_competency_freq",
            competency_freq_count=len(competency_freq),
            skill_weights_count=len(skill_weights),
        )

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

        logger.info(
            "merge_stats",
            generic_removed=generic_removed,
            from_competency_freq=len(competency_freq) - generic_removed,
            with_tfidf_weights=tfidf_weights_used,
            with_count_weights=count_weights_used,
        )

        # Добавляем навыки, которые есть только в skill_weights
        skills_only_in_tfidf = set(skill_weights.keys()) - set(merged.keys())
        for skill in skills_only_in_tfidf:
            skill_clean = skill.lower().strip()
            if skill_clean not in self.GENERIC_WORDS:
                merged[skill_clean] = skill_weights[skill]
                logger.debug("skill_added_from_tfidf", skill=skill_clean)

        logger.info("merge_total", total_skills=len(merged))

        return Ok(merged)

    def get_clean_weights(
        self,
        skill_weights_raw: dict[str, float],
        competency_freq: dict[str, int] = None,
        use_reference: bool = True,
        normalize_method: str = "minmax",
    ) -> Result[dict[str, float], DomainError]:
        """
        Финальная очистка весов с сохранением реальной важности навыков.
        """
        logger.info("final_cleanup_started")

        if not skill_weights_raw:
            logger.warning("skill_weights_raw_empty")
            return Ok({})

        # 1. Берём данные из competency_freq как основной источник частот
        if competency_freq and len(competency_freq) > 0:
            logger.info("using_competency_freq", count=len(competency_freq))
            raw_freq = {k.lower().strip(): float(v) for k, v in competency_freq.items() if v > 0}
        else:
            raw_freq = {k.lower().strip(): float(v) for k, v in skill_weights_raw.items() if v > 0}
            logger.info("using_skill_weights_raw", count=len(raw_freq))

        # 2. Убираем generic слова (оставляем без изменений)
        generic_removed = 0
        for word in list(raw_freq.keys()):
            if word in self.GENERIC_WORDS:
                del raw_freq[word]
                generic_removed += 1
        logger.info("generic_words_removed", count=generic_removed)

        # 3. Фильтрация по reference (понижаем вес неизвестным, НО не удаляем)
        if use_reference and self.reference_skills:
            filtered_by_ref = {}
            for skill, weight in raw_freq.items():
                if skill in self.reference_skills:
                    filtered_by_ref[skill] = weight
                else:
                    # Проверяем частичное совпадение
                    matched = any(ref in skill or skill in ref for ref in self.reference_skills)
                    if matched:
                        filtered_by_ref[skill] = weight * 0.85  # частичное совпадение
                    else:
                        filtered_by_ref[skill] = weight * 0.4  # совсем неизвестный, но сохраняем
            raw_freq = filtered_by_ref
            logger.info("after_reference_filter", count=len(raw_freq))

        # 4. ТОЛЬКО ТЕПЕРЬ удаляем длинные фразы (>6 слов)
        long_removed = 0
        for word in list(raw_freq.keys()):
            if len(word.split()) > 6:
                del raw_freq[word]
                long_removed += 1
        logger.info(
            "long_phrases_removed",
            removed=long_removed,
            remaining=len(raw_freq),
        )

        if not raw_freq:
            logger.warning("no_skills_after_cleanup")
            return Ok({})

        # 5. Нормализация с сохранением пропорций
        normalized_result = self.normalize_weights(raw_freq, method=normalize_method)
        if normalized_result.is_err():
            return Err(normalized_result.err())
        normalized = normalized_result.ok()
        if normalized:
            vals = list(normalized.values())
            logger.info(
                "cleanup_completed",
                total=len(normalized),
                min_weight=round(min(vals), 4),
                max_weight=round(max(vals), 4),
            )
        else:
            logger.warning("no_skills_after_normalization")

        # 6. Логируем топ-10
        logger.info("top_10_market_skills")
        top_skills = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:10]
        for skill, w in top_skills:
            original_weight = raw_freq.get(skill, 0)
            logger.info(
                "top_skill",
                skill=skill,
                normalized_weight=round(w, 4),
                original_weight=round(original_weight, 2),
            )

        return Ok(normalized)

    def validate_skills(self, skills: list[str]) -> Result[list[str], DomainError]:
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

        return Ok(validated)

    def get_skill_categories(self, skills: list[str]) -> Result[dict[str, list[str]], DomainError]:
        """
        Группирует навыки по категориям.

        Args:
            skills: Список навыков

        Returns:
            Словарь {категория: [список навыков]}
        """
        categories = {
            "programming_languages": [],
            "frameworks": [],
            "databases": [],
            "devops": [],
            "cloud": [],
            "data_science": [],
            "frontend": [],
            "testing": [],
            "tools": [],
            "other": [],
        }

        for skill in skills:
            skill_lower = skill.lower()

            if skill_lower in [
                "python",
                "javascript",
                "typescript",
                "java",
                "c++",
                "c#",
                "go",
                "rust",
                "kotlin",
                "swift",
                "php",
                "ruby",
            ]:
                categories["programming_languages"].append(skill)
            elif skill_lower in [
                "react",
                "vue",
                "angular",
                "django",
                "flask",
                "fastapi",
                "spring",
                "express",
                "next",
                "nuxt",
            ]:
                categories["frameworks"].append(skill)
            elif skill_lower in ["postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite"]:
                categories["databases"].append(skill)
            elif skill_lower in ["docker", "kubernetes", "jenkins", "git", "terraform", "ansible"]:
                categories["devops"].append(skill)
            elif skill_lower in ["aws", "azure", "gcp", "yandex cloud"]:
                categories["cloud"].append(skill)
            elif skill_lower in [
                "machine learning",
                "deep learning",
                "nlp",
                "pandas",
                "numpy",
                "tensorflow",
                "pytorch",
            ]:
                categories["data_science"].append(skill)
            elif skill_lower in ["html", "css", "sass", "scss", "webpack", "redux", "graphql"]:
                categories["frontend"].append(skill)
            elif skill_lower in ["jest", "pytest", "cypress", "playwright", "selenium"]:
                categories["testing"].append(skill)
            elif skill_lower in ["git", "figma", "storybook", "eslint", "prettier"]:
                categories["tools"].append(skill)
            else:
                categories["other"].append(skill)

        # Убираем пустые категории
        return Ok({k: v for k, v in categories.items() if v})
