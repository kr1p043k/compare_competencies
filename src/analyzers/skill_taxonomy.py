"""
Таксономия навыков — иерархическая классификация IT-навыков по категориям.
Singleton, загружается один раз при первом обращении.
"""

import json
import logging
from typing import Optional

from src import config

logger = logging.getLogger(__name__)


class SkillTaxonomy:
    """
    Категоризация навыков на основе skill_taxonomy.json.
    Используется для интерпретации кластеров, улучшения рекомендаций и визуализации.
    """

    _instance: Optional["SkillTaxonomy"] = None
    _taxonomy: dict | None = None
    _skill_to_category: dict[str, str] = {}  # python → programming_languages
    _category_info: dict[str, dict[str, str]] = {}  # programming_languages → {label, icon}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Загружает таксономию из JSON-файла."""
        path = config.DATA_DIR / "skill_taxonomy.json"
        if not path.exists():
            logger.warning(f"Файл таксономии не найден: {path}. Используется fallback 'other'.")
            return

        try:
            with open(path, encoding="utf-8") as f:
                self._taxonomy = json.load(f)

            # Строим обратный маппинг: skill_lower → category_id
            categories = self._taxonomy.get("categories", {})
            for cat_id, cat_data in categories.items():
                self._category_info[cat_id] = {"label": cat_data.get("label", cat_id), "icon": cat_data.get("icon", "")}
                for skill in cat_data.get("skills", []):
                    self._skill_to_category[skill.lower().strip()] = cat_id

            logger.info(f"Таксономия загружена: {len(categories)} категорий, {len(self._skill_to_category)} навыков")
        except Exception as e:
            logger.error(f"Ошибка загрузки таксономии: {e}")

    def get_category(self, skill: str) -> str:
        """
        Возвращает ID категории навыка.
        Пример: get_category('Python') → 'programming_languages'
        """
        return self._skill_to_category.get(skill.lower().strip(), "other")

    def get_category_label(self, skill: str) -> str:
        """
        Возвращает человекочитаемое название категории для навыка.
        Пример: get_category_label('Python') → 'Языки программирования'
        """
        cat_id = self.get_category(skill)
        return self._category_info.get(cat_id, {}).get("label", cat_id)

    def get_category_icon(self, skill: str) -> str:
        """
        Возвращает иконку категории для навыка.
        Пример: get_category_icon('Python') → '💻'
        """
        cat_id = self.get_category(skill)
        return self._category_info.get(cat_id, {}).get("icon", "")

    def get_category_label_by_id(self, category_id: str) -> str:
        """
        Возвращает человекочитаемое название категории по её ID.
        Пример: get_category_label_by_id('programming_languages') → 'Языки программирования'
        """
        return self._category_info.get(category_id, {}).get("label", category_id)

    def get_category_icon_by_id(self, category_id: str) -> str:
        """
        Возвращает иконку категории по её ID.
        Пример: get_category_icon_by_id('programming_languages') → '💻'
        """
        return self._category_info.get(category_id, {}).get("icon", "")

    def get_skills_in_category(self, category_id: str) -> list[str]:
        """
        Возвращает все навыки указанной категории.
        """
        return self._taxonomy.get("categories", {}).get(category_id, {}).get("skills", [])

    def get_all_categories(self) -> list[str]:
        """Возвращает список всех ID категорий."""
        return list(self._category_info.keys())

    def get_category_stats(self, skills: list[str]) -> dict[str, int]:
        """
        Подсчитывает количество навыков по категориям.
        Возвращает {category_id: count}.
        """
        stats: dict[str, int] = {}
        for skill in skills:
            cat = self.get_category(skill)
            stats[cat] = stats.get(cat, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

    def get_dominant_category(self, skills: list[str]) -> str:
        """
        Возвращает доминирующую категорию (самую частую).
        """
        stats = self.get_category_stats(skills)
        if not stats:
            return "other"
        return max(stats, key=stats.get)

    def get_dominant_category_label(self, skills: list[str]) -> str:
        """Возвращает человекочитаемое название доминирующей категории."""
        cat_id = self.get_dominant_category(skills)
        return self._category_info.get(cat_id, {}).get("label", cat_id)
