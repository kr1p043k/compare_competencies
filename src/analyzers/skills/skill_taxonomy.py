"""
Таксономия навыков - иерархическая классификация IT-навыков по категориям.
Singleton, загружается один раз при первом обращении.
"""

import json
from typing import ClassVar, Optional

import structlog

from src import Err, Ok, Result, config
from src.errors import DomainError

logger = structlog.get_logger(__name__)


class SkillTaxonomy:
    """
    Категоризация навыков на основе skill_taxonomy.json.
    Используется для интерпретации кластеров, улучшения рекомендаций и визуализации.
    """

    _instance: Optional["SkillTaxonomy"] = None
    _taxonomy: dict | None = None
    _skill_to_category: ClassVar[dict[str, str]] = {}  # python → programming_languages
    _skill_to_categories: ClassVar[dict[str, list[str]]] = {}  # python → [categories] (2+)
    _category_info: ClassVar[dict[str, dict[str, str]]] = {}  # programming_languages → {label, icon}
    _category_skills: ClassVar[dict[str, list[str]]] = {}  # programming_languages → [skills] (dedup)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Загружает таксономию из JSON-файла."""
        path = config.SKILL_TAXONOMY_PATH
        if not path.exists():
            logger.warning("taxonomy_file_not_found", path=str(path))
            return

        try:
            with open(path, encoding="utf-8") as f:
                self._taxonomy = json.load(f)

            # Строим обратный маппинг: skill_lower → category_id.
            #
            # Навык может лежать в нескольких категориях (javascript - и
            # frontend, и programming_languages). Правило разрешения -
            # ПОСЛЕДНЯЯ категория в файле: это историческое поведение
            # (раньше ключ просто перезаписывался), и на него опираются
            # дедап ролей и имена кластеров. Менять его молча нельзя, поэтому
            # оно зафиксировано тестом. Все варианты сохраняются в
            # _skill_to_categories - их видно через get_categories_for_skill().
            # Дубли внутри категории схлопываются по нормализованному ключу
            # (Java/java), чтобы счётчики и выборки не раздувались.
            categories = self._taxonomy.get("categories", {})
            for cat_id, cat_data in categories.items():
                self._category_info[cat_id] = {"label": cat_data.get("label", cat_id), "icon": cat_data.get("icon", "")}
                deduped: list[str] = []
                seen: set[str] = set()
                for skill in cat_data.get("skills", []):
                    if not isinstance(skill, str) or not skill.strip():
                        continue
                    key = skill.lower().strip()
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(skill)
                    self._skill_to_categories.setdefault(key, []).append(cat_id)
                    self._skill_to_category[key] = cat_id
                self._category_skills[cat_id] = deduped

            ambiguous = sum(1 for cats in self._skill_to_categories.values() if len(cats) > 1)
            if ambiguous:
                logger.warning(
                    "taxonomy_ambiguous_skills",
                    count=ambiguous,
                    note="skill is listed in 2+ categories; last category in file wins",
                )

            logger.info(
                "taxonomy_loaded",
                categories_count=len(categories),
                skills_count=len(self._skill_to_category),
                ambiguous_count=ambiguous,
            )
        except Exception:
            logger.exception("taxonomy_load_error")

    def get_category(self, skill: str) -> str:
        """
        Возвращает ID категории навыка.
        Пример: get_category('Python') → 'programming_languages'
        """
        return self._skill_to_category.get(skill.lower().strip(), "other")

    def get_categories_for_skill(self, skill: str) -> list[str]:
        """Все категории навыка в порядке файла (пустой список - навык неизвестен).

        Для неоднозначных навыков (javascript) здесь будет несколько значений,
        тогда как get_category() отдаёт только первую категорию.
        """
        return list(self._skill_to_categories.get(skill.lower().strip(), []))

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

    def get_skills_in_category(self, category_id: str) -> Result[list[str], DomainError]:
        """
        Возвращает все навыки указанной категории.
        """
        if self._taxonomy is None:
            return Err(DomainError("Taxonomy not loaded"))
        # Отдаём дедуплицированный по нормализованному ключу список: в файле
        # встречаются пары вида "Java"/"java", они не должны раздувать выборку.
        return Ok(list(self._category_skills.get(category_id, [])))

    def get_all_categories(self) -> Result[list[str], DomainError]:
        """Возвращает список всех ID категорий."""
        return Ok(list(self._category_info.keys()))

    def get_category_stats(self, skills: list[str]) -> Result[dict[str, int], DomainError]:
        """
        Подсчитывает количество навыков по категориям.
        Возвращает {category_id: count}.
        """
        stats: dict[str, int] = {}
        for skill in skills:
            cat = self.get_category(skill)
            stats[cat] = stats.get(cat, 0) + 1
        return Ok(dict(sorted(stats.items(), key=lambda x: x[1], reverse=True)))

    def get_dominant_category(self, skills: list[str]) -> str:
        """
        Возвращает доминирующую категорию (самую частую).
        """
        result = self.get_category_stats(skills)
        if result.is_err() or not result.unwrap():
            return "other"
        return max(result.unwrap(), key=result.unwrap().get)

    def get_dominant_category_label(self, skills: list[str]) -> str:
        """Возвращает человекочитаемое название доминирующей категории."""
        cat_id = self.get_dominant_category(skills)
        return self._category_info.get(cat_id, {}).get("label", cat_id)
