# src/parsing/__init__.py
"""
Пакет парсинга hh.ru + обработка навыков.
"""

from __future__ import annotations

from .vacancy_parser import VacancyParser
from .skill_parser import SkillParser, ExtractedSkill, SkillSource
from .skill_normalizer import SkillNormalizer
from .skill_validator import SkillValidator, ValidationReason
from .utils import (
    setup_logging,
    load_it_skills,
    filter_skills_by_whitelist,
    collect_vacancies_multiple,
    extract_and_count_skills,
    interactive_config,
    read_json,
    write_json,
)

__all__ = [
    "VacancyParser",
    "SkillParser",
    "ExtractedSkill",
    "SkillSource",
    "SkillNormalizer",
    "SkillValidator",
    "ValidationReason",
    "setup_logging",
    "load_it_skills",
    "filter_skills_by_whitelist",
    "collect_vacancies_multiple",
    "extract_and_count_skills",
    "interactive_config",
    "read_json",
    "write_json",
]