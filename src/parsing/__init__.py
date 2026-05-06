# src/parsing/__init__.py
"""
Пакет парсинга hh.ru + обработка навыков.
"""

from __future__ import annotations

from .skill_normalizer import SkillNormalizer
from .skill_parser import ExtractedSkill, SkillParser, SkillSource
from .skill_validator import SkillValidator, ValidationReason
from .utils import (
    collect_vacancies_multiple,
    extract_and_count_skills,
    filter_skills_by_whitelist,
    interactive_config,
    load_it_skills,
    read_json,
    setup_logging,
    write_json,
)
from .vacancy_parser import VacancyParser

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
