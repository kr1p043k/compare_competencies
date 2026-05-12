"""Централизованные перечисления для безопасной типизации проекта."""

from enum import StrEnum


class ExperienceLevel(StrEnum):
    """Уровни опыта (ранее определён в student.py, теперь перенесён сюда)."""

    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"


class PriorityLevel(StrEnum):
    """Приоритеты изучения навыка."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TrendType(StrEnum):
    """Тип тренда навыка."""

    RISING = "rising"
    FALLING = "falling"


class SkillCategory(StrEnum):
    """Категория навыка относительно профиля студента."""

    MISSING = "missing"  # навык отсутствует
    WEAK = "weak"  # навык есть, но слабый
    STRONG = "strong"  # навык уверенный


class ComparisonLevel(StrEnum):
    """Уровень, используемый для сравнения с рынком."""

    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"


class JobSearchLevel(StrEnum):
    """Уровень, используемый при поиске вакансий."""

    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"
