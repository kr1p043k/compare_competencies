# src/analyzers/gap_analyzer.py
from typing import Dict
import logging
from src.models.student import StudentProfile
from src.models.comparison import ComparisonReport
from .comparator import CompetencyComparator

logger = logging.getLogger(__name__)

class GapAnalyzer:
    """Анализатор дефицитов компетенций."""

    def __init__(self, competency_mapping: Dict):
        self.comparator = CompetencyComparator(competency_mapping)

    def analyze(self, student: StudentProfile, market_skills: Dict[str, int]) -> ComparisonReport:
        logger.info(f"Выполняется gap-анализ для студента: {student.name}")
        report = self.comparator.compare(student, market_skills)
        logger.info(f"Анализ завершён. Покрытие: {report.coverage_percent}% | "
                    f"Взвешенное: {report.weighted_coverage_percent}%")
        return report