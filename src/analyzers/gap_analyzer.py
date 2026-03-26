# src/analyzers/gap_analyzer.py
from typing import Dict
from src.models.student import StudentProfile
from src.models.comparison import ComparisonReport
from .comparator import CompetencyComparator   # ← Важный импорт!


class GapAnalyzer:

    def __init__(self, competency_mapping: Dict):
        self.comparator = CompetencyComparator(competency_mapping)

    def analyze(self, student: StudentProfile, market_skills: Dict[str, int]) -> ComparisonReport:
        """Основной метод анализа."""
        report = self.comparator.compare(student, market_skills)
        return report