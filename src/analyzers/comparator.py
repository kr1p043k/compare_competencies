# src/analyzers/comparator.py
from typing import Dict, Set, List
import logging
from src.models.student import StudentProfile
from src.models.comparison import ComparisonReport, GapResult

logger = logging.getLogger(__name__)

class CompetencyComparator:
    """Основной класс сравнения компетенций студента с рынком."""

    def __init__(self, competency_mapping: Dict):
        self.mapping = competency_mapping

    def get_skills_for_student(self, student: StudentProfile) -> Set[str]:
        skills: Set[str] = set()
        for code in student.competencies:
            clean_code = code.strip('. ').upper()
            if clean_code in self.mapping:
                skills.update(self.mapping[clean_code])
            elif code.strip('.') in self.mapping:
                skills.update(self.mapping[code.strip('.')])
        return skills

    def compare(self, student: StudentProfile, market_skills: Dict[str, int]) -> ComparisonReport:
        if not market_skills:
            logger.warning("market_skills пустой")
            return ComparisonReport(
                student_name=student.name,
                total_competencies=len(student.competencies),
                total_mapped_skills=0,
                coverage_percent=0.0,
                weighted_coverage_percent=0.0,
                covered_skills=[],
                high_demand_gaps=[],
                medium_demand_gaps=[],
                low_demand_gaps=[],
                recommendations=["Нет данных о рынке для анализа."]
            )

        student_skills = self.get_skills_for_student(student)
        market_set = set(market_skills.keys())

        covered = student_skills & market_set
        deficits = market_set - student_skills

        coverage = len(covered) / len(market_set)
        total_weight = sum(market_skills.values())
        weighted_coverage = sum(market_skills.get(s, 0) for s in covered) / total_weight

        # Динамические пороги
        sorted_freq = sorted(market_skills.values(), reverse=True)
        high_threshold = sorted_freq[int(len(sorted_freq) * 0.15)] if sorted_freq else 0
        medium_threshold = sorted_freq[int(len(sorted_freq) * 0.40)] if sorted_freq else 0

        high_gaps = [GapResult(skill=s, frequency=market_skills[s], demand_level="high")
                     for s in deficits if market_skills.get(s, 0) >= high_threshold]

        medium_gaps = [GapResult(skill=s, frequency=market_skills[s], demand_level="medium")
                       for s in deficits if medium_threshold <= market_skills.get(s, 0) < high_threshold]

        low_gaps = [GapResult(skill=s, frequency=market_skills[s], demand_level="low")
                    for s in deficits if market_skills.get(s, 0) < medium_threshold]

        recommendations = self._generate_recommendations(high_gaps, student.target_role)

        return ComparisonReport(
            student_name=student.name,
            total_competencies=len(student.competencies),
            total_mapped_skills=len(student_skills),
            coverage_percent=round(coverage * 100, 2),
            weighted_coverage_percent=round(weighted_coverage * 100, 2),
            covered_skills=sorted(list(covered)),
            high_demand_gaps=high_gaps[:15],
            medium_demand_gaps=medium_gaps[:10],
            low_demand_gaps=low_gaps[:10],
            recommendations=recommendations
        )

    def _generate_recommendations(self, high_gaps: List[GapResult], target_role: str) -> List[str]:
        recs = []
        if high_gaps:
            top = [g.skill for g in high_gaps[:5]]
            recs.append(f"Приоритет №1: освоить {', '.join(top)} (высокий спрос на рынке).")
            if any("pytorch" in s.lower() or "tensorflow" in s.lower() for s in top):
                recs.append(f"Для роли {target_role} рекомендуется углубить знания Deep Learning фреймворков.")
        return recs