# src/analyzers/comparator.py
from typing import Dict, Set, List
from src.models.student import StudentProfile
from src.models.comparison import ComparisonReport, GapResult


class CompetencyComparator:
    """Основной класс для сравнения компетенций студента с рыночными навыками."""

    def __init__(self, competency_mapping: Dict):
        self.mapping = competency_mapping

    def get_skills_for_student(self, student: StudentProfile) -> Set[str]:
        """Преобразует коды компетенций студента в множество рыночных навыков."""
        skills: Set[str] = set()
        for code in student.competencies:
            clean_code = code.strip('. ').upper()
            if clean_code in self.mapping:
                skills.update(self.mapping[clean_code])
            elif code.strip('.') in self.mapping:
                skills.update(self.mapping[code.strip('.')])
        return skills

    def compare(self, student: StudentProfile, market_skills: Dict[str, int]) -> ComparisonReport:
        """Сравнивает профиль студента с рынком и возвращает полный отчёт."""
        student_skills = self.get_skills_for_student(student)
        market_set = set(market_skills.keys())

        covered = student_skills & market_set
        deficits = market_set - student_skills

        coverage = len(covered) / len(market_set) if market_set else 0.0
        total_weight = sum(market_skills.values())
        weighted_coverage = sum(market_skills.get(s, 0) for s in covered) / total_weight if total_weight > 0 else 0.0

        # Динамические пороги
        if market_skills:
            sorted_freq = sorted(market_skills.values(), reverse=True)
            high_threshold = sorted_freq[int(len(sorted_freq) * 0.15)] if sorted_freq else 0
            medium_threshold = sorted_freq[int(len(sorted_freq) * 0.40)] if sorted_freq else 0
        else:
            high_threshold = medium_threshold = 0

        high_gaps = [
            GapResult(skill=s, frequency=market_skills[s], demand_level="high")
            for s in deficits if market_skills.get(s, 0) >= high_threshold
        ]
        medium_gaps = [
            GapResult(skill=s, frequency=market_skills[s], demand_level="medium")
            for s in deficits if medium_threshold <= market_skills.get(s, 0) < high_threshold
        ]
        low_gaps = [
            GapResult(skill=s, frequency=market_skills[s], demand_level="low")
            for s in deficits if market_skills.get(s, 0) < medium_threshold
        ]

        recommendations = self._generate_recommendations(high_gaps)

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

    def _generate_recommendations(self, high_gaps: List[GapResult]) -> List[str]:
        recs = []
        if high_gaps:
            top_skills = [g.skill for g in high_gaps[:5]]
            recs.append(f"Высокий приоритет: освоить {', '.join(top_skills[:4])} — наиболее востребованные навыки.")
        return recs