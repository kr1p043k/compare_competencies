"""
Подпакеты анализа:
- comparison  — CompetencyComparator, EmbeddingComparator, DomainAnalyzer
- gap         — GapAnalyzer, ProfileEvaluator
- skills      — SkillFilter, SkillLevelAnalyzer, SkillTaxonomy, SkillCorrelationAnalyzer, TrendAnalyzer
- clustering  — VacancyClusterer
"""

from .clustering.vacancy_clustering import VacancyClusterer
from .comparison.comparator import CompetencyComparator
from .comparison.embedding_comparator import EmbeddingComparator
from .gap.gap_analyzer import GapAnalyzer
from .gap.profile_evaluator import ProfileEvaluator
from .skills.skill_correlation import SkillCorrelationAnalyzer
from .skills.skill_filter import SkillFilter
from .skills.skill_level_analyzer import SkillLevelAnalyzer

__all__ = [
    "CompetencyComparator",
    "EmbeddingComparator",
    "GapAnalyzer",
    "ProfileEvaluator",
    "SkillCorrelationAnalyzer",
    "SkillFilter",
    "SkillLevelAnalyzer",
    "VacancyClusterer",
]
