"""
Модели для сравнения профилей студентов (актуальная версия 2026)
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from .student import ExperienceLevel, StudentProfile, ProfileEvaluation


class GapResult(BaseModel):
    """Результат по одному gap-навыку"""
    skill: str
    gap_j: float = 0.0
    gap_m: float = 0.0
    gap_s: float = 0.0
    max_gap: float = 0.0
    cluster_relevance: float = 0.0
    demand: float = 0.0
    priority: str = "LOW"   # HIGH, MEDIUM, LOW


class ComparisonReport(BaseModel):
    """Полный отчёт по сравнению нескольких профилей студентов"""
    
    # Основная информация
    timestamp: datetime = Field(default_factory=datetime.now)
    total_profiles: int
    profiles: List[str]  # ['base', 'dc', 'top_dc']
    
    # Агрегированные метрики
    average_readiness: float
    average_market_coverage: float
    average_skill_coverage: float
    average_domain_coverage: float
    
    # Лучший профиль
    best_profile: str
    best_readiness: float
    best_market_coverage: float
    
    # Детальные оценки
    evaluations: List[ProfileEvaluation]
    
    # Общие gaps (топ по приоритету)
    high_priority_gaps: List[GapResult] = Field(default_factory=list)
    medium_priority_gaps: List[GapResult] = Field(default_factory=list)
    low_priority_gaps: List[GapResult] = Field(default_factory=list)
    
    # Кластерная информация
    common_clusters: List[Dict] = Field(default_factory=list)  # кластеры, общие для всех профилей
    
    # Рекомендации
    overall_recommendations: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True

    def to_dict(self) -> Dict:
        """Удобный метод для сохранения в JSON"""
        return {
            "timestamp": str(self.timestamp),
            "total_profiles": self.total_profiles,
            "profiles": self.profiles,
            "average_readiness": round(self.average_readiness, 2),
            "average_market_coverage": round(self.average_market_coverage, 2),
            "average_skill_coverage": round(self.average_skill_coverage, 2),
            "average_domain_coverage": round(self.average_domain_coverage, 2),
            "best_profile": {
                "name": self.best_profile,
                "readiness": round(self.best_readiness, 2),
                "market_coverage": round(self.best_market_coverage, 2)
            },
            "evaluations": [e.dict() for e in self.evaluations],
            "high_priority_gaps": [g.dict() for g in self.high_priority_gaps],
            "medium_priority_gaps": [g.dict() for g in self.medium_priority_gaps],
            "low_priority_gaps": [g.dict() for g in self.low_priority_gaps],
            "overall_recommendations": self.overall_recommendations
        }


# ====================== Вспомогательная модель ======================

class SkillGapSummary(BaseModel):
    """Сводка по gap-анализу для одного профиля"""
    profile_name: str
    total_gaps: int
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int
    top_gaps: List[GapResult] = Field(default_factory=list)