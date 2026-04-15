"""
Модель данных студента с поддержкой уровней опыта и оценками
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class ExperienceLevel(str, Enum):
    """Уровни опыта"""
    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"


class StudentProfile(BaseModel):
    """
    Профиль студента с информацией о компетенциях и уровне
    
    Attributes:
        profile_name: Название профиля (base, dc, top_dc)
        competencies: Список кодов компетенций (SS1.1, DL-1.3 и т.д.)
        skills: Список названий навыков (парсированы из competencies)
        target_level: Целевой уровень опыта (junior/middle/senior)
        created_at: Дата создания профиля
    """
    profile_name: str  # base, dc, top_dc
    competencies: List[str] = Field(default_factory=list)  # коды (SS1.1, DL-1.3)
    skills: List[str] = Field(default_factory=list)        # названия навыков
    target_level: ExperienceLevel = ExperienceLevel.MIDDLE
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True
    
    def __repr__(self):
        return f"StudentProfile({self.profile_name}, {self.target_level}, {len(self.competencies)} competencies)"


class ProfileEvaluation(BaseModel):
    """
    Результаты оценки профиля студента
    
    Attributes:
        profile_name: Название профиля (base, dc, top_dc)
        student: Профиль студента
        level: Уровень, на котором оценивается
        raw_score: TF-IDF скор
        confidence: Уверенность в скоре
        coverage: Покрытие навыков (raw и adjusted)
        readiness_score: Готовность студента к уровню (0-100)
        recommendation: Текстовая рекомендация
        gaps: Пробелы в навыках по приоритету
        evaluated_at: Время оценки
    """
    profile_name: str
    student: StudentProfile
    level: ExperienceLevel
    raw_score: float
    confidence: float
    coverage: Dict[str, Any] = Field(default_factory=dict)  # {raw, adjusted, difficulty_multiplier}
    readiness_score: float       # 0-100
    recommendation: str
    gaps: Dict = Field(default_factory=dict)  # {high_priority, medium_priority, low_priority, total, skills}
    evaluated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True
    
    def __repr__(self):
        return f"ProfileEvaluation({self.profile_name}@{self.level}, readiness={self.readiness_score:.2f}%)"


class ProfileComparison(BaseModel):
    """
    Сравнение нескольких профилей студента
    
    Attributes:
        evaluations: Оценки для каждого профиля
        best_evaluation: Лучшая оценка
        average_readiness: Средняя готовность
        compared_at: Время сравнения
    """
    evaluations: List[ProfileEvaluation] = Field(default_factory=list)
    best_evaluation: Optional[ProfileEvaluation] = None
    average_readiness: float = 0.0
    compared_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True
    
    def to_dict_for_json(self) -> Dict:
        """Конвертирует в словарь для сохранения в JSON"""
        return {
            "timestamp": str(self.compared_at),
            "total_profiles_analyzed": len(self.evaluations),
            "average_readiness": round(self.average_readiness, 2),
            "best_profile": {
                "profile": self.best_evaluation.profile_name,
                "level": self.best_evaluation.level,
                "readiness_score": round(self.best_evaluation.readiness_score, 2),
                "adjusted_coverage": round(self.best_evaluation.coverage.get('adjusted', 0), 2),
                "recommendation": self.best_evaluation.recommendation
            } if self.best_evaluation else None,
            "profiles": [
                {
                    "profile": e.profile_name,
                    "level": e.level,
                    "readiness_score": round(e.readiness_score, 2),
                    "coverage_raw": round(e.coverage.get('raw', 0), 2),
                    "coverage_adjusted": round(e.coverage.get('adjusted', 0), 2),
                    "high_gaps": e.gaps.get('high_priority', 0),
                    "medium_gaps": e.gaps.get('medium_priority', 0),
                    "low_gaps": e.gaps.get('low_priority', 0),
                    "recommendation": e.recommendation
                }
                for e in self.evaluations
            ]
        }

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def merge_skills_hierarchically(
    top_skills: List[str],
    middle_skills: List[str],
    base_skills: List[str]
) -> List[str]:
    """
    Объединяет навыки трёх уровней (top_dc, dc, base) в один список,
    сохраняя порядок приоритета и исключая дубликаты.
    
    Используется для профиля 'top_dc', чтобы гарантировать наличие
    всех навыков из более низких уровней.
    
    Args:
        top_skills: Навыки top_dc
        middle_skills: Навыки dc
        base_skills: Навыки base
    
    Returns:
        Объединённый список навыков без дубликатов
    """
    seen = set()
    merged = []
    
    for skill in top_skills:
        if skill not in seen:
            merged.append(skill)
            seen.add(skill)
    
    for skill in middle_skills:
        if skill not in seen:
            merged.append(skill)
            seen.add(skill)
    
    for skill in base_skills:
        if skill not in seen:
            merged.append(skill)
            seen.add(skill)
    
    return merged