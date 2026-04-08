from pydantic import BaseModel, Field, validator
from typing import Literal, List

class GapResult(BaseModel):
    skill: str
    frequency: int = Field(ge=0)
    demand_level: Literal["high", "medium", "low"]   # строгая валидация

class ComparisonReport(BaseModel):
    student_name: str
    total_competencies: int
    total_mapped_skills: int
    coverage_percent: float
    weighted_coverage_percent: float
    covered_skills: List[str]
    high_demand_gaps: List[GapResult]
    medium_demand_gaps: List[GapResult]
    low_demand_gaps: List[GapResult]
    recommendations: List[str]