from pydantic import BaseModel
from typing import List, Tuple, Dict

class GapResult(BaseModel):
    skill: str
    frequency: int
    demand_level: str   # "high", "medium", "low"

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