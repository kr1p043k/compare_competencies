"""Compatibility shim — all logic lives in src.api_pkg.
This file exists so existing tests still import from src.api.
"""
from src.api_pkg import app
from src.api_pkg.deps import (
    basic_vacancies,
    skill_freq,
    student_profiles,
    skill_weights,
    hybrid_weights,
    taxonomy,
    evaluator,
    recommendation_engine,
    clusterer,
    trend_analyzer,
    current_skills_set,
)
