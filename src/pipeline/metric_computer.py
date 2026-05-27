"""MetricComputer — оценивает профили студентов."""

import structlog
from tqdm import tqdm

from src import DomainError, Err, Ok, Result
from src.analyzers.gap.profile_evaluator import ProfileEvaluator

logger = structlog.get_logger("metric_computer")


class MetricComputer:
    """Оценивает готовность профилей."""

    def __init__(self, skill_weights: dict, vacancies_skills: list, level_vacancies_data: list, hybrid_weights: dict):
        self.skill_weights = skill_weights
        self.vacancies_skills = vacancies_skills
        self.level_vacancies_data = level_vacancies_data
        self.hybrid_weights = hybrid_weights
        self.evaluator = None

    def prepare(self, skill_weights_by_level: dict) -> Result[None, DomainError]:
        try:
            self.evaluator = ProfileEvaluator(
                skill_weights=self.skill_weights,
                vacancies_skills=self.vacancies_skills,
                vacancies_skills_dict=self.level_vacancies_data,
                hybrid_weights=self.hybrid_weights,
                skill_weights_by_level=skill_weights_by_level,
            )
            return Ok(None)
        except Exception as e:
            return Err(DomainError(message=f"Ошибка подготовки оценщика: {e}"))

    def compute(self, profiles: dict) -> Result[dict, DomainError]:
        if self.evaluator is None:
            return Err(DomainError(message="Сначала вызовите prepare()"))
        try:
            evaluations = {}
            with tqdm(total=len(profiles), desc="Оценка профилей") as pbar:
                for pname, student in profiles.items():
                    evaluations[pname] = self.evaluator.evaluate_profile(student)
                    pbar.update(1)
            return Ok(evaluations)
        except Exception as e:
            return Err(DomainError(message=f"Ошибка оценки профилей: {e}"))
