import random
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.predictors.base import BasePredictor
from src import Ok, Err, Result

logger = structlog.get_logger(__name__)


@dataclass
class ForecastResult:
    skill: str
    current_frequency: float
    predicted_growth: float
    confidence: float
    next_year_frequency: float


GENE_POOL = [
    "python", "sql", "javascript", "typescript", "java", "go", "rust",
    "docker", "kubernetes", "aws", "machine learning", "ai",
    "react", "node.js", "api", "git", "linux", "postgresql",
    "redis", "kafka", "tensorflow", "pytorch", "fastapi",
]


class GeneticForecastSkill:
    def __init__(self, name: str, current_freq: float):
        self.name = name
        self.genes = {
            "growth_rate": random.uniform(-0.1, 0.3),
            "seasonality": random.uniform(-0.05, 0.05),
            "momentum": random.uniform(0.0, 0.2),
            "maturity": random.uniform(0.0, 1.0),
        }
        self.current_freq = current_freq
        self.fitness_score: float = 0.0

    def predict(self, months: int = 12) -> float:
        base = self.current_freq
        growth = self.genes["growth_rate"] * (months / 12)
        season = self.genes["seasonality"] * (1 if months % 12 < 6 else -1)
        return base * (1 + growth + season + self.genes["momentum"])

    def mutate(self, rate: float = 0.1):
        for key in self.genes:
            if random.random() < rate:
                self.genes[key] += random.uniform(-0.05, 0.05)
                self.genes[key] = max(-1.0, min(1.0, self.genes[key]))


class SkillForecastEngine(BasePredictor):
    def __init__(self):
        self._population: dict[str, GeneticForecastSkill] = {}
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "SkillForecastGA"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, skill_frequencies: dict[str, float] | None = None, **kwargs) -> Result["SkillForecastEngine", Exception]:
        freqs = skill_frequencies or {}
        self._population = {}
        for skill, freq in freqs.items():
            self._population[skill] = GeneticForecastSkill(skill, freq)
        for _ in range(5):
            self._evolve()
        self._is_fitted = True
        logger.info("skill_forecast_fitted", skills=len(self._population))
        return Ok(self)

    def _evolve(self):
        if not self._population:
            return
        for skill_forecast in self._population.values():
            skill_forecast.fitness_score = skill_forecast.predict() / max(skill_forecast.current_freq, 0.01)
        sorted_forecasts = sorted(
            self._population.values(),
            key=lambda x: x.fitness_score,
            reverse=True,
        )
        keep = max(len(sorted_forecasts) // 2, 1)
        survivors = sorted_forecasts[:keep]
        for sf in survivors:
            sf.mutate(rate=0.2)

    def forecast(self, skill: str, months: int = 12) -> ForecastResult | None:
        sf = self._population.get(skill)
        if sf is None:
            return None
        predicted = sf.predict(months)
        growth = (predicted - sf.current_freq) / max(sf.current_freq, 0.01)
        return ForecastResult(
            skill=skill,
            current_frequency=sf.current_freq,
            predicted_growth=growth,
            confidence=min(abs(growth) * 2, 0.95),
            next_year_frequency=predicted,
        )

    def forecast_all(self, months: int = 12) -> list[ForecastResult]:
        return [
            result for skill in self._population
            if (result := self.forecast(skill, months)) is not None
        ]

    def top_growing(self, n: int = 10, months: int = 12) -> list[ForecastResult]:
        results = self.forecast_all(months)
        results.sort(key=lambda x: x.predicted_growth, reverse=True)
        return results[:n]
