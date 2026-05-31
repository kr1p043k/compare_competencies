"""Tests for domain port interfaces — Protocol structural subtyping."""
from typing import Any

from src import Ok, Err, Result
from src.errors import DomainError, CacheError, DataSourceError
from src.domain.ports import (
    DataProvider,
    CacheProvider,
    Repository,
    SkillProvider,
    VacancyProvider,
    ForecastProvider,
)


class TestProtocols:
    def test_data_protocol(self):
        assert isinstance(ConcreteDataProvider(), DataProvider)

    def test_cache_protocol(self):
        assert isinstance(ConcreteCache(), CacheProvider)

    def test_repository_protocol(self):
        assert isinstance(ConcreteRepo(), Repository)

    def test_skill_protocol(self):
        assert isinstance(ConcreteSkillProvider(), SkillProvider)

    def test_vacancy_protocol(self):
        assert isinstance(ConcreteVacancyProvider(), VacancyProvider)

    def test_forecast_protocol(self):
        prov = ConcreteForecastProvider()
        assert isinstance(prov, ForecastProvider)

    def test_none_data_provider(self):
        assert not isinstance("string", DataProvider)

    def test_none_vacancy_provider(self):
        assert not isinstance(123, VacancyProvider)


class ConcreteDataProvider:
    def get_vacancies(self, queries: list[str], max_pages: int, **kwargs) -> Result[list[dict], DataSourceError]:
        return Ok([])

    def get_student_profiles(self) -> Result[dict[str, list], DomainError]:
        return Ok({})

    def get_reference_data(self, name: str) -> Result[dict, DomainError]:
        return Ok({})


class ConcreteCache:
    def get(self, key: str) -> Result[Any, CacheError]:
        return Ok(None)

    def set(self, key: str, value: Any, ttl: int | None = None) -> Result[None, CacheError]:
        return Ok(None)

    def exists(self, key: str) -> bool:
        return False

    def delete(self, key: str) -> Result[None, CacheError]:
        return Ok(None)

    def clear(self) -> Result[None, CacheError]:
        return Ok(None)


class ConcreteRepo:
    def save(self, entity: Any) -> Result[Any, DomainError]:
        return Ok(entity)

    def find(self, **filters) -> Result[list, DomainError]:
        return Ok([])

    def find_one(self, **filters) -> Result[Any | None, DomainError]:
        return Ok(None)

    def delete(self, entity: Any) -> Result[bool, DomainError]:
        return Ok(True)


class ConcreteSkillProvider:
    def normalize(self, skill: str) -> Result[str, DomainError]:
        return Ok(skill)

    def extract(self, text: str) -> Result[list[str], DomainError]:
        return Ok([])

    def embed(self, skills: list[str]) -> Result[list[list[float]], DomainError]:
        return Ok([])


class ConcreteVacancyProvider:
    def search(self, query: str, area: int, period: int, pages: int) -> Result[list[dict], DataSourceError]:
        return Ok([])

    def get_details(self, vacancy_id: str) -> Result[dict, DataSourceError]:
        return Ok({})

    def get_areas(self) -> Result[list[dict], DataSourceError]:
        return Ok([])


class ConcreteForecastProvider:
    def predict(self, skill: str, history: dict[str, float], months: int) -> Result[dict[str, float], DomainError]:
        return Ok({})

    def top_growing(self, forecasts: list[dict], n: int) -> list[dict]:
        return forecasts[:n]
