"""Domain port interfaces — абстракции для внешних зависимостей."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.result import Result
from src.errors import DomainError, CacheError, DataSourceError


@runtime_checkable
class DataProvider(Protocol):
    """Порт доступа к данным."""
    def get_vacancies(self, queries: list[str], max_pages: int, **kwargs) -> Result[list[dict], DataSourceError]:
        """Получить вакансии."""
        ...

    def get_student_profiles(self) -> Result[dict[str, list], DomainError]:
        """Получить профили студентов."""
        ...

    def get_reference_data(self, name: str) -> Result[dict, DomainError]:
        """Получить справочные данные."""
        ...


@runtime_checkable
class CacheProvider(Protocol):
    """Порт кэша."""
    def get(self, key: str) -> Result[Any, CacheError]:
        """Получить значение."""
        ...

    def set(self, key: str, value: Any, ttl: int | None = None) -> Result[None, CacheError]:
        """Сохранить значение."""
        ...

    def exists(self, key: str) -> bool:
        """Проверить наличие."""
        ...

    def delete(self, key: str) -> Result[None, CacheError]:
        """Удалить."""
        ...

    def clear(self) -> Result[None, CacheError]:
        """Очистить."""
        ...


@runtime_checkable
class Repository(Protocol):
    """Порт репозитория."""
    def save(self, entity: Any) -> Result[Any, DomainError]:
        """Сохранить."""
        ...

    def find(self, **filters) -> Result[list, DomainError]:
        """Найти записи."""
        ...

    def find_one(self, **filters) -> Result[Any | None, DomainError]:
        """Найти одну запись."""
        ...

    def delete(self, entity: Any) -> Result[bool, DomainError]:
        """Удалить."""
        ...


@runtime_checkable
class SkillProvider(Protocol):
    """Порт навыков."""
    def normalize(self, skill: str) -> Result[str, DomainError]:
        """Нормализовать навык."""
        ...

    def extract(self, text: str) -> Result[list[str], DomainError]:
        """Извлечь навыки."""
        ...

    def embed(self, skills: list[str]) -> Result[list[list[float]], DomainError]:
        """Эмбеддинги."""
        ...


@runtime_checkable
class VacancyProvider(Protocol):
    """Порт вакансий."""
    def search(self, query: str, area: int, period: int, pages: int) -> Result[list[dict], DataSourceError]:
        """Поиск."""
        ...

    def get_details(self, vacancy_id: str) -> Result[dict, DataSourceError]:
        """Детали."""
        ...

    def get_areas(self) -> Result[list[dict], DataSourceError]:
        """Регионы."""
        ...


@runtime_checkable
class ForecastProvider(Protocol):
    """Порт прогнозов."""
    def predict(self, skill: str, history: dict[str, float], months: int) -> Result[dict[str, float], DomainError]:
        """Прогноз."""
        ...

    def top_growing(self, forecasts: list[dict], n: int) -> list[dict]:
        """Топ растущих."""
        ...
