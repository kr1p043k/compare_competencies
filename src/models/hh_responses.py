"""
Pydantic-модели для ответов API hh.ru.
Используются для валидации и строгой типизации данных,
получаемых от внешнего сервиса.
"""

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Вложенные модели (могут использоваться в нескольких ответах)
# ---------------------------------------------------------------------------


class AreaResponse(BaseModel):
    """Регион."""

    id: int
    name: str
    url: str | None = None


class EmployerResponse(BaseModel):
    """Работодатель."""

    id: str | None = None
    name: str
    url: str | None = None
    logo_urls: dict[str, str] | None = None
    trusted: bool = False


class SalaryResponse(BaseModel):
    """Зарплата."""

    from_: int | None = Field(None, alias="from")
    to: int | None = None
    currency: str = "RUB"
    gross: bool = False

    model_config = {"populate_by_name": True}


class SnippetResponse(BaseModel):
    """Фрагменты описания вакансии."""

    requirement: str | None = None
    responsibility: str | None = None


class ExperienceResponse(BaseModel):
    """Требуемый опыт."""

    id: str
    name: str


class KeySkillResponse(BaseModel):
    """Ключевой навык."""

    name: str


# ---------------------------------------------------------------------------
# Ответы на поиск вакансий
# ---------------------------------------------------------------------------


class VacancySearchItem(BaseModel):
    """Элемент списка вакансий при поиске."""

    id: str
    name: str
    area: AreaResponse
    employer: EmployerResponse | None = None
    salary: SalaryResponse | None = None
    snippet: SnippetResponse | None = None
    experience: ExperienceResponse | None = None
    published_at: str | None = None
    url: str | None = None
    alternate_url: str | None = None

    model_config = {"extra": "allow"}


class VacancySearchResponse(BaseModel):
    """Ответ от /vacancies (поиск)."""

    items: list[VacancySearchItem]
    found: int
    pages: int
    page: int
    per_page: int


# ---------------------------------------------------------------------------
# Детали вакансии (более полная модель)
# ---------------------------------------------------------------------------


class VacancyDetailResponse(BaseModel):
    """Полная информация о вакансии (GET /vacancies/{id})."""

    id: str
    name: str
    area: AreaResponse
    employer: EmployerResponse | None = None
    salary: SalaryResponse | None = None
    snippet: SnippetResponse | None = None
    experience: ExperienceResponse | None = None
    description: str | None = None
    key_skills: list[KeySkillResponse] = Field(default_factory=list)
    published_at: str | None = None
    created_at: str | None = None
    archived: bool = False
    url: str | None = None
    alternate_url: str | None = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Ответ с токеном
# ---------------------------------------------------------------------------


class TokenResponse(BaseModel):
    """Ответ от /token (OAuth client_credentials)."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


# ---------------------------------------------------------------------------
# Вспомогательная функция для безопасного парсинга
# ---------------------------------------------------------------------------


def parse_response(data: dict[str, Any], model_class: type[BaseModel]) -> BaseModel:
    """Преобразует сырой словарь в Pydantic-модель, логируя ошибки."""
    try:
        return model_class.model_validate(data)
    except Exception as e:
        import structlog

        logger = structlog.get_logger(__name__)
        logger.warning("response_validation_failed", model=model_class.__name__, error=str(e))
        raise
