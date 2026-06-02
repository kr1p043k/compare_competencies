"""Pydantic-схемы для валидации сырых JSON-вакансий с hh.ru."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class KeySkillSchema(BaseModel):
    name: str
    id: str | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("skill name cannot be empty")
        return stripped


class SalarySchema(BaseModel):
    from_amount: int | None = Field(None, alias="from")
    to_amount: int | None = Field(None, alias="to")
    currency: str = "RUB"
    gross: bool | None = None


class AreaSchema(BaseModel):
    id: int
    name: str


class EmployerSchema(BaseModel):
    id: int | str
    name: str
    url: str | None = None


class SnippetSchema(BaseModel):
    requirement: str | None = None
    responsibility: str | None = None


class ExperienceSchema(BaseModel):
    id: str
    name: str


class VacancySchema(BaseModel):
    id: int
    name: str
    area: AreaSchema | None = None
    employer: EmployerSchema | None = None
    salary: SalarySchema | None = None
    experience: ExperienceSchema | None = None
    snippet: SnippetSchema | None = None
    key_skills: list[KeySkillSchema] = Field(default_factory=list, alias="keySkills")
    description: str | None = None
    created_at: datetime | None = Field(None, alias="created_at")
    published_at: datetime | None = Field(None, alias="published_at")
    alternate_url: str | None = Field(None, alias="alternate_url")
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vacancy name cannot be empty")
        return v.strip()

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "VacancySchema":
        return cls.model_validate(data)
