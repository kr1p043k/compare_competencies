"""Типизированные модели для работы с вакансиями."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class KeySkill:
    """
    Ключевой навык из API hh.ru

    Attributes:
        name: Название навыка (нормализованное)
        id: ID навыка в hh.ru (опционально)
    """

    name: str
    id: str | None = None

    def __post_init__(self):
        """Нормализация имени при создании"""
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Название навыка не может быть пустым")

    def __hash__(self):
        return hash(self.name.lower())

    def __eq__(self, other):
        if isinstance(other, KeySkill):
            return self.name.lower() == other.name.lower()
        return False

    def __repr__(self):
        return f"KeySkill('{self.name}')"


@dataclass(slots=True)
class Snippet:
    """
    Краткая информация о вакансии (требования и обязанности)

    Attributes:
        requirement: Требования к кандидату
        responsibility: Обязанности
    """

    requirement: str | None = None
    responsibility: str | None = None

    def has_content(self) -> bool:
        """Проверяет, есть ли контент"""
        return bool(self.requirement or self.responsibility)

    def get_full_text(self) -> str:
        """Получает полный текст требований и обязанностей"""
        parts = []
        if self.requirement:
            parts.append(self.requirement)
        if self.responsibility:
            parts.append(self.responsibility)
        return "\n".join(parts)


@dataclass(slots=True)
class Salary:
    """
    Информация о заработной плате

    Attributes:
        from_amount: Минимальная зарплата
        to_amount: Максимальная зарплата
        currency: Валюта (по умолчанию RUB)
    """

    from_amount: int | None = None
    to_amount: int | None = None
    currency: str = "RUB"

    def get_midpoint(self) -> int | None:
        """Возвращает среднее значение зарплаты"""
        if self.from_amount is not None and self.to_amount is not None:
            return (self.from_amount + self.to_amount) // 2
        return self.from_amount if self.from_amount is not None else self.to_amount

    def __repr__(self):
        if self.from_amount and self.to_amount:
            return f"{self.from_amount}-{self.to_amount} {self.currency}"
        elif self.from_amount:
            return f"от {self.from_amount} {self.currency}"
        elif self.to_amount:
            return f"до {self.to_amount} {self.currency}"
        return "Не указана"


@dataclass(slots=True)
class Area:
    """
    Регион (область)

    Attributes:
        id: ID региона в hh.ru
        name: Название региона
    """

    id: int
    name: str

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Area):
            return self.id == other.id
        return False

    def __repr__(self):
        return f"{self.name} (ID {self.id})"


@dataclass(slots=True)
class Employer:
    """
    Работодатель

    Attributes:
        id: ID работодателя в hh.ru
        name: Название компании
        url: URL профиля компании
    """

    id: str
    name: str
    url: str | None = None

    def __repr__(self):
        return f"{self.name}"


@dataclass(slots=True)
class Experience:
    """
    Требуемый опыт работы

    Attributes:
        id: ID опыта в hh.ru (например 'between3and6')
        name: Название опыта (например 'От 3 до 6 лет')
    """

    id: str
    name: str

    def get_level(self) -> str:
        """
        Преобразует ID опыта в уровень (junior/middle/senior)

        Returns:
            'junior', 'middle', 'senior' или 'unknown'
        """
        if not self.id:
            return "middle"

        exp_id = self.id.lower()

        # junior: нет опыта, менее 1 года
        if any(x in exp_id for x in ["no_experience", "less1", "junior"]):
            return "junior"

        # middle: 1-6 лет
        elif any(x in exp_id for x in ["between1and3", "between3and6", "middle"]):
            return "middle"

        # senior: 6+ лет
        elif any(x in exp_id for x in ["between6and10", "morethan10", "senior"]):
            return "senior"

        return "middle"  # default

    def __repr__(self):
        return f"Experience({self.get_level()}: {self.name})"


@dataclass(slots=True)
class Vacancy:
    """
    ПОЛНАЯ модель вакансии с валидацией и типизацией

    Attributes:
        id: Уникальный ID вакансии в hh.ru
        name: Название вакансии (позиция)
        area: Регион
        employer: Работодатель
        key_skills: Список ключевых навыков
        description: Полное описание вакансии
        snippet: Краткая информация (требования/обязанности)
        salary: Информация о зарплате
        experience: Требуемый опыт
        published_at: Дата публикации
        experience_level: Нормализованный уровень (junior/middle/senior)
        raw_data: Сырые данные для debug (должны быть только если нужны)
    """

    # Обязательные поля
    id: str
    name: str
    area: Area
    employer: Employer

    # Опциональные поля
    key_skills: list[KeySkill] = field(default_factory=list)
    description: str | None = None
    snippet: Snippet | None = None
    salary: Salary | None = None
    experience: Experience | None = None
    published_at: str | None = None

    # Вычисляемое поле (зависит от experience)
    experience_level: str = field(default="middle")

    # Служебные поля (NOT для use в основной логике!)
    raw_data: dict[str, Any] = field(default_factory=dict, repr=False)
    parsed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Валидация при создании"""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("ID вакансии должен быть непустой строкой")
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Название вакансии должно быть непустой строкой")

        # Вычисляем experience_level на основе experience
        if self.experience:
            self.experience_level = self.experience.get_level()
        else:
            self.experience_level = "middle"

        try:
            logger.debug(
                "vacancy_created",
                vacancy_id=self.id,
                name=self.name,
                level=self.experience_level,
                skills_count=len(self.key_skills),
            )
        except Exception:
            pass

    @classmethod
    def from_api(cls, data: dict) -> "Vacancy":
        """
        Преобразует сырые данные API hh.ru в типизированный объект

        Args:
            data: Сырой словарь от API

        Returns:
            Валидный объект Vacancy

        Raises:
            ValueError: Если данные невалидны
        """
        try:
            # Парсим обязательные поля
            vacancy_id = data.get("id")
            if not vacancy_id:
                raise ValueError("Отсутствует ID вакансии")

            name = data.get("name", "")
            if not name:
                raise ValueError("Отсутствует название вакансии")

            # Парсим область
            area_data = data.get("area", {})
            area = Area(id=area_data.get("id", 0), name=area_data.get("name", "Unknown"))

            # Парсим работодателя
            employer_data = data.get("employer", {})
            employer = Employer(
                id=employer_data.get("id", ""), name=employer_data.get("name", "Unknown"), url=employer_data.get("url")
            )

            # Парсим ключевые навыки
            key_skills = []
            invalid_skills = 0
            for skill_data in data.get("key_skills", []):
                try:
                    skill_name = skill_data.get("name", "").strip()
                    if skill_name:
                        key_skills.append(KeySkill(name=skill_name, id=skill_data.get("id")))
                except ValueError as e:
                    logger.warning("invalid_skill_in_vacancy", vacancy_id=vacancy_id, error=str(e))
                    invalid_skills += 1
                    continue

            if invalid_skills > 0:
                logger.debug(
                    "some_skills_invalid",
                    vacancy_id=vacancy_id,
                    invalid=invalid_skills,
                    valid=len(key_skills),
                )

            # Парсим snippet
            snippet_data = data.get("snippet", {})
            snippet = None
            if snippet_data:
                snippet = Snippet(
                    requirement=snippet_data.get("requirement"), responsibility=snippet_data.get("responsibility")
                )

            # Парсим зарплату
            salary = None
            salary_data = data.get("salary")
            if salary_data:
                salary = Salary(
                    from_amount=salary_data.get("from"),
                    to_amount=salary_data.get("to"),
                    currency=salary_data.get("currency", "RUB"),
                )

            # Парсим опыт
            experience = None
            experience_data = data.get("experience")
            if experience_data:
                experience = Experience(id=experience_data.get("id", ""), name=experience_data.get("name", "Не указан"))

            # Создаём вакансию
            vacancy = cls(
                id=vacancy_id,
                name=name,
                area=area,
                employer=employer,
                key_skills=key_skills,
                description=data.get("description"),
                snippet=snippet,
                salary=salary,
                experience=experience,
                published_at=data.get("published_at"),
                raw_data=data,
            )

            try:
                logger.debug(
                    "vacancy_parsed_from_api",
                    vacancy_id=vacancy_id,
                    name=name,
                    skills=len(key_skills),
                    has_description=bool(data.get("description")),
                    has_salary=salary is not None,
                )
            except Exception:
                pass

            return vacancy

        except (KeyError, TypeError) as e:
            logger.error("vacancy_parsing_failed", error=str(e), exc_info=True)
            raise ValueError(f"Невалидная структура вакансии: {e}") from e

    def get_all_text(self) -> str:
        """Получает весь текст вакансии для парсинга"""
        parts = [self.name]

        if self.description:
            parts.append(self.description)

        if self.snippet and self.snippet.has_content():
            parts.append(self.snippet.get_full_text())

        return "\n".join(parts)

    def get_skill_names(self) -> list[str]:
        """Получает список имён навыков"""
        return [skill.name for skill in self.key_skills]

    def has_skills(self) -> bool:
        """Проверяет, есть ли навыки"""
        return len(self.key_skills) > 0

    def __repr__(self):
        return f"Vacancy('{self.name}' @ {self.employer.name}, {self.experience_level}, ID={self.id})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Vacancy):
            return self.id == other.id
        return False


@dataclass(slots=True)
class VacancyCollection:
    """
    Коллекция вакансий с полезными методами

    Attributes:
        vacancies: Список вакансий
        query: Поисковый запрос (если применимо)
        fetched_at: Дата загрузки
    """

    vacancies: list[Vacancy] = field(default_factory=list)
    query: str | None = None
    fetched_at: datetime = field(default_factory=datetime.now)

    def __len__(self) -> int:
        return len(self.vacancies)

    def __iter__(self):
        return iter(self.vacancies)

    def add(self, vacancy: Vacancy) -> None:
        """Добавляет вакансию (избегает дубликатов)"""
        if vacancy not in self.vacancies:
            self.vacancies.append(vacancy)
            logger.debug("vacancy_added_to_collection", vacancy_id=vacancy.id, total=len(self.vacancies))
        else:
            logger.debug("vacancy_already_in_collection", vacancy_id=vacancy.id)

    def get_all_skills(self) -> list[KeySkill]:
        """Получает все уникальные навыки из всех вакансий"""
        skills_set = set()
        for vacancy in self.vacancies:
            skills_set.update(vacancy.key_skills)
        logger.debug("unique_skills_collected", total=len(skills_set), vacancies=len(self.vacancies))
        return list(skills_set)

    def get_stats(self) -> dict[str, Any]:
        """Возвращает статистику по коллекции"""
        skills_count = len(self.get_all_skills())
        vacancies_with_skills = sum(1 for v in self.vacancies if v.has_skills())

        # Статистика по уровням
        level_stats = {
            "junior": sum(1 for v in self.vacancies if v.experience_level == "junior"),
            "middle": sum(1 for v in self.vacancies if v.experience_level == "middle"),
            "senior": sum(1 for v in self.vacancies if v.experience_level == "senior"),
        }

        stats = {
            "total_vacancies": len(self),
            "vacancies_with_skills": vacancies_with_skills,
            "total_unique_skills": skills_count,
            "avg_skills_per_vacancy": skills_count / max(vacancies_with_skills, 1),
            "by_level": level_stats,
        }

        logger.info(
            "collection_stats",
            **stats,
        )

        return stats

    def __repr__(self):
        return f"VacancyCollection({len(self)} vacancies, query='{self.query}')"
