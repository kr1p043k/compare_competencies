"""
Типизированные модели для работы с вакансиями.
Замена сырых Dict на структурированные классы.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class KeySkill:
    """
    Ключевой навык из API hh.ru
    
    Attributes:
        name: Название навыка (нормализованное)
        id: ID навыка в hh.ru (опционально)
    """
    name: str
    id: Optional[str] = None
    
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


@dataclass
class Snippet:
    """
    Краткая информация о вакансии (требования и обязанности)
    
    Attributes:
        requirement: Требования к кандидату
        responsibility: Обязанности
    """
    requirement: Optional[str] = None
    responsibility: Optional[str] = None
    
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


@dataclass
class Salary:
    """
    Информация о заработной плате
    
    Attributes:
        from_amount: Минимальная зарплата
        to_amount: Максимальная зарплата
        currency: Валюта (по умолчанию RUB)
    """
    from_amount: Optional[int] = None
    to_amount: Optional[int] = None
    currency: str = "RUB"
    
    def get_midpoint(self) -> Optional[int]:
        """Возвращает среднее значение зарплаты"""
        if self.from_amount and self.to_amount:
            return (self.from_amount + self.to_amount) // 2
        return self.from_amount or self.to_amount
    
    def __repr__(self):
        if self.from_amount and self.to_amount:
            return f"{self.from_amount}-{self.to_amount} {self.currency}"
        elif self.from_amount:
            return f"от {self.from_amount} {self.currency}"
        elif self.to_amount:
            return f"до {self.to_amount} {self.currency}"
        return "Не указана"


@dataclass
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


@dataclass
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
    url: Optional[str] = None
    
    def __repr__(self):
        return f"{self.name}"


@dataclass
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
            return 'middle'
        
        exp_id = self.id.lower()
        
        # junior: нет опыта, менее 1 года
        if any(x in exp_id for x in ['no_experience', 'less1', 'junior']):
            return 'junior'
        
        # middle: 1-6 лет
        elif any(x in exp_id for x in ['between1and3', 'between3and6', 'middle']):
            return 'middle'
        
        # senior: 6+ лет
        elif any(x in exp_id for x in ['between6and10', 'morethan10', 'senior']):
            return 'senior'
        
        return 'middle'  # default
    
    def __repr__(self):
        return f"Experience({self.get_level()}: {self.name})"


@dataclass
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
    key_skills: List[KeySkill] = field(default_factory=list)
    description: Optional[str] = None
    snippet: Optional[Snippet] = None
    salary: Optional[Salary] = None
    experience: Optional[Experience] = None
    published_at: Optional[str] = None
    
    # Вычисляемое поле (зависит от experience)
    experience_level: str = field(default='middle')
    
    # Служебные поля (NOT для use в основной логике!)
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
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
            self.experience_level = 'middle'
    
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
            vacancy_id = data.get('id')
            if not vacancy_id:
                raise ValueError("Отсутствует ID вакансии")
            
            name = data.get('name', '')
            if not name:
                raise ValueError("Отсутствует название вакансии")
            
            # Парсим область
            area_data = data.get('area', {})
            area = Area(
                id=area_data.get('id', 0),
                name=area_data.get('name', 'Unknown')
            )
            
            # Парсим работодателя
            employer_data = data.get('employer', {})
            employer = Employer(
                id=employer_data.get('id', ''),
                name=employer_data.get('name', 'Unknown'),
                url=employer_data.get('url')
            )
            
            # Парсим ключевые навыки
            key_skills = []
            for skill_data in data.get('key_skills', []):
                try:
                    skill_name = skill_data.get('name', '').strip()
                    if skill_name:
                        key_skills.append(KeySkill(
                            name=skill_name,
                            id=skill_data.get('id')
                        ))
                except ValueError as e:
                    logger.warning(f"Невалидный навык в вакансии {vacancy_id}: {e}")
                    continue
            
            # Парсим snippet
            snippet_data = data.get('snippet', {})
            snippet = None
            if snippet_data:
                snippet = Snippet(
                    requirement=snippet_data.get('requirement'),
                    responsibility=snippet_data.get('responsibility')
                )
            
            # Парсим зарплату
            salary = None
            salary_data = data.get('salary')
            if salary_data:
                salary = Salary(
                    from_amount=salary_data.get('from'),
                    to_amount=salary_data.get('to'),
                    currency=salary_data.get('currency', 'RUB')
                )
            
            # Парсим опыт
            experience = None
            experience_data = data.get('experience')
            if experience_data:
                experience = Experience(
                    id=experience_data.get('id', ''),
                    name=experience_data.get('name', 'Не указан')
                )
            
            # Создаём вакансию
            return cls(
                id=vacancy_id,
                name=name,
                area=area,
                employer=employer,
                key_skills=key_skills,
                description=data.get('description'),
                snippet=snippet,
                salary=salary,
                experience=experience,
                published_at=data.get('published_at'),
                raw_data=data
            )
        
        except (KeyError, TypeError) as e:
            raise ValueError(f"Невалидная структура вакансии: {e}")
    
    def get_all_text(self) -> str:
        """Получает весь текст вакансии для парсинга"""
        parts = [self.name]
        
        if self.description:
            parts.append(self.description)
        
        if self.snippet and self.snippet.has_content():
            parts.append(self.snippet.get_full_text())
        
        return "\n".join(parts)
    
    def get_skill_names(self) -> List[str]:
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


@dataclass
class VacancyCollection:
    """
    Коллекция вакансий с полезными методами
    
    Attributes:
        vacancies: Список вакансий
        query: Поисковый запрос (если применимо)
        fetched_at: Дата загрузки
    """
    vacancies: List[Vacancy] = field(default_factory=list)
    query: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.now)
    
    def __len__(self) -> int:
        return len(self.vacancies)
    
    def __iter__(self):
        return iter(self.vacancies)
    
    def add(self, vacancy: Vacancy) -> None:
        """Добавляет вакансию (избегает дубликатов)"""
        if vacancy not in self.vacancies:
            self.vacancies.append(vacancy)
    
    def get_all_skills(self) -> List[KeySkill]:
        """Получает все уникальные навыки из всех вакансий"""
        skills_set = set()
        for vacancy in self.vacancies:
            skills_set.update(vacancy.key_skills)
        return list(skills_set)
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по коллекции"""
        skills_count = len(self.get_all_skills())
        vacancies_with_skills = sum(1 for v in self.vacancies if v.has_skills())
        
        # Статистика по уровням
        level_stats = {
            'junior': sum(1 for v in self.vacancies if v.experience_level == 'junior'),
            'middle': sum(1 for v in self.vacancies if v.experience_level == 'middle'),
            'senior': sum(1 for v in self.vacancies if v.experience_level == 'senior'),
        }
        
        return {
            'total_vacancies': len(self),
            'vacancies_with_skills': vacancies_with_skills,
            'total_unique_skills': skills_count,
            'avg_skills_per_vacancy': skills_count / max(vacancies_with_skills, 1),
            'by_level': level_stats
        }
    
    def __repr__(self):
        return f"VacancyCollection({len(self)} vacancies, query='{self.query}')"