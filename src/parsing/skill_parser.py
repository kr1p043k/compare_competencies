"""
Типизированный парсер навыков с полным логированием.
Отвечает ТОЛЬКО за извлечение навыков, не за валидацию!
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict
from enum import Enum
import re
import logging
from src.models.vacancy import Vacancy

logger = logging.getLogger(__name__)


class SkillSource(Enum):
    """Источник извлечения навыка"""
    KEY_SKILLS = "key_skills"
    DESCRIPTION = "description"
    SNIPPET_REQUIREMENT = "snippet_req"
    SNIPPET_RESPONSIBILITY = "snippet_resp"
    MARKER = "marker"
    REGEX = "regex"
    NGRAM = "ngram"
    DIRECT = "direct"


@dataclass
class ExtractedSkill:
    """Извлечённый из текста навык с метаданными"""
    text: str
    source: SkillSource
    raw_match: str = ""
    confidence: float = 1.0
    
    def __hash__(self):
        return hash(self.text.lower())
    
    def __eq__(self, other):
        if isinstance(other, ExtractedSkill):
            return self.text.lower() == other.text.lower()
        return False
    
    def __repr__(self):
        return f"ExtractedSkill('{self.text}', source={self.source.value}, conf={self.confidence:.2f})"


@dataclass
class ParsingStats:
    """Статистика парсинга"""
    total_extracted: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)
    text_length_processed: int = 0
    
    def __repr__(self):
        sources_str = ", ".join(f"{k}={v}" for k, v in sorted(self.by_source.items()))
        return f"ParsingStats(total={self.total_extracted}, sources=[{sources_str}])"


class SkillParser:
    """
    Парсер для извлечения навыков из вакансий.
    Фокусируется на ИЗВЛЕЧЕНИИ, не на фильтрации.
    """
    
    TECH_SKILLS = {
        "python", "python3", "py", "java", "javascript", "js", "typescript", "ts",
        "c++", "cpp", "c#", "csharp", "php", "ruby", "go", "golang", "rust", "swift",
        "kotlin", "scala", "r", "matlab", "sql", "nosql",
        "django", "flask", "fastapi", "spring", "spring boot", "react", "angular", "vue",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "docker", "kubernetes", "k8s", "jenkins", "git", "gitlab", "github",
        "aws", "azure", "gcp", "yandex cloud",
        "machine learning", "deep learning", "nlp", "data science", "big data", "etl",
    }
    
    SKILL_MARKERS = [
        "ключевые навыки", "ключевые компетенции", "требования", "требования к кандидату",
        "необходимые навыки", "мы ждем", "ожидаем от вас", "что нужно знать",
        "профессиональные навыки", "опыт работы с", "знание", "владение", "умение",
        "должен знать", "должен уметь", "stack", "технологии", "инструменты"
    ]
    
    # Слова, которые указывают на отрицание
    NEGATION_WORDS = ["не требуется", "не нужно", "не обязательно", "не критично", "не важно"]
    
    def __init__(self):
        self.stats = ParsingStats()
        self.logger = logging.getLogger(__name__)
    
    def parse_vacancy(self, vacancy: Vacancy) -> List[ExtractedSkill]:
        """Извлекает все навыки из вакансии"""
        skills = []
        
        skills.extend(self._extract_from_key_skills(vacancy))
        
        if vacancy.snippet:
            if vacancy.snippet.requirement:
                skills.extend(self._extract_from_text(
                    vacancy.snippet.requirement,
                    source=SkillSource.SNIPPET_REQUIREMENT,
                    max_text_length=500
                ))
            if vacancy.snippet.responsibility:
                skills.extend(self._extract_from_text(
                    vacancy.snippet.responsibility,
                    source=SkillSource.SNIPPET_RESPONSIBILITY,
                    max_text_length=500
                ))
        
        if vacancy.description:
            skills.extend(self._extract_from_text(
                vacancy.description,
                source=SkillSource.DESCRIPTION,
                max_text_length=10000
            ))
        
        self.stats.total_extracted += len(skills)
        
        return skills
    
    def _extract_from_key_skills(self, vacancy: Vacancy) -> List[ExtractedSkill]:
        """Извлекает из официального поля key_skills"""
        skills = []
        
        for key_skill in vacancy.key_skills:
            skill = ExtractedSkill(
                text=key_skill.name,
                source=SkillSource.KEY_SKILLS,
                raw_match=key_skill.name,
                confidence=1.0
            )
            skills.append(skill)
            self._update_stats(SkillSource.KEY_SKILLS)
        
        return skills
    
    def _extract_from_text(
        self,
        text: str,
        source: SkillSource,
        max_text_length: int = 1000
    ) -> List[ExtractedSkill]:
        """Извлекает навыки из текста несколькими методами"""
        if not text:
            return []
        
        text = text[:max_text_length]
                # === очистка HTML ===
        text = re.sub(r'<[^>]+>', ' ', text)          # удаляем все теги
        text = re.sub(r'\s+', ' ', text).strip()      # схлопываем пробелы
        text = text.replace('strong', '')             # убираем остатки "strong"
        self.stats.text_length_processed += len(text)
        
        skills = []
        
        skills.extend(self._direct_search(text, source))
        skills.extend(self._marker_search(text, source))
        skills.extend(self._regex_search(text, source))
        
        return skills
    
    def _direct_search(self, text: str, source: SkillSource) -> List[ExtractedSkill]:
        """Прямой поиск по списку TECH_SKILLS с контролем контекста"""
        skills = []
        text_lower = text.lower()
        
        # Расширяем список техническими фразами
        extended_skills = [
            "machine learning", "deep learning", "data science",
            "full stack", "rest api", "sql database",
            "spring boot", "scikit-learn"
        ]
        
        all_skills = list(self.TECH_SKILLS) + extended_skills
        
        # Сортируем по длине (длинные первыми)
        all_skills = sorted(set(all_skills), key=len, reverse=True)
        
        for tech in all_skills:
            pattern = rf'\b{re.escape(tech)}\b'
            match = re.search(pattern, text_lower, re.IGNORECASE)
            
            if match:
                # Проверяем контекст (не в "не требуется")
                start = max(0, match.start() - 50)
                context = text_lower[start:match.end() + 50]
                
                # Проверяем отрицание
                has_negation = any(
                    neg in context for neg in self.NEGATION_WORDS
                )
                
                if not has_negation:
                    skills.append(ExtractedSkill(
                        text=tech,
                        source=source,
                        raw_match=tech,
                        confidence=0.95
                    ))
                    self._update_stats(source)
        
        return skills
    
    def _marker_search(self, text: str, source: SkillSource) -> List[ExtractedSkill]:
        """Поиск навыков после маркеров"""
        skills = []
        text_lower = text.lower()
        
        for marker in self.SKILL_MARKERS:
            if marker not in text_lower:
                continue
            
            parts = text_lower.split(marker)
            if len(parts) < 2:
                continue
            
            after_marker = parts[1][:600]
            lines = re.split(r'[\n,•\-*;]', after_marker)
            
            for line in lines:
                line = line.strip()
                
                if 3 >= len(line) >= 100:
                    continue
                
                if any(tech in line for tech in self.TECH_SKILLS):
                    skills.append(ExtractedSkill(
                        text=line,
                        source=source,
                        raw_match=line,
                        confidence=0.8
                    ))
                    self._update_stats(source)
        
        return skills
    
    def _regex_search(self, text: str, source: SkillSource) -> List[ExtractedSkill]:
        """Поиск по regex паттернам"""
        skills = []
        
        patterns = [
            (r'(?:опыт работы с|опыт с|работа с|знание|владение|умение)\s+([a-z0-9\s\+\#\-]+)', 0.85),
            (r'(?:должен (?:знать|уметь))\s+([a-z0-9\s\+\#\-]+)', 0.75),
            (r'(?:требуется|требуется знание)\s+([a-z0-9\s\+\#\-]+)', 0.80),
        ]
        
        for pattern, confidence in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                match = match.strip()
                
                if 3 < len(match) < 100:
                    skills.append(ExtractedSkill(
                        text=match,
                        source=source,
                        raw_match=match,
                        confidence=confidence
                    ))
                    self._update_stats(source)
        
        return skills
    
    def _update_stats(self, source: SkillSource) -> None:
        """Обновляет статистику"""
        key = source.value
        self.stats.by_source[key] = self.stats.by_source.get(key, 0) + 1
    
    def get_stats(self) -> ParsingStats:
        """Возвращает статистику парсинга"""
        return self.stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        self.stats = ParsingStats()