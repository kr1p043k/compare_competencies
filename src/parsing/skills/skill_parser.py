"""
Типизированный парсер навыков с полным логированием.
Отвечает ТОЛЬКО за извлечение навыков, не за валидацию!
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import structlog

from src import Err, Ok, Result
from src.errors import DomainError
from src.models.vacancy import Vacancy

logger = structlog.get_logger(__name__)

# Cyrillic → Latin homoglyph map for skill matching
_CYR_TO_LAT = str.maketrans({
    'а': 'a', 'А': 'a',
    'е': 'e', 'Е': 'e',
    'о': 'o', 'О': 'o',
    'р': 'p', 'Р': 'p',
    'с': 'c', 'С': 'c',
    'у': 'y', 'У': 'y',
    'х': 'x', 'Х': 'x',
    'к': 'k', 'К': 'k',
    'м': 'm', 'М': 'm',
    'н': 'n', 'Н': 'n',
    'в': 'b', 'В': 'b',
})


def _normalize_for_matching(text: str) -> str:
    """Lowercase + replace Cyrillic homoglyphs with Latin."""
    return text.lower().translate(_CYR_TO_LAT)


def _load_it_skills() -> set[str]:
    """Load skill list from it_skills.json, normalized for matching."""
    path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reference" / "it_skills.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        skills = {_normalize_for_matching(s.strip()) for s in raw if s.strip()}
        logger.info("it_skills_loaded", count=len(skills), path=str(path))
        return skills
    except Exception as exc:
        logger.warning("it_skills_load_failed", error=str(exc), path=str(path))
        return set()


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
    by_source: dict[str, int] = field(default_factory=dict)
    text_length_processed: int = 0

    def __repr__(self):
        sources_str = ", ".join(f"{k}={v}" for k, v in sorted(self.by_source.items()))
        return f"ParsingStats(total={self.total_extracted}, sources=[{sources_str}])"


class SkillParser:
    """
    Парсер для извлечения навыков из вакансий.
    Фокусируется на ИЗВЛЕЧЕНИИ, не на фильтрации.
    """

    # populated in __init__ from it_skills.json; kept as fallback
    TECH_SKILLS: set[str] = set()

    SKILL_MARKERS = [
        "ключевые навыки",
        "ключевые компетенции",
        "требования",
        "требования к кандидату",
        "необходимые навыки",
        "мы ждем",
        "ожидаем от вас",
        "что нужно знать",
        "профессиональные навыки",
        "опыт работы с",
        "знание",
        "владение",
        "умение",
        "должен знать",
        "должен уметь",
        "stack",
        "технологии",
        "инструменты",
    ]

    # Слова, которые указывают на отрицание
    NEGATION_WORDS = ["не требуется", "не нужно", "не обязательно", "не критично", "не важно"]

    def __init__(self):
        self.stats = ParsingStats()
        if not SkillParser.TECH_SKILLS:
            SkillParser.TECH_SKILLS = _load_it_skills()

    def parse_vacancy(self, vacancy: Vacancy) -> Result[list[ExtractedSkill], DomainError]:
        """Извлекает все навыки из вакансии"""
        try:
            skills = []

            match self._extract_from_key_skills(vacancy):
                case Ok(ks):
                    skills.extend(ks)
                case Err(e):
                    logger.debug("key_skills_extract_failed", error=str(e))

            text_sources = []
            if vacancy.snippet:
                if vacancy.snippet.requirement:
                    text_sources.append((vacancy.snippet.requirement, SkillSource.SNIPPET_REQUIREMENT, 2000))
                if vacancy.snippet.responsibility:
                    text_sources.append((vacancy.snippet.responsibility, SkillSource.SNIPPET_RESPONSIBILITY, 2000))
            if vacancy.description:
                text_sources.append((vacancy.description, SkillSource.DESCRIPTION, 10000))

            for text, source, max_len in text_sources:
                match self._extract_from_text(text, source=source, max_text_length=max_len):
                    case Ok(extracted):
                        skills.extend(extracted)
                    case Err(e):
                        logger.debug("text_extract_failed", source=source, error=str(e))

            self.stats.total_extracted += len(skills)

            return Ok(skills)
        except Exception as e:
            logger.exception("parse_vacancy_failed", vacancy_id=vacancy.id)
            return Err(DomainError(message=f"Failed to parse vacancy {vacancy.id}", detail=str(e)))

    def _extract_from_key_skills(self, vacancy: Vacancy) -> Result[list[ExtractedSkill], DomainError]:
        """Извлекает из официального поля key_skills"""
        try:
            skills = []
            for key_skill in vacancy.key_skills:
                skill = ExtractedSkill(
                    text=key_skill.name, source=SkillSource.KEY_SKILLS, raw_match=key_skill.name, confidence=1.0
                )
                skills.append(skill)
                self._update_stats(SkillSource.KEY_SKILLS)
            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_extract_from_key_skills"))

    def _extract_from_text(self, text: str, source: SkillSource, max_text_length: int = 1000) -> Result[list[ExtractedSkill], DomainError]:
        """Извлекает навыки из текста несколькими методами"""
        try:
            if not text:
                return Ok([])

            text = text[:max_text_length]
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            self.stats.text_length_processed += len(text)

            skills = []
            for method in (self._direct_search, self._marker_search, self._regex_search):
                match method(text, source):
                    case Ok(extracted):
                        skills.extend(extracted)
                    case Err(e):
                        logger.debug("skill_extract_method_failed", method=method.__name__, error=str(e))

            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_extract_from_text"))

    def _direct_search(self, text: str, source: SkillSource) -> Result[list[ExtractedSkill], DomainError]:
        """Прямой поиск по списку TECH_SKILLS с контролем контекста"""
        try:
            skills = []
            text_norm = _normalize_for_matching(text)

            extended_skills = [
                "machine learning", "deep learning", "data science",
                "full stack", "rest api", "sql database",
                "spring boot", "scikit-learn",
            ]

            all_skills = list(self.TECH_SKILLS) + extended_skills
            all_skills = sorted(set(all_skills), key=len, reverse=True)

            for tech in all_skills:
                pattern = rf"\b{re.escape(tech)}\b"
                for match in re.finditer(pattern, text_norm):
                    start = max(0, match.start() - 50)
                    context = text_norm[start : match.end() + 50]

                    has_negation = any(neg in context for neg in self.NEGATION_WORDS)
                    if has_negation:
                        continue

                    if len(tech) == 1 and re.search(r'[а-яё]', text[match.start():match.end()], re.IGNORECASE):
                        after = text_norm[match.end():match.end() + 3]
                        if after.startswith("++"):
                            pass
                        elif re.search(r'[а-яё]', text_norm[max(0, match.start()-5):match.start()], re.IGNORECASE):
                            continue

                    original = text[match.start():match.end()].strip()
                    skills.append(ExtractedSkill(text=original or tech, source=source, raw_match=tech, confidence=0.95))
                    self._update_stats(source)

            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_direct_search"))

    def _marker_search(self, text: str, source: SkillSource) -> Result[list[ExtractedSkill], DomainError]:
        """Поиск навыков после маркеров"""
        try:
            skills = []
            text_norm = _normalize_for_matching(text)

            for marker in self.SKILL_MARKERS:
                if marker not in text_norm:
                    continue
                parts = text_norm.split(marker)
                if len(parts) < 2:
                    continue
                after_marker = parts[1][:600]
                lines = re.split(r"[\n,•\-*;]", after_marker)
                for line in lines:
                    line = line.strip()
                    if len(line) <= 3 or len(line) >= 100:
                        continue
                    if any(tech in line for tech in self.TECH_SKILLS):
                        skills.append(ExtractedSkill(text=line, source=source, raw_match=line, confidence=0.8))
                        self._update_stats(source)

            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_marker_search"))

    def _regex_search(self, text: str, source: SkillSource) -> Result[list[ExtractedSkill], DomainError]:
        """Поиск по regex паттернам"""
        try:
            skills = []
            text_norm = _normalize_for_matching(text)

            patterns = [
                (r"(?:опыт работы с|опыт с|работа с|знание|владение|умение)\s+([a-z0-9\s\+\#\-]+)", 0.85),
                (r"(?:должен (?:знать|уметь))\s+([a-z0-9\s\+\#\-]+)", 0.75),
                (r"(?:требуется|требуется знание)\s+([a-z0-9\s\+\#\-]+)", 0.80),
            ]

            for pattern, confidence in patterns:
                matches = re.findall(pattern, text_norm)
                for match in matches:
                    match = match.strip()
                    if 3 < len(match) < 100:
                        skills.append(ExtractedSkill(text=match, source=source, raw_match=match, confidence=confidence))
                        self._update_stats(source)

            return Ok(skills)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="_regex_search"))

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
