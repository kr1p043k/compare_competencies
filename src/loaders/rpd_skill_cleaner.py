"""Clean and normalize skills extracted from RPD PDFs.

Filters out:
- Exam/seminar/lecture scheduling text
- Generic evaluation criteria
- Overly verbose descriptions (>20 tokens, <2 tokens)
- Non-skill prose (full sentences with verbs in first position)
"""

import re
from typing import Any

import structlog

from src import Err, Ok, Result
from src.errors import DomainError

logger = structlog.get_logger(__name__)

_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"подготовки,\s*специальности", re.IGNORECASE),
    re.compile(r"\bвопрос[ы]?\b", re.IGNORECASE),
    re.compile(r"\bсеминар\b", re.IGNORECASE),
    re.compile(r"^вопросы:", re.IGNORECASE),
    re.compile(r"тема занятия", re.IGNORECASE),
    re.compile(r"самостоятельная работа", re.IGNORECASE),
    re.compile(r"контактная работа", re.IGNORECASE),
    re.compile(r"оценочн(?:ого|ые|ых)\s+средств", re.IGNORECASE),
    re.compile(r"промежуточн(?:ая|ой)\s+аттестаци", re.IGNORECASE),
    re.compile(r"текущ(?:ий|его|их)\s+контрол", re.IGNORECASE),
    re.compile(r"рубежн(?:ый|ого)\s+контрол", re.IGNORECASE),
    re.compile(r"экзамен", re.IGNORECASE),
    re.compile(r"зачет", re.IGNORECASE),
    re.compile(r"рейтинг", re.IGNORECASE),
    re.compile(r"балл", re.IGNORECASE),
    re.compile(r"ликвидировать задолженност", re.IGNORECASE),
    re.compile(r"номер варианта", re.IGNORECASE),
    re.compile(r"типов(?:ой|ые|ых)\s+задач", re.IGNORECASE),
    re.compile(r"\bп/п\b"),
    re.compile(r"^\d+\s*[–\-]\s*\d+\s*$"),
    re.compile(r"^-\s*$"),
]

_VERBOSE_FILTERS: list[re.Pattern] = [
    re.compile(r"^(?:проводить|выполнять|осуществлять|разрабатывать|создавать|применять|использовать|организовывать|реализовывать|анализировать|проектировать|оценивать|исследовать|выбирать|интегрировать|демонстрировать|составлять|оформлять|настраивать|сопровождать|контролировать|выявлять|определять|рассчитывать|классифицировать|интерпретировать)\s", re.IGNORECASE),
    re.compile(r"^[а-яё]\s", re.IGNORECASE),
    re.compile(r"^-+\s"),
    re.compile(r"\bосновн(?:ые|ых|ые)\s+теорем", re.IGNORECASE),
    re.compile(r"^2\s+–\s+–\s+1\s+семин", re.IGNORECASE),
    re.compile(r"отчет предоставляется в", re.IGNORECASE),
    re.compile(r"не смогли набрать необходимое", re.IGNORECASE),
    re.compile(r"подготовка отчета", re.IGNORECASE),
]

_KNOWN_SKILL_WORDS: set[str] = {
    "язык", "среда", "платформа", "фреймворк", "библиотека", "инструмент",
    "технология", "метод", "алгоритм", "протокол", "архитектура", "структура",
    "модель", "система", "приложение", "сервис", "компонент", "интерфейс",
    "база", "сеть", "процесс", "стандарт", "норма", "правило", "принцип",
    "подход", "концепция", "парадигма", "паттерн", "практика", "нотация",
    "программа", "скрипт", "команда", "утилита", "апи", "api",
    "linux", "windows", "docker", "git", "sql", "python", "java", "c++",
    "javascript", "typescript", "react", "angular", "vue", "node.js",
    "postgresql", "mysql", "mongodb", "redis", "kubernetes", "tensorflow",
    "pytorch", "scikit-learn", "pandas", "numpy", "fastapi", "django",
    "spring", "hibernate", "maven", "gradle", "junit", "selenium",
}


class RPDSkillCleaner:
    def __init__(self):
        self._stats = {"total": 0, "noise_removed": 0, "kept": 0}

    def clean_skills(self, skills: list[str]) -> Result[list[str], DomainError]:
        try:
            self._stats = {"total": len(skills), "noise_removed": 0, "kept": 0}
            cleaned = []
            for s in skills:
                cleaned_s = self._clean_one(s)
                if cleaned_s is not None:
                    cleaned.append(cleaned_s)
            cleaned = list(dict.fromkeys(cleaned))
            self._stats["kept"] = len(cleaned)
            logger.info("rpd_skills_cleaned", **self._stats)
            return Ok(cleaned)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="RPDSkillCleaner.clean_skills"))

    def clean_discipline_data(self, data: dict) -> Result[dict, DomainError]:
        try:
            cleaned_count = 0
            for disc_name, disc_data in data.get("09.03.02", {}).get("disciplines", {}).items():
                for comp in disc_data.get("skills", {}):
                    raw = disc_data["skills"][comp]
                    match self.clean_skills(raw):
                        case Ok(cleaned):
                            disc_data["skills"][comp] = cleaned
                            cleaned_count += len(raw) - len(cleaned)
                        case _:
                            pass
                for comp in disc_data.get("ksa", {}):
                    for ksa_type in ("knowledge", "abilities", "skills"):
                        raw = disc_data["ksa"][comp].get(ksa_type, [])
                        match self.clean_skills(raw):
                            case Ok(cleaned):
                                disc_data["ksa"][comp][ksa_type] = cleaned
            logger.info("rpd_disciplines_cleaned", disciplines=len(data.get("09.03.02", {}).get("disciplines", {})), removed=cleaned_count)
            return Ok(data)
        except Exception as e:
            return Err(DomainError(message=str(e), detail="RPDSkillCleaner.clean_discipline_data"))

    def _clean_one(self, raw: str) -> str | None:
        text = raw.strip().rstrip(";.,:-").strip()
        if not text or len(text) < 5:
            self._stats["noise_removed"] += 1
            return None

        if len(text.split()) > 30:
            self._stats["noise_removed"] += 1
            return None

        for pat in _NOISE_PATTERNS:
            if pat.search(text):
                self._stats["noise_removed"] += 1
                return None

        for pat in _VERBOSE_FILTERS:
            if pat.match(text):
                self._stats["noise_removed"] += 1
                return None

        normalized = self._normalize(text)
        if normalized is None:
            self._stats["noise_removed"] += 1
            return None

        return normalized

    def _normalize(self, text: str) -> str | None:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" ,;.:-")

        prefixes_to_strip = [
            "− ", "– ", "— ", "- ", "• ", "· ", "‣ ", "⁃ ",
            "Знания: ", "Умения: ", "Навыки: ", "Знать: ", "Уметь: ", "Владеть: ",
            "Skills: ", "Knowledge: ", "Abilities: ",
            "Владеет ", "Умеет ", "Знает ",
            "− Владеет ", "− Умеет ", "− Знает ",
        ]
        for p in prefixes_to_strip:
            if text.startswith(p):
                text = text[len(p):]
                break

        text = text.strip().strip(";.,:").strip()
        if not text or len(text) < 5:
            return None

        tokens = text.split()
        if len(tokens) > 25:
            return None

        tech_tokens = sum(1 for t in tokens if t.lower() in _KNOWN_SKILL_WORDS or re.match(r'^[a-z][a-z0-9+#.\-]*$', t.lower()))
        has_tech_keyword = any(t.lower() in _KNOWN_SKILL_WORDS for t in tokens)

        if not has_tech_keyword and len(tokens) > 8:
            return None

        return text.lower()

    def stats(self) -> dict[str, int]:
        return dict(self._stats)


def clean_rpd_skills_file(input_path: str, output_path: str | None = None) -> Result[int, DomainError]:
    try:
        import json
        from pathlib import Path

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaner = RPDSkillCleaner()
        match cleaner.clean_discipline_data(data):
            case Ok(cleaned):
                out = output_path or input_path
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, ensure_ascii=False, indent=2)
                removed = cleaner.stats()["noise_removed"]
                logger.info("rpd_skills_file_cleaned", path=out, removed=removed)
                return Ok(removed)
            case Err(e):
                return Err(e)
    except Exception as e:
        return Err(DomainError(message=str(e), detail="clean_rpd_skills_file"))
