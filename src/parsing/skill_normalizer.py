"""
Нормализация навыков + fuzzy fallback.
Критично для качества skill_weights!
"""

import re
from typing import List, Dict, Optional, Set
import logging
from pathlib import Path

from rapidfuzz import process, fuzz

from src import config
from src.parsing.utils import load_it_skills  # уже есть в проекте

logger = logging.getLogger(__name__)

class SkillNormalizer:
    """
    Нормализует навыки для TF-IDF анализа.
    
    """
    
    # Маппинг синонимов
    SKILL_SYNONYMS = {
        # Языки
        "javascript": "js",
        "typescript": "ts",
        "python3": "python",
        "python 3": "python",
        "py3": "python",
        "py": "python",
        "golang": "go",
        "c++": "cpp",
        "c#": "csharp",
        "c sharp": "csharp",
        
        # Фреймворки
        "vue.js": "vue",
        "react.js": "react",
        "angular.js": "angular",
        "express.js": "express",
        "node.js": "nodejs",
        "fastapi": "fastapi",
        "django rest": "django",
        "django rest framework": "django",
        
        # БД
        "postgres": "postgresql",
        "psql": "postgresql",
        "mysql": "mysql",
        "mongo": "mongodb",
        "mongo db": "mongodb",
        
        # DevOps
        "kubernetes": "k8s",
        "k8": "k8s",
        "docker": "docker",
        "jenkins": "jenkins",
        
        # Data Science
        "machine learning": "ml",
        "deep learning": "dl",
        "data science": "data science",
        "big data": "big data",
        "nlp": "nlp",
        "computer vision": "cv",
        "cv": "cv",
        "scikit learn": "scikit-learn",
        
        # Облако
        "amazon web services": "aws",
        "microsoft azure": "azure",
        "google cloud": "gcp",
        "yandex cloud": "yandex cloud",
    }
    
    # Версии и варианты (удаляются полностью)
    VERSION_PATTERNS = [
        r'\s*v?\d+(\.\d+)*',  # v1, 1.0, 3.10
        r'\s*\(.*?\)',  # (описание)
        r'\s*\[.*?\]',  # [описание]
    ]
    
    # Слова, которые можно безопасно удалить из конца
    SUFFIX_REMOVALS = [
        'язык', 'язык программирования',
        'фреймворк', 'библиотека', 'инструмент',
        'database', 'server', 'client',
        'framework', 'library', 'tool',
    ]
    # === НОВОЕ: fuzzy-настройки ===
    FUZZY_THRESHOLD = 85          # % сходства (можно вынести в config)
    MAX_FUZZY_CANDIDATES = 3

    _whitelist: Optional[Set[str]] = None
    @classmethod
    def _get_whitelist(cls) -> Set[str]:
        if cls._whitelist is None:
            cls._whitelist = load_it_skills()
            logger.info(f"Загружен whitelist для fuzzy: {len(cls._whitelist)} навыков")
        return cls._whitelist

    @staticmethod
    def normalize(skill: str) -> str:
        """Полная нормализация с fuzzy fallback"""
        if not skill:
            return ""

        original = skill.strip()
        skill_lower = original.lower()

        # 1. Правило-based (твои старые правила)
        normalized = skill_lower

        # Синонимы (длинные первыми)
        for synonym in sorted(SkillNormalizer.SKILL_SYNONYMS.keys(), key=len, reverse=True):
            canonical = SkillNormalizer.SKILL_SYNONYMS[synonym]
            if re.search(rf'\b{re.escape(synonym)}\b', normalized):
                normalized = re.sub(rf'\b{re.escape(synonym)}\b', canonical, normalized)

        # Версии и скобки
        for pattern in SkillNormalizer.VERSION_PATTERNS:
            normalized = re.sub(pattern, '', normalized)

        # Суффиксы
        for suffix in SkillNormalizer.SUFFIX_REMOVALS:
            normalized = re.sub(rf'\s+{re.escape(suffix)}\s*$', '', normalized)

        # Чистка
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        normalized = re.sub(r'[^\w\s\+\#\-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # 2. Fuzzy fallback (ТОЛЬКО если не нашли точное совпадение в whitelist)
        whitelist = SkillNormalizer._get_whitelist()
        if normalized in whitelist:
            return normalized

        # Ищем лучшее fuzzy-совпадение
        matches = process.extract(
            normalized,
            whitelist,
            scorer=fuzz.WRatio,
            limit=SkillNormalizer.MAX_FUZZY_CANDIDATES
        )

        if matches and matches[0][1] >= SkillNormalizer.FUZZY_THRESHOLD:
            best_match = matches[0][0]
            logger.debug(f"Fuzzy match: '{original}' → '{best_match}' (score={matches[0][1]})")
            return best_match

        # Если ничего не подошло — возвращаем то, что есть (валидатор потом отрежет)
        logger.debug(f"No good fuzzy match for: '{original}' → '{normalized}'")
        return normalized

    @staticmethod
    def normalize_batch(skills: List[str]) -> List[str]:
        """Нормализует батч"""
        return [SkillNormalizer.normalize(s) for s in skills if s]

    @staticmethod
    def deduplicate(skills: List[str]) -> List[str]:
        """Дедупликация с сохранением порядка"""
        seen = set()
        result = []
        for skill in skills:
            norm = SkillNormalizer.normalize(skill)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)
        return result