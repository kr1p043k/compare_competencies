"""
Нормализация навыков перед TF-IDF анализом.
Критично для качества результатов!
"""

import re
from typing import List
import logging

logger = logging.getLogger(__name__)


class SkillNormalizer:
    """
    Нормализует навыки для TF-IDF анализа.
    
    Проблема: навыки из разных вакансий могут быть одним и тем же,
    но написаны по-разному:
    - "python" vs "Python" vs "Python 3" vs "python3.10"
    Всё это должно быть "python"!
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
    
    @staticmethod
    def normalize(skill: str) -> str:
        """
        Нормализует один навык
        
        Args:
            skill: Исходный навык
        
        Returns:
            Нормализованный навык
        """
        if not skill:
            return ""
        
        skill = skill.strip()
        skill_lower = skill.lower()
        
        # 1. Приводим к нижнему регистру
        normalized = skill_lower
        
        # 2. Заменяем синонимы (длинные первыми)
        for synonym in sorted(SkillNormalizer.SKILL_SYNONYMS.keys(), key=len, reverse=True):
            canonical = SkillNormalizer.SKILL_SYNONYMS[synonym]
            if re.search(rf'\b{re.escape(synonym)}\b', normalized):
                normalized = re.sub(rf'\b{re.escape(synonym)}\b', canonical, normalized)
        
        # 3. Удаляем версии
        for pattern in SkillNormalizer.VERSION_PATTERNS:
            normalized = re.sub(pattern, '', normalized)
        
        # 4. Удаляем суффиксы
        for suffix in SkillNormalizer.SUFFIX_REMOVALS:
            normalized = re.sub(rf'\s+{re.escape(suffix)}\s*$', '', normalized)
        
        # 5. Удаляем лишние пробелы
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # 6. Удаляем спецсимволы кроме + и #
        normalized = re.sub(r'[^\w\s\+\#\-]', '', normalized)
        
        # 7. Ещё раз чистим пробелы
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    @staticmethod
    def normalize_batch(skills: List[str]) -> List[str]:
        """Нормализует список навыков"""
        return [SkillNormalizer.normalize(s) for s in skills if s]
    
    @staticmethod
    def deduplicate(skills: List[str]) -> List[str]:
        """
        Удаляет дубликаты в нормализованных навыках.
        Сохраняет порядок первого появления.
        """
        seen = set()
        result = []
        
        for skill in skills:
            normalized = SkillNormalizer.normalize(skill)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        
        return result