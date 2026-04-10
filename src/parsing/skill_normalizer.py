"""
Нормализация навыков + fuzzy fallback.
Критично для качества skill_weights!
"""
#skill_normalizer.py
import re
from typing import List, Dict, Optional, Set
import logging
from pathlib import Path
from rapidfuzz import process, fuzz
from src import config
from src.parsing.utils import load_it_skills  # уже есть в проекте
from collections import defaultdict
logger = logging.getLogger(__name__)

class SkillNormalizer:
    # Теперь это список пар (синоним, каноническое_имя)
    SKILL_SYNONYMS = [
        # Языки
        ("javascript", "node.js"),
        ("typescript", "ts"),
        ("python3", "python"),
        ("python 3", "python"),
        ("py3", "python"),
        ("py", "python"),
        ("golang", "go"),
        ("c sharp", "csharp"),
        ("c#", "csharp"),
        ("c++", "cpp"),
        ("С++", "cpp"),
        ("C++", "cpp"),
        ("cpp", "c++"),          # циклично, лучше выбрать одно каноническое (рекомендую всегда "cpp")
        
        # Фреймворки
        ("vue.js", "vue"),
        ("react.js", "react"),
        ("angular.js", "angular"),
        ("express.js", "express"),
        ("node.js", "nodejs"),
        ("node js", "node.js"),
        ("nodejs", "node.js"),
        ("fastapi", "fastapi"),
        ("django rest", "django"),
        ("django rest framework", "django"),
        ("react native", "react"),
        
        # БД
        ("postgres", "postgresql"),
        ("psql", "postgresql"),
        ("mysql", "mysql"),
        ("mongo", "mongodb"),
        ("mongo db", "mongodb"),
        
        # DevOps
        ("kubernetes", "k8s"),
        ("k8", "k8s"),
        ("docker", "docker"),
        ("jenkins", "jenkins"),
        
        # Data Science
        ("machine learning", "mlops"),
        ("ml", "mlops"),
        ("mlops", "machine learning"),   # циклично, выберите один канон (лучше "mlops")
        ("deep learning", "dl"),
        ("data science", "data science"),
        ("big data", "big data"),
        ("nlp", "nlp"),
        ("computer vision", "cv"),
        ("cv", "cv"),
        ("scikit learn", "scikit-learn"),
        
        # Облако
        ("amazon web services", "aws"),
        ("microsoft azure", "azure"),
        ("google cloud", "gcp"),
        ("yandex cloud", "yandex cloud"),
    ]
    _canonical_map: Optional[Dict[str, str]] = None
    @classmethod
    def _build_canonical_map(cls) -> Dict[str, str]:
        """Разрешает цепочки синонимов, строя стабильный маппинг на единого представителя."""
        # 1. Строим граф: ключ → множество значений, куда он ведёт
        graph = defaultdict(set)
        all_nodes = set()
        for a, b in cls.SKILL_SYNONYMS:
            graph[a].add(b)
            all_nodes.add(a)
            all_nodes.add(b)

        # 2. Обход в глубину для нахождения компонент связности
        visited = set()
        components = []

        def dfs(node, comp):
            stack = [node]
            while stack:
                n = stack.pop()
                if n not in visited:
                    visited.add(n)
                    comp.append(n)
                    for neighbor in graph[n]:
                        if neighbor not in visited:
                            stack.append(neighbor)
                    # Также учитываем обратные связи (для циклов)
                    for k, vset in graph.items():
                        if n in vset and k not in visited:
                            stack.append(k)

        for node in all_nodes:
            if node not in visited:
                comp = []
                dfs(node, comp)
                components.append(comp)

        # 3. Для каждой компоненты выбираем канонического представителя
        #    (например, лексикографически наименьший или первый из whitelist)
        whitelist = cls._get_whitelist()
        canon_map = {}
        for comp in components:
            # Приоритет: 1) элемент из whitelist (если есть) 2) самый короткий 3) первый по алфавиту
            comp_in_whitelist = [c for c in comp if c in whitelist]
            if comp_in_whitelist:
                representative = min(comp_in_whitelist, key=lambda x: (len(x), x))
            else:
                representative = min(comp, key=lambda x: (len(x), x))
            for node in comp:
                canon_map[node] = representative

        logger.info(f"Построен канонический маппинг из {len(canon_map)} синонимов")
        return canon_map

    @classmethod
    def _get_canonical_map(cls) -> Dict[str, str]:
        if cls._canonical_map is None:
            cls._canonical_map = cls._build_canonical_map()
        return cls._canonical_map
    
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
    FUZZY_THRESHOLD = 85         # % сходства (можно вынести в config)
    MAX_FUZZY_CANDIDATES = 3

    _whitelist: Optional[Set[str]] = None
    @classmethod
    def _get_whitelist(cls) -> Set[str]:
        """Загружает и дополняет белый список."""
        if cls._whitelist is None:
            cls._whitelist = load_it_skills()
            # Гарантируем наличие ключевых канонических навыков
            cls._whitelist.update([
                "python", "node.js", "react", "angular", "vue", "django",
                "flask", "fastapi", "sql", "postgresql", "mysql", "mongodb",
                "docker", "kubernetes", "git", "mlops", "cpp", "csharp",
                "go", "java", "html", "css", "javascript", "typescript"
            ])
            logger.info(f"Whitelist загружен и дополнен: {len(cls._whitelist)} навыков")
        return cls._whitelist

    @staticmethod
    def normalize(skill: str) -> str:
        if not skill:
            return ""
        original = skill.strip()
        skill_lower = original.lower()
        normalized = skill_lower

        # Шаг 1: замена синонимов (используем канонический маппинг)
        canon_map = SkillNormalizer._get_canonical_map()
        # Разбиваем на слова, но также учитываем многословные синонимы
        # Чтобы не усложнять, применим подход: для каждого ключа в canon_map,
        # если он встречается как целое слово или фраза, заменяем.
        # Сортируем ключи по убыванию длины, чтобы длинные фразы имели приоритет.
        sorted_keys = sorted(canon_map.keys(), key=len, reverse=True)
        for synonym in sorted_keys:
            if synonym == canon_map[synonym]:
                continue  # пропускаем канонические сами на себя
            # Для многословных синонимов используем границы без \b
            if ' ' in synonym or '-' in synonym:
                pattern = r'(?<!\w)' + re.escape(synonym) + r'(?!\w)'
            else:
                pattern = r'\b' + re.escape(synonym) + r'\b'
            # Заменяем все вхождения (не жадничаем, т.к. карта уже плоская)
            normalized = re.sub(pattern, canon_map[synonym], normalized)

        # Шаг 2: версии и скобки (без изменений)
        for pattern in SkillNormalizer.VERSION_PATTERNS:
            normalized = re.sub(pattern, '', normalized)

        # Шаг 3: суффиксы (без изменений)
        for suffix in SkillNormalizer.SUFFIX_REMOVALS:
            normalized = re.sub(rf'\s+{re.escape(suffix)}\s*$', '', normalized)

        # Шаг 4: финальная чистка
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        normalized = re.sub(r'[^\w\s\+\#\-\.]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        normalized = re.sub(r'\.$', '', normalized)

        # Шаг 5: fuzzy fallback (без изменений)
        whitelist = SkillNormalizer._get_whitelist()
        if normalized in whitelist:
            return normalized

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

        logger.debug(f"No good fuzzy match for: '{original}' → '{normalized}'")
        return normalized
    
    @staticmethod
    def normalize_batch(skills: List[str]) -> List[str]:
        """Только нормализация, без глобальной дедупликации (для частот)"""
        return [SkillNormalizer.normalize(skill) for skill in skills if skill]

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