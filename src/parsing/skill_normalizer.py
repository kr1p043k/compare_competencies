"""
Нормализация навыков + fuzzy fallback.
Критично для качества skill_weights!
"""
import re
from typing import List, Dict, Optional, Set
import logging
from rapidfuzz import process, fuzz
from src.parsing.utils import load_it_skills 

logger = logging.getLogger(__name__)

class SkillNormalizer:
    # ================= СЛОВАРЬ СИНОНИМОВ =================
    SYNONYM_MAP = {
        # Языки
        "python": ["python3", "python 3", "py3", "py"],
        "javascript": ["js", "java script"],
        "typescript": ["ts", "type script"],
        "go": ["golang", "go lang"],
        "csharp": ["c#", "c sharp", ".net"],
        "cpp": ["c++", "c plus plus", "С++", "C++"],
        "rust": ["rustlang"],
        "swift": ["swiftui"],
        "kotlin": ["kotlin android"],
        "php": ["php8", "php 8"],
        "ruby": ["ruby on rails", "rails"],
        "r": ["r language"],
        "matlab": ["matlab"],
        "scala": ["scala"],
        "perl": ["perl"],

        # Frontend
        "react": ["react.js", "reactjs"],
        "react native": ["reactnative"],
        "vue": ["vue.js", "vuejs", "vue3", "vue 3"],
        "angular": ["angular.js", "angularjs"],
        "next": ["next.js", "nextjs", "next js"],
        "nuxt": ["nuxt.js", "nuxtjs"],
        "svelte": ["sveltekit"],
        "tailwind": ["tailwind css", "tailwindcss"],
        "bootstrap": ["bootstrap5"],
        "jquery": ["jquery"],
        "redux": ["redux"],
        "mobx": ["mobx"],
        "webpack": ["webpack"],
        "vite": ["vite"],
        "babel": ["babel"],
        "eslint": ["eslint"],
        "prettier": ["prettier"],

        # Backend / Frameworks
        "nodejs": ["node.js", "node js", "node"],
        "fastapi": ["fast api", "fast-api"],
        "django": ["django rest", "django rest framework", "drf"],
        "flask": ["flask restful"],
        "spring": ["spring boot", "springboot", "spring framework"],
        "express": ["express.js", "expressjs"],
        "nestjs": ["nest.js", "nest js"],
        "laravel": ["laravel"],
        "symfony": ["symfony"],

        # Базы данных
        "postgresql": ["postgres", "postgre", "psql", "pg"],
        "mysql": ["mariadb", "mysql"],
        "mongodb": ["mongo", "mongo db"],
        "redis": ["redis cache"],
        "elasticsearch": ["elastic", "elastic search", "es"],
        "sqlserver": ["mssql", "ms sql", "sql server"],
        "sqlite": ["sqlite"],
        "oracle": ["oracle db"],
        "cassandra": ["cassandra"],
        "dynamodb": ["dynamodb"],
        "couchdb": ["couchdb"],
        "neo4j": ["neo4j"],
        "clickhouse": ["clickhouse"],

        # DevOps / Cloud
        "docker": ["docker container", "docker-compose", "docker compose", "containerization"],
        "kubernetes": ["k8s", "kuber", "k8"],
        "terraform": ["terraform", "tf"],
        "ansible": ["ansible"],
        "jenkins": ["jenkins"],
        "git": ["git"],
        "github": ["github"],
        "gitlab": ["gitlab"],
        "bitbucket": ["bitbucket"],
        "prometheus": ["prometheus"],
        "grafana": ["grafana"],
        "nginx": ["nginx"],
        "apache": ["apache"],
        "kafka": ["apache kafka", "kafka streams"],
        "rabbitmq": ["rabbitmq"],
        "celery": ["celery"],
        "airflow": ["apache airflow"],
        "mlflow": ["mlflow"],

        # CI/CD
        "ci/cd": ["cicd", "ci cd", "continuous integration", "continuous delivery", "continuous deployment"],
        "github actions": ["github action", "actions"],
        "gitlab ci/cd": ["gitlab ci"],

        # ML / AI / LLM
        "ml": ["machine learning", "ml"],
        "dl": ["deep learning"],
        "mlops": ["ml ops", "mlo ps"],
        "llm": ["large language model", "large language models", "llms"],
        "rag": ["retrieval augmented generation"],
        "langchain": ["langchain"],
        "huggingface": ["hugging face"],
        "transformers": ["transformers"],
        "pytorch": ["pytorch"],
        "tensorflow": ["tensorflow"],
        "keras": ["keras"],
        "scikit-learn": ["scikit learn", "sklearn"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "matplotlib": ["matplotlib"],
        "seaborn": ["seaborn"],
        "plotly": ["plotly"],
        "xgboost": ["xgboost"],
        "lightgbm": ["lightgbm"],
        "catboost": ["catboost"],
        "spark": ["apache spark"],
        "hadoop": ["hadoop"],
        "dvc": ["dvc"],
        "fine-tuning": ["fine tuning", "finetuning"],
        "lora": ["qlora"],
        "prompt engineering": ["prompting", "prompt engineering"],
        "openai api": ["openai api"],

        # Cloud
        "aws": ["amazon web services", "amazon aws"],
        "azure": ["microsoft azure"],
        "gcp": ["google cloud platform", "google cloud"],
        "yandex cloud": ["yandex cloud"],
        "digitalocean": ["digital ocean"],
        "heroku": ["heroku"],

        # Тестирование
        "jest": ["jest"],
        "pytest": ["pytest"],
        "cypress": ["cypress"],
        "playwright": ["playwright"],
        "selenium": ["selenium"],
        "junit": ["junit"],
        "testng": ["testng"],

        # Frontend basics
        "html": ["html5", "html"],
        "css": ["css3", "css"],
        "sass": ["scss"],
        "less": ["less"],
        "rest": ["rest api", "restful api", "restful", "restapi"],
        "graphql": ["graph ql"],
        "apollo": ["apollo"],

        # Разное
        "figma": ["figma"],
        "storybook": ["storybook"],
        "npm": ["npm"],
        "yarn": ["yarn"],
        "webpack": ["webpack"],
        "vite": ["vite"],
    }

    _canonical_map: Optional[Dict[str, str]] = None

    @classmethod
    def _get_canonical_map(cls) -> Dict[str, str]:
        """Строит плоский маппинг напрямую из SYNONYM_MAP."""
        if cls._canonical_map is None:
            canon = {}
            for canonical, variants in cls.SYNONYM_MAP.items():
                for v in variants:
                    canon[v] = canonical
                canon[canonical] = canonical  # каноник тоже маппится сам в себя
            cls._canonical_map = canon
            logger.info(f"Построен канонический маппинг из {len(canon)} терминов")
        return cls._canonical_map

    # Версии и варианты (удаляются полностью)
    VERSION_PATTERNS = [
        r'\s*v?\d+(\.\d+)*',
        r'\s*\(.*?\)',
        r'\s*\[.*?\]',
    ]
    PREFIX_REMOVALS = [
        r'^опыт\s+(работы\s+)?(с\s+)?',
        r'^знание\s+',
        r'^владение\s+',
        r'^умение\s+(работать\s+)?(с\s+)?',
        r'^навык(и)?\s+(работы\s+)?(с\s+)?',
        r'^понимание\s+',
        r'^разработка\s+',
        r'^программирование\s+на\s+',
        r'^работа\s+с\s+',
        r'^участие\s+в\s+',
        r'^проведение\s+',
        r'^принципов\s+', 
        r'^организация\s+',
        r'^управление\s+',
        r'^построение\s+',
        r'^создание\s+',
        r'^внедрение\s+',
        r'^оценивать\s+',
        r'^(уверенный|senior|middle|junior)\s+',
        r'^(опыт|знание|умение|владение|навык)\s+(работы\s+)?(с\s+)?',
        r'^(разработчик|разработчика)\s+(уровня\s+)?(senior|middle|junior)?'
    ]
    SUFFIX_REMOVALS = [
        'язык', 'язык программирования',
        'фреймворк', 'библиотека', 'инструмент',
        'database', 'server', 'client',
        'framework', 'library', 'tool',
        r'\s+или\s+подобных\s+языках?',
        r'\s+и\s+(принципов|основ|т\.д\.|т\.п\.|др\.)',
        r'\s+или\s+(аналогичных|подобных)\s+(языков|технологий)?',
        r'\s+(хорошее|отличное|базовое)\s+(знание|понимание|умение)?$',
        r'\s+(и|или)\s+\w+$',
        r'\s+или\s+аналогичных\s+(языков|языках)',
        r'\s+и\s+т\.\s*д\.',
        r'\s+и\s+т\.\s*п\.',
        r'\s+и\s+др\.',
        r'\s+и\s+проч\.',
        r'\s+etc\.?',
        r'\s+(senior|middle|junior)$',
        r'\s+и\s+(другие|т\.д\.|т\.п\.|etc)$',
    ]

    FUZZY_THRESHOLD = 85
    MAX_FUZZY_CANDIDATES = 3

    _whitelist: Optional[Set[str]] = None

    @classmethod
    def _get_whitelist(cls) -> Set[str]:
        if cls._whitelist is None:
            cls._whitelist = load_it_skills()
            cls._whitelist.update([
                "python", "node.js", "react", "angular", "vue", "django",
                "flask", "fastapi", "sql", "postgresql", "mysql", "mongodb",
                "docker", "kubernetes", "git", "mlops", "cpp", "csharp",
                "go", "java", "html", "css", "javascript", "typescript", "c++"
            ])
            logger.info(f"Whitelist загружен и дополнен: {len(cls._whitelist)} навыков")
        return cls._whitelist

    @staticmethod
    def normalize(skill: str) -> str:
        if not skill:
            return ""
        original = skill.strip()
        text = original.lower()

        for pattern in SkillNormalizer.VERSION_PATTERNS:
            text = re.sub(pattern, '', text)
        for pattern in SkillNormalizer.PREFIX_REMOVALS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        for suffix in SkillNormalizer.SUFFIX_REMOVALS:
            text = re.sub(suffix, '', text, flags=re.IGNORECASE)

        text = re.sub(r'\s+', ' ', text).strip()

        text = SkillNormalizer._apply_synonym_map(text)

        text = re.sub(r'[^\w\s\+\#\-\.]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        whitelist = SkillNormalizer._get_whitelist()
        if text in whitelist:
            return text

        matches = process.extract(
            text, whitelist, scorer=fuzz.WRatio,
            limit=SkillNormalizer.MAX_FUZZY_CANDIDATES
        )
        if matches and matches[0][1] >= SkillNormalizer.FUZZY_THRESHOLD:
            best = matches[0][0]
            logger.debug(f"Fuzzy match: '{original}' → '{best}' (score={matches[0][1]})")
            return best

        logger.debug(f"No good fuzzy match for: '{original}' → '{text}'")
        return text

    @classmethod
    def _apply_synonym_map(cls, text: str) -> str:
        text_lower = text.lower().strip()
        canon_map = cls._get_canonical_map()
        if text_lower in canon_map:
            return canon_map[text_lower]
        return text_lower

    @staticmethod
    def normalize_batch(skills: List[str]) -> List[str]:
        return [SkillNormalizer.normalize(skill) for skill in skills if skill]

    @staticmethod
    def deduplicate(skills: List[str]) -> List[str]:
        seen = set()
        result = []
        for skill in skills:
            norm = SkillNormalizer.normalize(skill)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)
        return result