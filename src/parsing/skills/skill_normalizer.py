"""
Нормализация навыков + fuzzy fallback.
Критично для качества skill_weights!
"""

import re
from functools import cache

import structlog
from rapidfuzz import fuzz, process

from src import Result, Ok, Err
from src.errors import DomainError
from src.parsing.utils import load_it_skills

logger = structlog.get_logger(__name__)


class SkillNormalizer:
    # ================= СЛОВАРЬ СИНОНИМОВ =================
    SYNONYM_MAP = {
        # Языки
        "python": ["python3", "python 3", "py3", "py", "python programming", "python разработка"],
        "javascript": ["js", "java script"],
        "typescript": ["ts", "type script"],
        "go": ["golang", "go lang"],
        "csharp": ["c#", "c sharp"],
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
        "mlops": ["ml ops"],
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
        # Misc aliases from usage tracking
        "c++": ["cc++", "c/c++", "с++", "С++"],
        "ci/cd": ["cicd", "ci cd"],
        "msoffice": ["ms office", "microsoft office"],
        "neo4j": ["neoj"],
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
    }

    _canonical_map: dict[str, str] | None = None

    @classmethod
    def _get_canonical_map(cls) -> dict[str, str]:
        """Строит плоский маппинг из SYNONYM_MAP + таксономии."""
        if cls._canonical_map is None:
            canon = {}
            for canonical, variants in cls.SYNONYM_MAP.items():
                for v in variants:
                    canon[v] = canonical
                canon[canonical] = canonical
            # Load aliases from taxonomy
            try:
                from pathlib import Path
                tax_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reference" / "skill_taxonomy.json"
                if tax_path.exists():
                    import json
                    tax = json.loads(tax_path.read_text(encoding="utf-8"))
                    for cat in tax.get("categories", {}).values():
                        for alias, target in cat.get("aliases", {}).items():
                            canon[alias.lower()] = target
            except Exception as exc:
                logger.warning("taxonomy_aliases_load_failed", error=str(exc))
            cls._canonical_map = canon
            logger.info("canonical_map_built", terms=len(canon))
        return cls._canonical_map

    # Версии и варианты (удаляются полностью)
    VERSION_PATTERNS = [
        r"\s*v?\d+(\.\d+)*",
        r"\s*\(.*?\)",
        r"\s*\[.*?\]",
    ]
    PREFIX_REMOVALS = [
        r"^опыт\s+(работы\s+)?(с\s+)?",
        r"^знание\s+",
        r"^владение\s+",
        r"^умение\s+(работать\s+)?(с\s+)?",
        r"^навык(и)?\s+(работы\s+)?(с\s+)?",
        r"^понимание\s+",
        r"^разработка\s+",
        r"^программирование\s+на\s+",
        r"^работа\s+с\s+",
        r"^участие\s+в\s+",
        r"^проведение\s+",
        r"^принципов\s+",
        r"^организация\s+",
        r"^управление\s+",
        r"^построение\s+",
        r"^создание\s+",
        r"^внедрение\s+",
        r"^оценивать\s+",
        r"^(уверенный|senior|middle|junior)\s+",
        r"^(опыт|знание|умение|владение|навык)\s+(работы\s+)?(с\s+)?",
        r"^(разработчик|разработчика)\s+(уровня\s+)?(senior|middle|junior)?",
    ]
    SUFFIX_REMOVALS = [
        "язык",
        "язык программирования",
        "фреймворк",
        "библиотека",
        "инструмент",
        "database",
        "server",
        "client",
        "framework",
        "library",
        "tool",
        r"\s+или\s+подобных\s+языках?",
        r"\s+и\s+(принципов|основ|т\.д\.|т\.п\.|др\.)",
        r"\s+или\s+(аналогичных|подобных)\s+(языков|технологий)?",
        r"\s+(хорошее|отличное|базовое)\s+(знание|понимание|умение)?$",
        r"\s+(и|или)\s+\w+$",
        r"\s+или\s+аналогичных\s+(языков|языках)",
        r"\s+и\s+т\.\s*д\.",
        r"\s+и\s+т\.\s*п\.",
        r"\s+и\s+др\.",
        r"\s+и\s+проч\.",
        r"\s+etc\.?",
        r"\s+(senior|middle|junior)$",
        r"\s+и\s+(другие|т\.д\.|т\.п\.|etc)$",
        r"[-–—](?:обязательно|предпочтительно|приветствуется|преимущество)\b",
    ]

    FUZZY_THRESHOLD = 88
    MAX_FUZZY_CANDIDATES = 3

    _whitelist: set[str] | None = None

    @classmethod
    def _get_whitelist(cls) -> set[str]:
        if cls._whitelist is None:
            cls._whitelist = load_it_skills()
            cls._whitelist.update(
                [
                    "python",
                    "node.js",
                    "react",
                    "angular",
                    "vue",
                    "django",
                    "flask",
                    "fastapi",
                    "sql",
                    "postgresql",
                    "mysql",
                    "mongodb",
                    "docker",
                    "kubernetes",
                    "git",
                    "mlops",
                    "cpp",
                    "csharp",
                    "go",
                    "java",
                    "html",
                    "css",
                    "javascript",
                    "typescript",
                    "c++",
                    ".net",
                    "asp.net",
                    "node.js",
                ]
            )
            logger.info("whitelist_loaded_and_extended", count=len(cls._whitelist))
        return cls._whitelist

    JOB_TITLE_PATTERNS = [
        r"^(project|product|delivery|program|engineering|engineering) manager",
        r"^(tech|team|scrum|product|engineering|development) lead",
        r"^(senior|junior|middle|lead|principal|chief|head of).*(engineer|developer|manager|architect)$",
        r"engineer$",
        r"(developer|manager)$",
        r"специалист",
        r"руководитель",
        r"инженер$",
        r"^(team|project|product|delivery|program) ",
    ]

    @staticmethod
    @cache
    def normalize(skill: str) -> Result[str, DomainError]:
        try:
            if not skill:
                return Ok("")
            original = skill.strip()
            text = original.lower()

            # Early rejection of junk text
            if len(original) > 40:
                return Ok("")
            # Mixed Cyrillic+Latin in same word = evasion technique
            if re.search(r'[a-z][а-яё]|[а-яё][a-z]', original):
                return Ok("")
            # Compound tech stack: 4+ space-separated short alnum words
            words = text.split()
            if len(words) >= 4 and all(len(w) <= 12 for w in words) and all(w.isalnum() for w in words):
                return Ok("")

            for pattern in SkillNormalizer.VERSION_PATTERNS:
                text = re.sub(pattern, "", text)
            for pattern in SkillNormalizer.PREFIX_REMOVALS:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            for suffix in SkillNormalizer.SUFFIX_REMOVALS:
                text = re.sub(suffix, "", text, flags=re.IGNORECASE)

            text = re.sub(r"\s+", " ", text).strip()

            text = SkillNormalizer._apply_synonym_map(text)

            # Reject single-letter results (leftovers from "C" removal, "c 9:00" etc.)
            if len(text) <= 1:
                logger.debug("too_short_after_normalization", original=original, text=text)
                return Ok("")

            # Reject job titles
            for pat in SkillNormalizer.JOB_TITLE_PATTERNS:
                if re.search(pat, text):
                    logger.debug("job_title_rejected", original=original)
                    return Ok("")

            text = re.sub(r"[^\w\s\+\#\-\.]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            whitelist = SkillNormalizer._get_whitelist()
            if text in whitelist:
                return Ok(text)

            # Token pre-filter: only compare against whitelist entries sharing tokens
            input_tokens = set(text.split())
            candidates = [
                w for w in whitelist
                if input_tokens & set(w.split())
            ]

            if not candidates:
                logger.debug("no_fuzzy_candidates", original=original, normalized=text)
                return Ok(text)

            matches = process.extract(text, candidates, scorer=fuzz.WRatio,
                                      limit=SkillNormalizer.MAX_FUZZY_CANDIDATES)
            if matches and matches[0][1] >= SkillNormalizer.FUZZY_THRESHOLD:
                best = matches[0][0]
                # Reject fuzzy match if candidate is too short (< 3 chars) unless input is also short
                if len(best) < 3 and len(text) >= 3:
                    logger.debug("fuzzy_skipped_short_candidate", original=original, candidate=best, score=matches[0][1])
                    return Ok(text)
                logger.debug("fuzzy_match", original=original, matched=best, score=matches[0][1])
                return Ok(best)

            logger.debug("no_fuzzy_match", original=original, normalized=text)
            return Ok(text)
        except Exception as e:
            return Err(DomainError(message=str(e), detail=f"normalize({skill})"))

    @classmethod
    def _apply_synonym_map(cls, text: str) -> str:
        text_lower = text.lower().strip()
        canon_map = cls._get_canonical_map()
        if text_lower in canon_map:
            return canon_map[text_lower].lower()
        return text_lower

    @staticmethod
    def normalize_batch(skills: list[str]) -> Result[list[str], DomainError]:
        results = []
        for skill in skills:
            if skill:
                r = SkillNormalizer.normalize(skill)
                if r.is_ok():
                    results.append(r.unwrap())
        return Ok(results)

    @staticmethod
    def resolve(skill: str) -> str:
        """Fast alias resolution: lower + strip versions + synonym map.
        Unlike normalize(), does not check whitelist or run fuzzy match."""
        if not skill:
            return skill
        text = skill.lower().strip()
        if len(text) > 40:
            return ""
        for pattern in SkillNormalizer.VERSION_PATTERNS:
            text = re.sub(pattern, "", text)
        for pattern in SkillNormalizer.PREFIX_REMOVALS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        for suffix in SkillNormalizer.SUFFIX_REMOVALS:
            text = re.sub(suffix, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        text = SkillNormalizer._apply_synonym_map(text)
        text = re.sub(r"[^\w\s\+\#\-\.]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def deduplicate(skills: list[str]) -> Result[list[str], DomainError]:
        seen = set()
        result = []
        for skill in skills:
            r = SkillNormalizer.normalize(skill)
            norm = r.unwrap() if r.is_ok() else ""
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)
        return Ok(result)
