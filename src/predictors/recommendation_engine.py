"""
Движок рекомендаций: анализирует gap между студентом и рынком,
даёт человекочитаемые советы по развитию навыков.
Улучшенная версия с LTR-ранжированием, персонализацией и LLM-объяснениями (YandexGPT).
Поддерживает кластерный контекст для персонализированных рекомендаций.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Optional, Any
import requests
import time
import pandas as pd
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.skill_filter import SkillFilter
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine
from src.models.student import StudentProfile
from src import config

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Движок рекомендаций с TF-IDF анализом, LTR-ранжированием и генерацией естественного языка.
    Поддерживает LLM (YandexGPT) для живых объяснений.
    """

    def __init__(self, use_ltr: bool = True, use_llm: bool = False):
        self.comparator = CompetencyComparator(use_embeddings=True, level="middle")
        self.gap_analyzer: Optional[GapAnalyzer] = None
        self.skill_filter = SkillFilter()
        self.is_fitted = False
        if use_llm is None:
            use_llm = bool(config.YC_API_KEY and config.YC_FOLDER_ID)
        self.use_llm = use_llm
        # LTR
        self.use_ltr = use_ltr
        self.ltr_engine: Optional[LTRRecommendationEngine] = None
        if use_ltr:
            self.ltr_engine = LTRRecommendationEngine()
            ltr_model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
            if ltr_model_path.exists():
                try:
                    self.ltr_engine.load_model(ltr_model_path)
                    logger.info("✓ LTR-модель загружена")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить LTR-модель: {e}")
                    self.ltr_engine = None

        # LLM (YandexGPT)
        self.use_llm = use_llm and bool(config.YC_API_KEY and config.YC_FOLDER_ID)
        if use_llm and not self.use_llm:
            logger.warning("use_llm=True, но YC_API_KEY или YC_FOLDER_ID не заданы. LLM отключен.")

        # Кластерный контекст
        self.cluster_skills: Optional[List[str]] = None
        self.cluster_weights: Optional[Dict[str, float]] = None

        self._load_templates()
        logger.info(f"✓ RecommendationEngine инициализирован (LTR={self.use_ltr}, LLM={self.use_llm})")

    def set_cluster_context(self, cluster_skills: List[str], cluster_weights: Dict[str, float]):
        """Устанавливает контекст ближайшего кластера для персонализации рекомендаций."""
        self.cluster_skills = cluster_skills
        self.cluster_weights = cluster_weights
        logger.info(f"Установлен кластерный контекст: {len(cluster_skills)} навыков")

    def clear_cluster_context(self):
        """Сбрасывает кластерный контекст."""
        self.cluster_skills = None
        self.cluster_weights = None

    def _load_templates(self):
        templates_path = config.DATA_DIR / "templates" / "recommendation_templates.json"
        if templates_path.exists():
            try:
                with open(templates_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.HARD_SKILL_TEMPLATES = data.get("hard_skills", {})
                self.SOFT_SKILL_TEMPLATES = data.get("soft_skills", {})
                self.HARD_LEARNING_PATHS = data.get("hard_paths", {})
                self.SOFT_LEARNING_PATHS = data.get("soft_paths", {})
                logger.info(f"Шаблоны загружены из {templates_path}")
                return
            except Exception as e:
                logger.warning(f"Ошибка загрузки шаблонов: {e}")

        self.HARD_SKILL_TEMPLATES = {
            "python": "Python — основной язык для бэкенда и data science.",
            "java": "Java — стандарт для enterprise-приложений.",
            "sql": "SQL — язык работы с базами данных.",
            "docker": "Docker — стандарт для контейнеризации.",
        }
        self.SOFT_SKILL_TEMPLATES = {
            "английский язык": "Английский язык B2+ открывает доступ к документации.",
            "аналитическое мышление": "Аналитическое мышление критично для решения сложных задач.",
        }
        self.HARD_LEARNING_PATHS = {
            "python": "1. Основы Python. 2. Практика: 10+ мини-проектов. 3. Углубление.",
            "sql": "1. Основы SELECT. 2. JOIN и подзапросы. 3. Оптимизация.",
        }
        self.SOFT_LEARNING_PATHS = {
            "английский язык": "Занимайтесь ежедневно 30 минут.",
        }

    def fit(self, vacancies_skills: List[List[str]], skill_weights: Optional[Dict[str, float]] = None):
        if not vacancies_skills:
            logger.warning("❌ Нет данных вакансий для обучения")
            return

        self.comparator.fit_market(vacancies_skills)
        if skill_weights is None:
            skill_weights = self.comparator.get_skill_weights()
        self.gap_analyzer = GapAnalyzer(skill_weights)
        self.is_fitted = True
        logger.info(f"✓ RecommendationEngine обучен на {len(vacancies_skills)} вакансиях")

    def analyze(self, student_skills: List[str]) -> Dict[str, Any]:
        if not self.is_fitted:
            logger.warning("❌ RecommendationEngine не обучен")
            return {}

        # Используем кластерные веса, если они заданы, иначе глобальные
        weights = self.cluster_weights if self.cluster_weights else self.gap_analyzer.skill_weights
        temp_gap_analyzer = GapAnalyzer(weights) if self.cluster_weights else self.gap_analyzer

        score, confidence = self.comparator.compare(student_skills)
        gaps = temp_gap_analyzer.analyze_gap(student_skills, top_n=30)
        coverage, coverage_details = temp_gap_analyzer.coverage(student_skills)
        top_market = temp_gap_analyzer.top_market_skills(20)

        return {
            "match_score": round(score, 4),
            "confidence": round(confidence, 4),
            "coverage": round(coverage, 2),
            "coverage_details": coverage_details,
            "gaps": gaps,
            "top_market_skills": top_market,
            "used_cluster_context": self.cluster_weights is not None
        }

    def generate_recommendations(
        self,
        student_skills: List[str],
        student_profile: Optional[StudentProfile] = None
    ) -> Dict[str, Any]:
        analysis = self.analyze(student_skills)
        if not analysis:
            return {}

        gaps = analysis.get("gaps", {})
        high_priority = gaps.get("high_priority", [])[:10]
        medium_priority = gaps.get("medium_priority", [])[:8]

        all_missing = [g["skill"] for g in high_priority + medium_priority]

        ltr_scores: Dict[str, float] = {}
        ltr_explanations: Dict[str, str] = {}
        if self.ltr_engine and self.ltr_engine.is_fitted and all_missing:
            try:
                ltr_recs = self.ltr_engine.predict_skill_impact(student_skills, all_missing)
                for skill, score, expl in ltr_recs:
                    ltr_scores[skill] = score / 100.0
                    ltr_explanations[skill] = expl
            except Exception as e:
                logger.debug(f"LTR prediction failed: {e}")

        # Собираем рекомендации без ранга
        temp_recs = []
        for gap in high_priority + medium_priority:
            skill = gap.get("skill", "")
            importance = gap.get("importance", 0)
            priority = "HIGH" if gap in high_priority else "MEDIUM"

            combined_importance = importance
            if skill in ltr_scores:
                combined_importance = max(importance, ltr_scores[skill])

            rec = self._generate_skill_recommendation(
                skill=skill,
                importance=combined_importance,
                priority=priority,
                rank=0,  # временный ранг
                student_profile=student_profile,
                ltr_explanation=ltr_explanations.get(skill),
                student_skills=student_skills,
                coverage=analysis['coverage']
            )
            if rec:
                temp_recs.append(rec)

        # Сортируем по убыванию важности и присваиваем правильные ранги
        temp_recs.sort(key=lambda x: x['importance_score'], reverse=True)
        detailed_recs = []
        for idx, rec in enumerate(temp_recs, 1):
            rec['rank'] = idx
            detailed_recs.append(rec)

        return {
            "summary": {
                "match_score": analysis["match_score"],
                "confidence": analysis["confidence"],
                "coverage": analysis["coverage"],
                "covered_skills": analysis.get("coverage_details", {}).get("covered_skills_count", 0),
                "total_market_skills": analysis.get("coverage_details", {}).get("total_market_skills", 0),
                "used_cluster_context": analysis["used_cluster_context"]
            },
            "recommendations": detailed_recs,
            "top_market_skills": analysis["top_market_skills"]
        }

    def _generate_skill_recommendation(
        self,
        skill: str,
        importance: float,
        priority: str,
        rank: int,
        student_profile: Optional[StudentProfile] = None,
        ltr_explanation: Optional[str] = None,
        student_skills: Optional[List[str]] = None,
        coverage: float = 0.0,
        shap_values: Optional[np.ndarray] = None,
        X: Optional[pd.DataFrame] = None,
        idx: int = 0
    ) -> Dict[str, Any]:
        skill_lower = skill.lower()
        is_soft = not self._is_hard_skill(skill_lower)

        return {
            "rank": rank,
            "skill": skill,
            "importance_score": round(importance, 4),
            "priority": priority,
            "is_soft_skill": is_soft,
            "suggestion": self._get_suggestion(skill, is_soft),
            "why_important": self._why_important(
                skill, importance, priority, ltr_explanation,
                student_skills, coverage, shap_values, X, idx
            ),
            "how_to_learn": self._get_learning_path(skill, is_soft, student_profile),
            "expected_timeframe": self._get_timeframe(skill),
            "expected_outcome": self._get_expected_outcome(skill, student_profile),
            "market_frequency_percent": round(importance * 100, 1),
        }

    def _is_hard_skill(self, skill_lower: str) -> bool:
        cats = self.skill_filter.get_skill_categories([skill_lower])
        hard_cats = {"programming_languages", "frameworks", "databases", "devops",
                     "cloud", "data_science", "frontend", "testing", "tools"}
        if any(cat in hard_cats for cat in cats.keys()):
            return True

        hard_keywords = [
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "kotlin", "swift", "php", "ruby", "scala",
            "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra", "oracle", "mssql",
            "docker", "kubernetes", "k8s", "jenkins", "git", "gitlab", "github", "bitbucket", "terraform", "ansible", "prometheus", "grafana",
            "aws", "azure", "gcp", "yandex cloud",
            "machine learning", "deep learning", "nlp", "computer vision", "data science", "mlops", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
            "react", "vue", "angular", "django", "flask", "fastapi", "spring", "express", "node.js", "next", "nuxt",
            "rest api", "restful", "graphql", "api",
            "html", "css", "sass", "scss", "webpack", "vite", "redux", "mobx",
            "jest", "pytest", "cypress", "playwright", "selenium",
            "figma", "storybook", "eslint", "prettier", "babel", "npm", "yarn",
            "kafka streams", "greenplum", "powershell", "nginx", "flask", "kotlin", "terraform",
            "linux", "unix", "bash", "shell", "powershell", "cmd", "terminal",
            "spark", "hadoop", "airflow", "kafka", "rabbitmq", "celery"
        ]
        for kw in hard_keywords:
            if kw in skill_lower:
                return True
        return False

    def _get_suggestion(self, skill: str, is_soft: bool) -> str:
        skill_lower = skill.lower()
        if is_soft:
            return self.SOFT_SKILL_TEMPLATES.get(
                skill_lower,
                f"Развитие soft skill '{skill}' улучшит вашу эффективность в команде."
            )
        return self.HARD_SKILL_TEMPLATES.get(
            skill_lower,
            f"Навык '{skill}' высоко востребован на рынке и повысит вашу конкурентоспособность."
        )

    def _llm_explain_with_retry(self, skill: str, importance: float, priority: str,
                                student_skills: List[str], coverage: float,
                                max_retries: int = 2, delay: float = 2.0) -> Optional[str]:
        if not self.use_llm:
            return None

        prompt = f"""Ты — карьерный консультант в IT. Студент владеет навыками: {', '.join(student_skills[:10])} (показано до 10). 
Покрытие рынка составляет {coverage:.1f}%. 
Недостающий навык: {skill}. Важность навыка: {importance:.3f} (приоритет {priority}).
Объясни кратко (2-3 предложения), почему этот навык важен и как он сочетается с уже имеющимися у студента. 
Дай один конкретный совет по изучению."""

        headers = {
            "Authorization": f"Api-Key {config.YC_API_KEY}",
            "x-folder-id": config.YC_FOLDER_ID
        }
        payload = {
            "modelUri": f"gpt://{config.YC_FOLDER_ID}/{config.YANDEXGPT_MODEL}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.7,
                "maxTokens": "300"
            },
            "messages": [{"role": "user", "text": prompt}]
        }

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["result"]["alternatives"][0]["message"]["text"].strip()
                elif resp.status_code == 429:
                    wait_time = delay * (attempt + 1)
                    logger.warning(f"YandexGPT 429 (attempt {attempt+1}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"YandexGPT error {resp.status_code}: {resp.text[:200]}")
                    return None
            except Exception as e:
                logger.warning(f"YandexGPT exception (attempt {attempt+1}): {e}")
                if attempt < max_retries:
                    time.sleep(delay)
                else:
                    return None
        return None

    def _llm_explain(self, skill: str, importance: float, priority: str,
                     student_skills: List[str], coverage: float) -> Optional[str]:
        time.sleep(1.0)
        return self._llm_explain_with_retry(skill, importance, priority, student_skills, coverage)
    
    def _shap_explain(self, skill: str, shap_values: Optional[np.ndarray], 
                      idx: int, X: pd.DataFrame) -> Optional[str]:
        if shap_values is None or idx >= len(shap_values):
            return None
        top_idx = np.argmax(np.abs(shap_values[idx]))
        # feature_names должен быть у LTR engine, но здесь мы не имеем доступа, поэтому пропускаем
        return None

    def _why_important(self, skill: str, importance: float, priority: str,
                       ltr_explanation: Optional[str] = None,
                       student_skills: Optional[List[str]] = None,
                       coverage: float = 0.0,
                       shap_values: Optional[np.ndarray] = None,
                       X: Optional[pd.DataFrame] = None,
                       idx: int = 0) -> str:
        # 1. LLM
        if self.use_llm:
            llm_expl = self._llm_explain(skill, importance, priority, student_skills or [], coverage)
            if llm_expl:
                return f"🤖 {llm_expl}"

        # 2. SHAP (через LTR explanation, который уже содержит SHAP)
        if ltr_explanation:
            return f"🎯 Модель: {ltr_explanation}"

        # 3. Шаблоны
        if priority == "HIGH":
            return f"🔴 ВЫСОКИЙ приоритет: '{skill}' — один из самых востребованных навыков в вашей целевой роли."
        elif priority == "MEDIUM":
            return f"🟡 СРЕДНИЙ приоритет: '{skill}' значительно повысит вашу конкурентоспособность."
        else:
            return f"🟢 НИЗКИЙ приоритет: '{skill}' полезен для расширения кругозора и специализированных задач."

    def _get_learning_path(self, skill: str, is_soft: bool, student_profile: Optional[StudentProfile] = None) -> str:
        skill_lower = skill.lower()
        level = student_profile.target_level if student_profile else "middle"

        if is_soft:
            base = self.SOFT_LEARNING_PATHS.get(skill_lower, "Практикуйте навык постоянно.")
        else:
            base = self.HARD_LEARNING_PATHS.get(skill_lower, f"Изучите документацию '{skill}' и выполните проекты.")

        if level == "junior":
            base = "Сфокусируйтесь на основах: " + base
        elif level == "senior":
            base = "Углублённое изучение: " + base + " + архитектурные паттерны."
        return base

    def _get_timeframe(self, skill: str) -> str:
        skill_lower = skill.lower()
        easy = {"git", "html", "css", "sass", "english", "английский язык"}
        medium = {"javascript", "python", "sql", "redis", "docker", "react"}
        hard = {"java", "kubernetes", "aws", "machine learning", "tensorflow"}

        if any(k in skill_lower for k in easy):
            return "1-2 недели"
        elif any(k in skill_lower for k in medium):
            return "1-2 месяца"
        elif any(k in skill_lower for k in hard):
            return "2-6 месяцев"
        return "1-3 месяца"

    def _get_expected_outcome(self, skill: str, student_profile: Optional[StudentProfile]) -> str:
        role = student_profile.target_role if student_profile and hasattr(student_profile, 'target_role') else "вашей целевой роли"
        return f"Освоение '{skill}' позволит вам уверенно работать в роли '{role}'."