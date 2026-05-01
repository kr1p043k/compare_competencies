# src/predictors/recommendation_engine.py
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Any
import requests
import time
import pandas as pd

from src.parsing.skill_normalizer import SkillNormalizer
from src.analyzers.profile_evaluator import ProfileEvaluator
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

    def __init__(self, use_ltr: bool = True, use_llm: bool = False, profile_evaluator=None):
        self.comparator = CompetencyComparator(use_embeddings=True, level="middle")
        self.gap_analyzer: Optional[GapAnalyzer] = None
        self.skill_filter = SkillFilter()
        self.is_fitted = False
        self.profile_evaluator = profile_evaluator
        
        self.use_llm = use_llm and bool(config.YC_API_KEY and config.YC_FOLDER_ID)
        self.use_ltr = use_ltr
        self.ltr_engine: Optional[LTRRecommendationEngine] = None

        if use_ltr:
            self.ltr_engine = LTRRecommendationEngine()
            model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
            if model_path.exists():
                try:
                    self.ltr_engine.load_model(model_path)
                    logger.info("✓ LTR-модель успешно загружена")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить LTR: {e}")
                    self.ltr_engine = None

        self.cluster_weights = None
        self._load_templates()
        logger.info(f"✓ RecommendationEngine инициализирован (LTR={self.use_ltr}, LLM={self.use_llm})")
        
    def set_cluster_context(self, weights: Dict[str, float]) -> None:
        self.cluster_weights = weights
        if weights:
            logger.info(f"Установлен кластерный контекст: {len(weights)} навыков, сумма весов {sum(weights.values()):.2f}")
        else:
            logger.info("Кластерный контекст пуст (используются глобальные веса)")
        
    def clear_cluster_context(self):
        self.cluster_weights = None
        logger.info("Кластерный контекст сброшен")

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
    
    def fit(self, vacancies_skills: List[List[str]], skill_weights: Dict[str, float]) -> None:
        if not vacancies_skills:
            logger.warning("❌ Нет данных вакансий для обучения")
            return
        if not skill_weights:
            raise ValueError("skill_weights обязательны для fit")

        self.comparator.fit_market(vacancies_skills)
        self.gap_analyzer = GapAnalyzer(skill_weights)
        self.comparator.set_skill_weights(skill_weights)
        self.is_fitted = True

        logger.info(f"✓ RecommendationEngine обучен на {len(vacancies_skills)} вакансиях, веса для {len(skill_weights)} навыков")
        
    def analyze(self, student_skills: List[str]) -> Dict[str, Any]:
        if not self.is_fitted:
            return {}

        # Для score и confidence используем контекстные веса (кластер или глобальные)
        weights_for_score = self.cluster_weights if self.cluster_weights else self.gap_analyzer.skill_weights
        if not weights_for_score:
            logger.warning("analyze: веса пусты!")
            return {}

        self.comparator.set_skill_weights(weights_for_score)
        score, confidence = self.comparator.compare(student_skills)

        # Гэпы и топ-навыки ВСЕГДА считаем по глобальному рынку
        gaps = self.gap_analyzer.analyze_gap(student_skills, top_n=30)
        coverage, coverage_details = self.gap_analyzer.coverage(student_skills)
        top_market = self.gap_analyzer.top_market_skills(20)

        if 'covered_skills_count' not in coverage_details:
            coverage_details['covered_skills_count'] = coverage_details.get('covered_skills_count', 0)
        if 'total_market_skills' not in coverage_details:
            coverage_details['total_market_skills'] = coverage_details.get('total_market_skills', len(self.gap_analyzer.skill_weights))

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
        student: StudentProfile,
        user_type: str = 'student'
    ) -> Dict[str, Any]:
        """
        Генерирует рекомендации, используя новую модель ProfileEvaluator.
        """
        if not hasattr(self, 'profile_evaluator'):
            raise RuntimeError("RecommendationEngine должен быть инициализирован с profile_evaluator")

        # Получаем полную оценку от ProfileEvaluator
        eval_result = self.profile_evaluator.evaluate_profile(student, user_type=user_type)

        # Формируем рекомендации на основе top_recommendations
        recommendations = []
        for skill, score in eval_result.get('top_recommendations', []):
            rec = {
                "rank": 0,  # будет перезаписано при сортировке
                "skill": skill,
                "importance_score": score,
                "priority": "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW",
                "why_important": self._generate_explanation(skill, score, eval_result),
                "how_to_learn": self._get_learning_path(skill, False, student),
                "expected_timeframe": self._get_timeframe(skill),
                "expected_outcome": f"Освоение '{skill}' повысит вашу конкурентоспособность на рынке.",
                "is_soft_skill": not self._is_hard_skill(skill),
                "market_frequency_percent": score * 100
            }
            recommendations.append(rec)

        # Сортируем по важности
        recommendations.sort(key=lambda x: x['importance_score'], reverse=True)
        for idx, rec in enumerate(recommendations, 1):
            rec['rank'] = idx

        return {
            "summary": {
                "match_score": eval_result['market_coverage_score'],
                "confidence": eval_result['readiness_score'],
                "market_coverage_score": eval_result['market_coverage_score'],
                "skill_coverage": eval_result['skill_coverage'],
                "domain_coverage_score": eval_result['domain_coverage_score'],
                "readiness_score": eval_result['readiness_score'],
                "avg_gap": eval_result.get('avg_gap', 0),
                "coverage": eval_result['market_coverage_score'],
                "coverage_details": {
                    "covered_skills_count": len(
                        set(s.lower() for s in student.skills) &
                        set(eval_result.get('skill_metrics', {}).keys())
                    ),
                    "total_market_skills": len(eval_result.get('skill_metrics', {}))
                },
                "market_skill_coverage": eval_result.get('market_skill_coverage', 0.0)   # ← новая строка
            },
            "recommendations": recommendations[:15],
            "domain_coverage": eval_result.get('domain_coverage', {}),
            "gaps": eval_result.get('gaps', {})
        }
        
    def _generate_explanation(self, skill: str, score: float, eval_result: Dict) -> str:
        metric = eval_result.get('skill_metrics', {}).get(skill, {})
        cluster_rel = metric.get('cluster_relevance', 0)
        if cluster_rel > 0.7:
            return f"🎯 Сильно связан с вашим целевым профилем и востребован в ведущих компаниях."
        elif score > 0.7:
            return f"🔴 Один из самых востребованных навыков на рынке."
        elif score > 0.4:
            return f"🟡 Значительно повысит вашу конкурентоспособность."
        return f"🟢 Полезен для расширения кругозора."

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
        idx: int = 0,
        feature_names: Optional[List[str]] = None
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
                student_skills, coverage, shap_values, X, idx, feature_names
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
    def _get_priority(self, gap: float) -> str:
        if gap > 0.55:
            return "HIGH"
        elif gap > 0.30:
            return "MEDIUM"
        return "LOW"

    def _llm_explain(self, skill: str, importance: float, priority: str,
                     student_skills: List[str], coverage: float) -> Optional[str]:
        time.sleep(1.0)
        return self._llm_explain_with_retry(skill, importance, priority, student_skills, coverage)

    def _shap_explain(self, skill: str, shap_values: Optional[np.ndarray], 
                      idx: int, X: pd.DataFrame, feature_names: List[str]) -> Optional[str]:
        if shap_values is None or idx >= len(shap_values):
            return None
        top_idx = np.argmax(np.abs(shap_values[idx]))
        feat_name = feature_names[top_idx]
        feat_val = X.iloc[idx][feat_name]
        if feat_name == "cosine_sim":
            return f"сильно связан с вашим текущим профилем (сходство {feat_val:.2f})"
        elif feat_name == "level_encoded":
            level_str = {1: "junior", 2: "middle", 3: "senior"}.get(int(feat_val), "middle")
            return f"востребован на уровне {level_str}"
        elif feat_name == "category_encoded":
            return f"относится к востребованной категории навыков"
        return None

    def _why_important(self, skill: str, importance: float, priority: str,
                       ltr_explanation: Optional[str] = None,
                       student_skills: Optional[List[str]] = None,
                       coverage: float = 0.0,
                       shap_values: Optional[np.ndarray] = None,
                       X: Optional[pd.DataFrame] = None,
                       idx: int = 0,
                       feature_names: Optional[List[str]] = None) -> str:
        if self.use_llm:
            llm_expl = self._llm_explain(skill, importance, priority, student_skills or [], coverage)
            if llm_expl:
                return f"🤖 {llm_expl}"

        if shap_values is not None and X is not None and feature_names is not None:
            shap_expl = self._shap_explain(skill, shap_values, idx, X, feature_names)
            if shap_expl:
                base = f"🎯 Навык '{skill}' {shap_expl}."
                if priority == "HIGH":
                    base += " Это один из самых важных навыков для вашего уровня."
                elif priority == "MEDIUM":
                    base += " Его освоение повысит вашу конкурентоспособность."
                return base

        if ltr_explanation:
            return f"🎯 Модель: {ltr_explanation}"
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