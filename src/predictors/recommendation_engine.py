# src/predictors/recommendation_engine.py
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import structlog

from src import config
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.skill_filter import SkillFilter
from src.analyzers.skill_taxonomy import SkillTaxonomy
from src.models.student import StudentProfile
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine

logger = structlog.get_logger(__name__)


class RecommendationEngine:
    """
    Движок рекомендаций с TF-IDF анализом, LTR-ранжированием и генерацией естественного языка.
    Поддерживает LLM (YandexGPT) для живых объяснений.
    """

    def __init__(self, use_ltr: bool = True, use_llm: bool = False, profile_evaluator=None):
        self.comparator = CompetencyComparator(use_embeddings=True, level="middle")
        self.gap_analyzer: GapAnalyzer | None = None
        self.skill_filter = SkillFilter()
        self.is_fitted = False
        self.profile_evaluator = profile_evaluator

        self.use_llm = use_llm and bool(config.YC_API_KEY and config.YC_FOLDER_ID)
        self.use_ltr = use_ltr
        self.ltr_engine: LTRRecommendationEngine | None = None

        if use_ltr:
            self.ltr_engine = LTRRecommendationEngine()
            model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
            if model_path.exists():
                try:
                    self.ltr_engine.load_model(model_path)
                    logger.info("ltr_model_loaded")
                except Exception as e:
                    logger.warning("ltr_model_load_failed", error=str(e))
                    self.ltr_engine = None

        self.cluster_weights = None
        self._load_templates()
        logger.info("recommendation_engine_initialized", ltr=self.use_ltr, llm=self.use_llm)

    def set_cluster_context(self, weights: dict[str, float]) -> None:
        self.cluster_weights = weights
        if weights:
            logger.info(
                "cluster_context_set",
                skills_count=len(weights),
                total_weight=round(sum(weights.values()), 2),
            )
        else:
            logger.info("cluster_context_empty")

    def clear_cluster_context(self):
        self.cluster_weights = None
        logger.info("cluster_context_cleared")

    def _load_templates(self):
        templates_path = config.DATA_DIR / "templates" / "recommendation_templates.json"
        if templates_path.exists():
            try:
                with open(templates_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.HARD_SKILL_TEMPLATES = data.get("hard_skills", {})
                self.SOFT_SKILL_TEMPLATES = data.get("soft_skills", {})
                self.HARD_LEARNING_PATHS = data.get("hard_paths", {})
                self.SOFT_LEARNING_PATHS = data.get("soft_paths", {})
                logger.info("templates_loaded", path=str(templates_path))
                return
            except Exception as e:
                logger.warning("templates_load_error", error=str(e))

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

    def fit(self, vacancies_skills: list[list[str]], skill_weights: dict[str, float]) -> None:
        if not vacancies_skills:
            logger.warning("no_vacancy_data_for_training")
            return
        if not skill_weights:
            raise ValueError("skill_weights обязательны для fit")

        self.comparator.fit_market(vacancies_skills)
        self.gap_analyzer = GapAnalyzer(skill_weights)
        self.comparator.set_skill_weights(skill_weights)
        self.is_fitted = True

        logger.info(
            "recommendation_engine_fitted",
            vacancies=len(vacancies_skills),
            skills=len(skill_weights),
        )

    def generate_recommendations(self, student: StudentProfile, user_type: str = "student") -> dict[str, Any]:
        """
        Генерирует рекомендации с кластерным контекстом.
        Включает сводку ближайших ролей и умные объяснения.
        """
        import time

        from src.utils import atomic_write_json

        if not hasattr(self, "profile_evaluator"):
            raise RuntimeError("RecommendationEngine должен быть инициализирован с profile_evaluator")

        profile_name = student.profile_name
        logger.info("generate_recommendations_started", profile=profile_name)

        try:
            # Шаг 1: получаем оценку профиля
            eval_result = self.profile_evaluator.evaluate_profile(student, user_type=user_type)

            if eval_result is None:
                logger.error("eval_result_is_none", profile=profile_name)
                return self._empty_recommendations()

            logger.info(
                "eval_result_received",
                profile=profile_name,
                keys=list(eval_result.keys()),
                has_cluster_context="cluster_context" in eval_result,
                has_skill_metrics="skill_metrics" in eval_result,
                has_top_recommendations="top_recommendations" in eval_result,
            )

            # Шаг 2: кластерный контекст
            cluster_context = eval_result.get("cluster_context") or {}
            closest_clusters = cluster_context.get("closest_clusters", [])
            cluster_skills_map = cluster_context.get("skills", {})
            student_set = set(s.lower() for s in student.skills)

            logger.info(
                "cluster_context_parsed",
                profile=profile_name,
                closest_count=len(closest_clusters),
                context_skills_count=len(cluster_skills_map),
            )

            # Шаг 3: формируем роли
            closest_roles = []
            for c in closest_clusters[:3]:
                name = c.get("name", f"Кластер {c['id']}")
                sim = c.get("similarity", 0)
                cluster_id = c["id"]

                cluster_all_skills = set()
                if hasattr(self.profile_evaluator, "clusterer") and self.profile_evaluator.clusterer:
                    try:
                        cluster_all_skills = set(
                            s.lower()
                            for s in self.profile_evaluator.clusterer.get_top_skills_in_cluster(cluster_id, top_n=50)
                        )
                    except Exception as e:
                        logger.warning("cluster_skills_fetch_failed", cluster=cluster_id, error=str(e))
                        cluster_all_skills = set(cluster_skills_map.keys())
                else:
                    cluster_all_skills = set(cluster_skills_map.keys())

                covered = len(student_set & cluster_all_skills)
                total = len(cluster_all_skills)

                closest_roles.append(
                    {
                        "role": name,
                        "semantic_similarity": round(sim * 100, 1),
                        "similarity_explanation": (
                            f"Ваш профиль семантически близок к этой роли на {sim * 100:.0f}%. "
                            f"Это означает, что ваш набор навыков похож на требования вакансий "
                            f"в этом кластере, но не гарантирует полного соответствия."
                        ),
                        "skills_covered": f"{covered}/{total}",
                        "coverage_percent": round(covered / total * 100, 1) if total > 0 else 0,
                        "coverage_explanation": (
                            f"Вы уже знаете {covered} из {total} ключевых навыков этой роли "
                            f"({round(covered / total * 100, 1) if total > 0 else 0}%). "
                            f"Рекомендации ниже помогут закрыть пробелы."
                        ),
                    }
                )

            # Шаг 4: формируем рекомендации
            recommendations = []
            skill_metrics = eval_result.get("skill_metrics", {})
            top_recs = eval_result.get("top_recommendations", [])

            if not top_recs:
                logger.warning("no_top_recommendations", profile=profile_name)

            for skill, score in top_recs:
                metric = skill_metrics.get(skill, {})
                try:
                    explanation = self._generate_explanation(skill, score, eval_result)
                except Exception as e:
                    logger.error("explanation_generation_failed", skill=skill, error=str(e))
                    explanation = f"Навык '{skill}' востребован на рынке."

                rec = {
                    "rank": 0,
                    "skill": skill,
                    "importance_score": score,
                    "priority": "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW",
                    "category": metric.get("category", "missing"),
                    "why_important": explanation,
                    "how_to_learn": self._get_learning_path(skill, False, student),
                    "expected_timeframe": self._get_timeframe(skill),
                    "expected_outcome": self._get_role_outcome(skill, closest_roles),
                    "is_soft_skill": not self._is_hard_skill(skill),
                    "market_frequency_percent": score * 100,
                }
                recommendations.append(rec)

            recommendations.sort(key=lambda x: x["importance_score"], reverse=True)
            for idx, rec in enumerate(recommendations, 1):
                rec["rank"] = idx

            top_recommendations = recommendations[:15]

            # Шаг 5: сохраняем LTR
            ltr_file = config.DATA_DIR / "result" / profile_name / f"ltr_recommendations_{profile_name}.json"
            ltr_file.parent.mkdir(parents=True, exist_ok=True)
            ltr_data = {
                "profile": profile_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "recommendations": [
                    {
                        "skill": rec["skill"],
                        "score": round(rec["importance_score"] * 100, 2),
                        "explanation": rec["why_important"],
                    }
                    for rec in top_recommendations[:10]
                ],
            }
            try:
                atomic_write_json(ltr_data, ltr_file)
            except Exception as e:
                logger.warning("ltr_save_failed", profile=profile_name, error=str(e))

            logger.info("generate_recommendations_completed", profile=profile_name, recs=len(top_recommendations))

            return {
                "summary": {
                    "match_score": eval_result.get("market_coverage_score", 0),
                    "confidence": eval_result.get("readiness_score", 0),
                    "market_coverage_score": eval_result.get("market_coverage_score", 0),
                    "skill_coverage": eval_result.get("skill_coverage", 0),
                    "domain_coverage_score": eval_result.get("domain_coverage_score", 0),
                    "readiness_score": eval_result.get("readiness_score", 0),
                    "avg_gap": eval_result.get("avg_gap", 0),
                    "coverage": eval_result.get("market_coverage_score", 0),
                    "coverage_details": {
                        "covered_skills_count": len(student_set & set(eval_result.get("skill_metrics", {}).keys())),
                        "total_market_skills": len(eval_result.get("skill_metrics", {})),
                    },
                    "market_skill_coverage": eval_result.get("market_skill_coverage", 0.0),
                },
                "closest_roles": closest_roles,
                "recommendations": top_recommendations,
                "domain_coverage": eval_result.get("domain_coverage", {}),
                "gaps": eval_result.get("gaps", {}),
            }

        except Exception as e:
            logger.exception("generate_recommendations_crashed", profile=profile_name, error=str(e))
            return self._empty_recommendations()

    def _empty_recommendations(self) -> dict[str, Any]:
        """Возвращает пустой результат рекомендаций при ошибке."""
        return {
            "summary": {
                "match_score": 0,
                "confidence": 0,
                "market_coverage_score": 0,
                "skill_coverage": 0,
                "domain_coverage_score": 0,
                "readiness_score": 0,
                "avg_gap": 0,
                "coverage": 0,
                "coverage_details": {"covered_skills_count": 0, "total_market_skills": 0},
                "market_skill_coverage": 0,
            },
            "closest_roles": [],
            "recommendations": [],
            "domain_coverage": {},
            "gaps": {},
        }

    def _get_role_outcome(self, skill: str, closest_roles: list[dict]) -> str:
        """
        Формирует понятный ожидаемый результат освоения навыка.
        """
        if not closest_roles:
            return f"Освоение '{skill}' расширит ваш технический кругозор."

        top_role = closest_roles[0]
        role_name = top_role["role"]
        similarity = top_role["semantic_similarity"]
        coverage = top_role["coverage_percent"]

        # Оцениваем, насколько вырастет покрытие после освоения навыка
        total_skills = int(top_role["skills_covered"].split("/")[1]) if "/" in top_role["skills_covered"] else 50
        new_coverage = round((coverage * total_skills / 100 + 1) / total_skills * 100, 1)
        improvement = round(new_coverage - coverage, 1)

        return (
            f"После освоения '{skill}' ваше покрытие навыков для роли "
            f"«{role_name}» вырастет с {coverage}% до {new_coverage}% (+{improvement}%). "
            f"Семантическая близость к роли сейчас {similarity}% — навык усилит ваши позиции "
            f"и откроет доступ к смежным вакансиям."
        )

    def _generate_explanation(self, skill: str, score: float, eval_result: dict) -> str:
        """
        Генерирует понятное объяснение, почему навык важен.
        Ключевое: навык привязывается к роли только если он реально входит в топ навыков кластера.
        """
        metric = eval_result.get("skill_metrics", {}).get(skill, {})
        cluster_rel = metric.get("cluster_relevance", 0)
        category = metric.get("category", "missing")

        # Получаем ближайший кластер — с защитой от None
        cluster_context = eval_result.get("cluster_context") or {}
        closest = cluster_context.get("closest_clusters", [])
        top_cluster = closest[0] if closest else None
        top_cluster_name = top_cluster.get("name") if top_cluster else None
        top_cluster_sim = top_cluster.get("similarity", 0) if top_cluster else 0

        # Проверяем, входит ли навык в топ-навыки ближайшего кластера
        cluster_skills = cluster_context.get("skills", {})
        skill_in_top_cluster = skill in cluster_skills

        # Дополнительно: проверяем, что навык не из категории "other"
        try:
            from src.analyzers.skill_taxonomy import SkillTaxonomy

            taxonomy = SkillTaxonomy()
            skill_cat = taxonomy.get_category_label(skill)
        except Exception:
            skill_cat = "технический"

        if category == "weak":
            prefix = "🔶 УСИЛИТЬ: "
            suffix = " У вас уже есть базовое понимание — углубите его."
        else:
            prefix = ""
            suffix = ""

        # === ТОЛЬКО если навык РЕАЛЬНО входит в топ кластера ===
        if skill_in_top_cluster and top_cluster_name and cluster_rel > 0.5:
            return (
                f"{prefix}🎯 Ключевой навык для роли «{top_cluster_name}».\n"
                f"   Ваш профиль семантически близок к этой роли ({top_cluster_sim * 100:.0f}%), "
                f"но для полного соответствия не хватает '{skill}'.\n"
                f"   Навык входит в топ-требований этого направления."
                f"{suffix}"
            )

        # === Высокий спрос на рынке (без привязки к роли) ===
        if score > 0.6:
            return (
                f"{prefix}🔴 Высокий рыночный спрос.\n"
                f"   '{skill}' ({skill_cat}) — один из самых востребованных навыков "
                f"на рынке. Работодатели часто указывают его в требованиях."
                f"{suffix}"
            )

        # === Умеренный спрос ===
        if score > 0.35:
            return (
                f"{prefix}🟡 Умеренный спрос.\n"
                f"   '{skill}' ({skill_cat}) дополнит ваш профиль и повысит "
                f"привлекательность для работодателей в смежных областях."
                f"{suffix}"
            )

        return (
            f"{prefix}🟢 Дополнительное преимущество.\n"
            f"   '{skill}' полезен для расширения кругозора в категории '{skill_cat}'."
            f"{suffix}"
        )

    def _get_learning_path(self, skill: str, is_soft: bool, student_profile: StudentProfile | None = None) -> str:
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

        # Для weak навыков — более короткий путь
        import re

        if re.search(r"(усилить|🔶)", base, re.IGNORECASE) or True:  # заглушка, можно убрать
            base = base.replace("Изучите документацию", "Углубите знания")

        return base

    def _generate_skill_recommendation(
        self,
        skill: str,
        importance: float,
        priority: str,
        rank: int,
        student_profile: StudentProfile | None = None,
        ltr_explanation: str | None = None,
        student_skills: list[str] | None = None,
        coverage: float = 0.0,
        shap_values: np.ndarray | None = None,
        X: pd.DataFrame | None = None,
        idx: int = 0,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
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
                skill,
                importance,
                priority,
                ltr_explanation,
                student_skills,
                coverage,
                shap_values,
                X,
                idx,
                feature_names,
            ),
            "how_to_learn": self._get_learning_path(skill, is_soft, student_profile),
            "expected_timeframe": self._get_timeframe(skill),
            "expected_outcome": self._get_expected_outcome(skill, student_profile),
            "market_frequency_percent": round(importance * 100, 1),
        }

    def _is_hard_skill(self, skill_lower: str) -> bool:
        """
        Определяет, является ли навык техническим (hard skill).
        Использует таксономию; fallback — ключевые слова.
        """
        # Пробуем через таксономию
        try:
            from src.analyzers.skill_taxonomy import SkillTaxonomy

            taxonomy = SkillTaxonomy()
            cat = taxonomy.get_category(skill_lower)
            # Явно soft-категории
            soft_cats = {"soft_skills", "management"}
            if cat in soft_cats:
                return False
            # Явно hard-категории
            hard_cats = {
                "programming_languages",
                "frameworks",
                "databases",
                "devops",
                "cloud",
                "data_science",
                "ml_advanced",
                "frontend",
                "mobile",
                "testing_qa",
                "security",
                "llm_ai",
                "enterprise",
                "gis",
                "embedded",
                "game_dev",
                "mathematics",
                "methodologies_concepts",
            }
            if cat in hard_cats:
                return True
            # Если категория не определена — fallback
        except Exception:
            pass

        # Fallback: проверка по ключевым словам
        hard_keywords = [
            "python",
            "java",
            "javascript",
            "typescript",
            "c++",
            "c#",
            "go",
            "rust",
            "kotlin",
            "swift",
            "php",
            "ruby",
            "scala",
            "sql",
            "postgresql",
            "mysql",
            "mongodb",
            "redis",
            "elasticsearch",
            "cassandra",
            "oracle",
            "mssql",
            "docker",
            "kubernetes",
            "k8s",
            "jenkins",
            "git",
            "gitlab",
            "github",
            "bitbucket",
            "terraform",
            "ansible",
            "prometheus",
            "grafana",
            "aws",
            "azure",
            "gcp",
            "yandex cloud",
            "machine learning",
            "deep learning",
            "nlp",
            "computer vision",
            "data science",
            "mlops",
            "pandas",
            "numpy",
            "scikit-learn",
            "tensorflow",
            "pytorch",
            "keras",
            "react",
            "vue",
            "angular",
            "django",
            "flask",
            "fastapi",
            "spring",
            "express",
            "node.js",
            "next",
            "nuxt",
            "rest api",
            "restful",
            "graphql",
            "api",
            "html",
            "css",
            "sass",
            "scss",
            "webpack",
            "vite",
            "redux",
            "mobx",
            "jest",
            "pytest",
            "cypress",
            "playwright",
            "selenium",
            "figma",
            "storybook",
            "eslint",
            "prettier",
            "babel",
            "npm",
            "yarn",
            "kafka streams",
            "greenplum",
            "powershell",
            "nginx",
            "flask",
            "linux",
            "unix",
            "bash",
            "shell",
            "cmd",
            "terminal",
            "spark",
            "hadoop",
            "airflow",
            "kafka",
            "rabbitmq",
            "celery",
        ]
        return any(kw in skill_lower for kw in hard_keywords)

    def _get_suggestion(self, skill: str, is_soft: bool) -> str:
        skill_lower = skill.lower()
        if is_soft:
            return self.SOFT_SKILL_TEMPLATES.get(
                skill_lower, f"Развитие soft skill '{skill}' улучшит вашу эффективность в команде."
            )
        return self.HARD_SKILL_TEMPLATES.get(
            skill_lower, f"Навык '{skill}' высоко востребован на рынке и повысит вашу конкурентоспособность."
        )

    def _llm_explain_with_retry(
        self,
        skill: str,
        importance: float,
        priority: str,
        student_skills: list[str],
        coverage: float,
        max_retries: int = 2,
        delay: float = 2.0,
    ) -> str | None:
        if not self.use_llm:
            return None

        prompt = (
            f"Ты — карьерный консультант в IT. Студент владеет навыками: "
            f"{', '.join(student_skills[:10])} (показано до 10).\n"
            f"Покрытие рынка составляет {coverage:.1f}%.\n"
            f"Недостающий навык: {skill}. Важность навыка: {importance:.3f} "
            f"(приоритет {priority}).\n"
            f"Объясни кратко (2-3 предложения), почему этот навык важен и "
            f"как он сочетается с уже имеющимися у студента.\n"
            f"Дай один конкретный совет по изучению."
        )

        headers = {"Authorization": f"Api-Key {config.YC_API_KEY}", "x-folder-id": config.YC_FOLDER_ID}
        payload = {
            "modelUri": f"gpt://{config.YC_FOLDER_ID}/{config.YANDEXGPT_MODEL}",
            "completionOptions": {"stream": False, "temperature": 0.7, "maxTokens": "300"},
            "messages": [{"role": "user", "text": prompt}],
        }

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["result"]["alternatives"][0]["message"]["text"].strip()
                elif resp.status_code == 429:
                    wait_time = delay * (attempt + 1)
                    logger.warning("yandexgpt_rate_limited", attempt=attempt + 1, wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    logger.warning("yandexgpt_error", status=resp.status_code, response=resp.text[:200])
                    return None
            except Exception as e:
                logger.warning("yandexgpt_exception", attempt=attempt + 1, error=str(e))
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

    def _llm_explain(
        self, skill: str, importance: float, priority: str, student_skills: list[str], coverage: float
    ) -> str | None:
        time.sleep(1.0)
        return self._llm_explain_with_retry(skill, importance, priority, student_skills, coverage)

    def _shap_explain(
        self, skill: str, shap_values: np.ndarray | None, idx: int, X: pd.DataFrame, feature_names: list[str]
    ) -> str | None:
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
            return "относится к востребованной категории навыков"
        return None

    def _why_important(
        self,
        skill: str,
        importance: float,
        priority: str,
        ltr_explanation: str | None = None,
        student_skills: list[str] | None = None,
        coverage: float = 0.0,
        shap_values: np.ndarray | None = None,
        X: pd.DataFrame | None = None,
        idx: int = 0,
        feature_names: list[str] | None = None,
    ) -> str:
        try:
            taxonomy = SkillTaxonomy()
            category = taxonomy.get_category_label(skill)
            icon = taxonomy.get_category_icon(skill)
            category_str = f" ({icon} {category})" if category != "other" else ""
        except Exception:
            category_str = ""

        if self.use_llm:
            llm_expl = self._llm_explain(skill, importance, priority, student_skills or [], coverage)
            if llm_expl:
                return f"🤖 {llm_expl}"

        if shap_values is not None and X is not None and feature_names is not None:
            shap_expl = self._shap_explain(skill, shap_values, idx, X, feature_names)
            if shap_expl:
                base = f"🎯 Навык '{skill}'{category_str} {shap_expl}."
                if priority == "HIGH":
                    base += " Это один из самых важных навыков для вашего уровня."
                elif priority == "MEDIUM":
                    base += " Его освоение повысит вашу конкурентоспособность."
                return base

        if ltr_explanation:
            return f"🎯 Модель: {ltr_explanation}"

        if priority == "HIGH":
            return (
                f"🔴 ВЫСОКИЙ приоритет: '{skill}'{category_str} — "
                f"один из самых востребованных навыков в вашей целевой роли."
            )
        elif priority == "MEDIUM":
            return f"🟡 СРЕДНИЙ приоритет: '{skill}'{category_str} значительно повысит вашу конкурентоспособность."
        else:
            return (
                f"🟢 НИЗКИЙ приоритет: '{skill}'{category_str} "
                f"полезен для расширения кругозора и специализированных задач."
            )

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

    def _get_expected_outcome(self, skill: str, student_profile: StudentProfile | None) -> str:
        role = (
            student_profile.target_role
            if student_profile and hasattr(student_profile, "target_role")
            else "вашей целевой роли"
        )
        return f"Освоение '{skill}' позволит вам уверенно работать в роли '{role}'."
