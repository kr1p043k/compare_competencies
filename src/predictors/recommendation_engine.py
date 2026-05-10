# src/predictors/recommendation_engine.py
import json
import re
import time
from typing import Any

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
    Движок рекомендаций с профильной оценкой, LTR-ранжированием
    и генерацией естественноязыковых объяснений.

    Изменения относительно предыдущей версии:
    - LTR реально используется в generate_recommendations: его скоры
      смешиваются со скорами ProfileEvaluator (60/40).
    - Добавлена диверсификация рекомендаций по категориям.
    - Исправлен баг `or True` в _get_learning_path.
    - Удалены мёртвые методы: _generate_skill_recommendation, _why_important,
      _get_expected_outcome (никогда не вызывались из публичного API).
    - _get_role_outcome использует target_level вместо несуществующего target_role.
    """

    def __init__(
        self,
        use_ltr: bool = True,
        use_llm: bool = False,
        profile_evaluator=None,
    ):
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
        logger.info(
            "recommendation_engine_initialized",
            ltr=self.use_ltr,
            ltr_fitted=self.ltr_engine.is_fitted if self.ltr_engine else False,
            llm=self.use_llm,
        )

    # ------------------------------------------------------------------
    # ПУБЛИЧНЫЙ API
    # ------------------------------------------------------------------

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

    def clear_cluster_context(self) -> None:
        self.cluster_weights = None
        logger.info("cluster_context_cleared")

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
        Генерирует персонализированные рекомендации.

        Порядок работы:
        1. ProfileEvaluator оценивает профиль и возвращает top_recommendations.
        2. LTRRecommendationEngine (если обучен) возвращает свои скоры.
        3. Скоры смешиваются: 60% evaluator + 40% LTR.
        4. Рекомендации диверсифицируются по категориям навыков.
        """
        from src.utils import atomic_write_json

        if not hasattr(self, "profile_evaluator") or self.profile_evaluator is None:
            raise RuntimeError("RecommendationEngine должен быть инициализирован с profile_evaluator")

        profile_name = student.profile_name
        logger.info("generate_recommendations_started", profile=profile_name)

        try:
            # ── Шаг 1: оценка профиля ─────────────────────────────────────
            eval_result = self.profile_evaluator.evaluate_profile(student, user_type=user_type)
            if eval_result is None:
                logger.error("eval_result_is_none", profile=profile_name)
                return self._empty_recommendations()

            logger.info(
                "eval_result_received",
                profile=profile_name,
                keys=list(eval_result.keys()),
            )

            # ── Шаг 2: кластерный контекст ────────────────────────────────
            cluster_context = eval_result.get("cluster_context") or {}
            closest_clusters = cluster_context.get("closest_clusters", [])
            cluster_skills_map = cluster_context.get("skills", {})
            student_set = set(s.lower() for s in student.skills)

            # ── Шаг 3: ближайшие роли ─────────────────────────────────────
            closest_roles = self._build_closest_roles(closest_clusters, cluster_skills_map, student_set)

            # ── Шаг 4: скоры от ProfileEvaluator ─────────────────────────
            skill_metrics = eval_result.get("skill_metrics", {})
            top_recs: list[tuple[str, float]] = eval_result.get("top_recommendations", [])

            if not top_recs:
                logger.warning("no_top_recommendations", profile=profile_name)

            evaluator_scores: dict[str, float] = {skill: score for skill, score in top_recs}

            # ── Шаг 5: скоры от LTR (если модель обучена) ────────────────
            ltr_scores: dict[str, float] = {}
            if self.ltr_engine and self.ltr_engine.is_fitted:
                all_market = list(self.ltr_engine.skill_metadata.keys())
                missing_for_ltr = [s for s in all_market if s not in student_set]
                try:
                    ltr_impacts = self.ltr_engine.predict_skill_impact(student.skills, missing_for_ltr)
                    # Нормализуем LTR-скоры к диапазону evaluator (0–1)
                    max_ltr = max((s for _, s, _ in ltr_impacts), default=1.0) or 1.0
                    ltr_scores = {skill: score / max_ltr for skill, score, _ in ltr_impacts}
                    logger.info(
                        "ltr_scores_computed",
                        profile=profile_name,
                        ltr_skills=len(ltr_scores),
                    )
                except Exception as e:
                    logger.warning("ltr_scoring_failed", error=str(e))

            # ── Шаг 6: смешиваем скоры (60% evaluator + 40% LTR) ─────────
            all_skills_to_rank = set(evaluator_scores) | set(ltr_scores)
            combined_scores: dict[str, float] = {}
            for skill in all_skills_to_rank:
                ev = evaluator_scores.get(skill, 0.0)
                ltr = ltr_scores.get(skill, 0.0)
                if ltr_scores:
                    combined_scores[skill] = 0.6 * ev + 0.4 * ltr
                else:
                    combined_scores[skill] = ev

            # ── Шаг 7: формируем объекты рекомендаций ────────────────────
            recommendations: list[dict[str, Any]] = []
            for skill, score in combined_scores.items():
                metric = skill_metrics.get(skill, {})
                try:
                    explanation = self._generate_explanation(skill, score, eval_result)
                except Exception as e:
                    logger.error("explanation_failed", skill=skill, error=str(e))
                    explanation = f"Навык '{skill}' востребован на рынке."

                recommendations.append(
                    {
                        "rank": 0,
                        "skill": skill,
                        "importance_score": score,
                        "priority": ("HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"),
                        "category": metric.get("category", "missing"),
                        "why_important": explanation,
                        "how_to_learn": self._get_learning_path(skill, False, student),
                        "expected_timeframe": self._get_timeframe(skill),
                        "expected_outcome": self._get_role_outcome(skill, closest_roles),
                        "is_soft_skill": not self._is_hard_skill(skill),
                        "market_frequency_percent": score * 100,
                    }
                )

            recommendations.sort(key=lambda x: x["importance_score"], reverse=True)

            # ── Шаг 8: диверсификация категорий ──────────────────────────
            # Не более 3 навыков одной категории подряд в топе
            top_recommendations = self._diversify_recommendations(recommendations, max_per_category=3)[:15]

            for idx, rec in enumerate(top_recommendations, 1):
                rec["rank"] = idx

            # ── Шаг 9: сохраняем результат LTR для дебага ────────────────
            ltr_file = config.DATA_DIR / "result" / profile_name / f"ltr_recommendations_{profile_name}.json"
            ltr_file.parent.mkdir(parents=True, exist_ok=True)
            ltr_data = {
                "profile": profile_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ltr_used": bool(ltr_scores),
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

            logger.info(
                "generate_recommendations_completed",
                profile=profile_name,
                recs=len(top_recommendations),
                ltr_contributed=bool(ltr_scores),
            )

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

    # ------------------------------------------------------------------
    # ПРИВАТНЫЕ МЕТОДЫ
    # ------------------------------------------------------------------

    def _build_closest_roles(
        self,
        closest_clusters: list[dict],
        cluster_skills_map: dict,
        student_set: set[str],
    ) -> list[dict]:
        roles = []
        for c in closest_clusters[:3]:
            name = c.get("name", f"Кластер {c['id']}")
            sim = c.get("similarity", 0)
            cluster_id = c["id"]

            cluster_all_skills: set[str] = set()
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

            roles.append(
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
        return roles

    def _diversify_recommendations(self, recs: list[dict], max_per_category: int = 3) -> list[dict]:
        """
        Диверсифицирует рекомендации: не более max_per_category навыков
        одной категории в приоритетной части. Остальные добавляются следом.
        """
        seen: dict[str, int] = {}
        priority: list[dict] = []
        leftover: list[dict] = []

        for rec in recs:
            cat = rec.get("category", "other")
            if seen.get(cat, 0) < max_per_category:
                seen[cat] = seen.get(cat, 0) + 1
                priority.append(rec)
            else:
                leftover.append(rec)

        return priority + leftover

    def _empty_recommendations(self) -> dict[str, Any]:
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
                "coverage_details": {
                    "covered_skills_count": 0,
                    "total_market_skills": 0,
                },
                "market_skill_coverage": 0,
            },
            "closest_roles": [],
            "recommendations": [],
            "domain_coverage": {},
            "gaps": {},
        }

    def _get_role_outcome(self, skill: str, closest_roles: list[dict]) -> str:
        """Формирует ожидаемый результат освоения навыка через ближайшую роль."""
        if not closest_roles:
            return f"Освоение '{skill}' расширит ваш технический кругозор."

        top_role = closest_roles[0]
        role_name = top_role["role"]
        similarity = top_role["semantic_similarity"]
        coverage = top_role["coverage_percent"]

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
        Привязывает навык к роли только если он реально входит в топ кластера.
        """
        metric = eval_result.get("skill_metrics", {}).get(skill, {})
        cluster_rel = metric.get("cluster_relevance", 0)
        category = metric.get("category", "missing")

        cluster_context = eval_result.get("cluster_context") or {}
        closest = cluster_context.get("closest_clusters", [])
        top_cluster = closest[0] if closest else None
        top_cluster_name = top_cluster.get("name") if top_cluster else None
        top_cluster_sim = top_cluster.get("similarity", 0) if top_cluster else 0

        cluster_skills = cluster_context.get("skills", {})
        skill_in_top_cluster = skill in cluster_skills

        try:
            taxonomy = SkillTaxonomy()
            skill_cat = taxonomy.get_category_label(skill)
        except Exception:
            skill_cat = "технический"

        prefix = "🔶 УСИЛИТЬ: " if category == "weak" else ""
        suffix = " У вас уже есть базовое понимание — углубите его." if category == "weak" else ""

        if skill_in_top_cluster and top_cluster_name and cluster_rel > 0.5:
            return (
                f"{prefix}🎯 Ключевой навык для роли «{top_cluster_name}».\n"
                f"   Ваш профиль семантически близок к этой роли "
                f"({top_cluster_sim * 100:.0f}%), но для полного соответствия "
                f"не хватает '{skill}'.\n"
                f"   Навык входит в топ-требований этого направления."
                f"{suffix}"
            )

        if score > 0.6:
            return (
                f"{prefix}🔴 Высокий рыночный спрос.\n"
                f"   '{skill}' ({skill_cat}) — один из самых востребованных навыков "
                f"на рынке. Работодатели часто указывают его в требованиях."
                f"{suffix}"
            )

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

    def _get_learning_path(
        self,
        skill: str,
        is_soft: bool,
        student_profile: StudentProfile | None = None,
    ) -> str:
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

        # Для weak навыков — более короткая формулировка
        if re.search(r"(усилить|🔶)", base, re.IGNORECASE):
            base = base.replace("Изучите документацию", "Углубите знания")

        return base

    def _is_hard_skill(self, skill_lower: str) -> bool:
        """Определяет, является ли навык техническим (hard skill)."""
        try:
            taxonomy = SkillTaxonomy()
            cat = taxonomy.get_category(skill_lower)
            soft_cats = {"soft_skills", "management"}
            if cat in soft_cats:
                return False
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
        except Exception:
            pass

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
            "kafka",
            "rabbitmq",
            "celery",
            "spark",
            "hadoop",
            "airflow",
            "nginx",
            "linux",
            "unix",
            "bash",
            "shell",
        ]
        return any(kw in skill_lower for kw in hard_keywords)

    def _get_suggestion(self, skill: str, is_soft: bool) -> str:
        skill_lower = skill.lower()
        if is_soft:
            return self.SOFT_SKILL_TEMPLATES.get(
                skill_lower,
                f"Развитие soft skill '{skill}' улучшит вашу эффективность в команде.",
            )
        return self.HARD_SKILL_TEMPLATES.get(
            skill_lower,
            f"Навык '{skill}' высоко востребован на рынке и повысит вашу конкурентоспособность.",
        )

    def _get_priority(self, gap: float) -> str:
        if gap > 0.55:
            return "HIGH"
        elif gap > 0.30:
            return "MEDIUM"
        return "LOW"

    def _get_timeframe(self, skill: str) -> str:
        skill_lower = skill.lower()
        easy = {"git", "html", "css", "sass", "english", "английский язык"}
        medium = {"javascript", "python", "sql", "redis", "docker", "react"}
        hard = {"java", "kubernetes", "aws", "machine learning", "tensorflow"}

        if any(k in skill_lower for k in easy):
            return "1-2 недели"
        if any(k in skill_lower for k in medium):
            return "1-2 месяца"
        if any(k in skill_lower for k in hard):
            return "2-6 месяцев"
        return "1-3 месяца"

    # ------------------------------------------------------------------
    # LLM (YandexGPT) — вспомогательный, необязательный
    # ------------------------------------------------------------------

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

        headers = {
            "Authorization": f"Api-Key {config.YC_API_KEY}",
            "x-folder-id": config.YC_FOLDER_ID,
        }
        payload = {
            "modelUri": f"gpt://{config.YC_FOLDER_ID}/{config.YANDEXGPT_MODEL}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.7,
                "maxTokens": "300",
            },
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
                    logger.warning(
                        "yandexgpt_rate_limited",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        "yandexgpt_error",
                        status=resp.status_code,
                        response=resp.text[:200],
                    )
                    return None
            except Exception as e:
                logger.warning("yandexgpt_exception", attempt=attempt + 1, error=str(e))
                if attempt < max_retries:
                    time.sleep(delay)
                else:
                    return None
        return None

    def _llm_explain(
        self,
        skill: str,
        importance: float,
        priority: str,
        student_skills: list[str],
        coverage: float,
    ) -> str | None:
        time.sleep(1.0)
        return self._llm_explain_with_retry(skill, importance, priority, student_skills, coverage)

    # ------------------------------------------------------------------
    # ШАБЛОНЫ
    # ------------------------------------------------------------------

    def _load_templates(self) -> None:
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
