"""Оценка профилей с привязкой к целевой профессии и доменам."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from src import Result, Ok, Err, config
from src.errors import DomainError
from src.analyzers.clustering.vacancy_clustering import VacancyClusterer
from src.analyzers.comparison.comparator import CompetencyComparator
from src.analyzers.comparison.domain_analyzer import DomainAnalyzer
from src.analyzers.gap.gap_analyzer import GapAnalyzer
from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
from src.artifacts import ArtifactManifest
from src.models.data_contracts import ProfileEvaluationResult
from src.models.enums import ExperienceLevel
from src.models.student import StudentProfile
from src.parsing.api.embedding_loader import get_embedding_model
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.utils import atomic_read_json, atomic_write_json

logger = structlog.get_logger(__name__)


class ProfileEvaluator:
    def __init__(
        self,
        skill_weights: dict[str, float],
        vacancies_skills: list[list[str]],
        vacancies_skills_dict: list[dict],
        hybrid_weights: dict[str, float] | None = None,
        use_clustering: bool = True,
        skill_weights_by_level: dict[str, dict[str, float]] | None = None,
        readiness_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        level_difficulty: dict[str, float] | None = None,
    ):
        self.skill_weights = skill_weights
        self.hybrid_weights = hybrid_weights or {}
        self.vacancies_skills = vacancies_skills
        self.vacancies_skills_dict = vacancies_skills_dict
        self.comparators = {}
        self.clusterer = VacancyClusterer()
        self.use_clustering = use_clustering
        self.domain_analyzer = DomainAnalyzer()
        self.readiness_weights = readiness_weights

        self.skill_weights_by_level = skill_weights_by_level or {}
        if self.skill_weights_by_level:
            self.gap_analyzer_new = GapAnalyzer(self.skill_weights_by_level)
        else:
            self.gap_analyzer_new = None

        self.cluster_models_loaded = {
            ExperienceLevel.JUNIOR: self.clusterer.load_model(ExperienceLevel.JUNIOR),
            ExperienceLevel.MIDDLE: self.clusterer.load_model(ExperienceLevel.MIDDLE),
            ExperienceLevel.SENIOR: self.clusterer.load_model(ExperienceLevel.SENIOR),
        }
        logger.info(
            "cluster_models_loaded",
            junior=self.cluster_models_loaded["junior"],
            middle=self.cluster_models_loaded["middle"],
            senior=self.cluster_models_loaded["senior"],
        )

        self._cache = {}
        self._cache_path = config.DATA_PROCESSED_DIR / "evaluation_cache.json"
        self._load_cache()

    def evaluate_profile(
        self,
        student: StudentProfile,
        user_type: str = "student",
        target_domains: list[str] | None = None,
        taxonomy: ProfessionTaxonomy | None = None,
    ) -> Result[dict[str, Any], DomainError]:
        if not self.gap_analyzer_new:
            return Err(DomainError(message="skill_weights_by_level не были переданы в конструктор"))

        user_skills_list = student.skills
        from src.models.enums import ExperienceLevel
        _base_level = {
            ExperienceLevel.JUNIOR: 0.3,
            ExperienceLevel.MIDDLE: 0.6,
            ExperienceLevel.SENIOR: 0.9,
        }.get(student.target_level, 0.5)
        user_levels = {skill: getattr(student, 'skill_levels', {}).get(skill, _base_level) for skill in user_skills_list}
        user_skills_set = set(s.lower().strip() for s in user_skills_list)

        # === Фильтрация навыков по целевой профессии/домену ===
        domain_skill_count = 0
        if target_domains and taxonomy:
            domain_skills: set[str] = set()
            for dom in target_domains:
                domain_skills.update(taxonomy.get_domain_skills(dom))
            # Фильтруем metrics только до навыков целевого домена
            # (gap_analyzer уже посчитал по всем — будем фильтровать постфактум)
            logger.info(
                "domain_skills_selected",
                target_domains=target_domains,
                domain_skill_count=len(domain_skills),
            )
        else:
            domain_skills = set()

        # === 1. Skill-level метрики ===
        match self.gap_analyzer_new.compute_metrics(user_skills_list, user_levels):
            case Err(err):
                return Err(err)
            case Ok(val):
                metrics = val

        # Если задана целевая профессия — фильтруем метрики до её доменов
        if domain_skills:
            metrics = {s: m for s, m in metrics.items() if s.lower().strip() in domain_skills}
            domain_skill_count = len(domain_skills)
            logger.info(
                "metrics_filtered_by_domain",
                remaining_skills=len(metrics),
                total_domain_skills=len(domain_skills),
            )

        # === 2. Cluster Context + Relevance ===
        target_level = getattr(student, "target_level", ExperienceLevel.MIDDLE)
        match self._get_cluster_context(student, target_level):
            case Ok(cluster_context):
                cluster_context_val = cluster_context
                for skill, metric in metrics.items():
                    if skill in cluster_context.get("skills", {}):
                        metric.cluster_relevance = cluster_context["skills"][skill]
                    else:
                        metric.cluster_relevance = 0.15 * getattr(metric, "cluster_relevance", 0.0)
            case Err(_):
                cluster_context_val = None

        # === 3. Domain-level coverage ===
        match self.domain_analyzer.compute_domain_coverage(user_skills_list):
            case Err(err):
                return Err(err)
            case Ok(val):
                domain_coverages = val

        if domain_coverages:
            dominant_domain = max(domain_coverages.items(), key=lambda x: x[1].coverage)
            dominant_name = dominant_domain[0]
        else:
            dominant_name = None

        weighted_cov_sum = 0.0
        other_count = len(domain_coverages) - 1 if len(domain_coverages) > 1 else 0

        for dom_name, dom in domain_coverages.items():
            if dom_name == dominant_name:
                weight = config.DOMINANT_DOMAIN_WEIGHT
            else:
                remaining = 1 - config.DOMINANT_DOMAIN_WEIGHT
                weight = remaining / other_count if other_count > 0 else config.DOMINANT_DOMAIN_WEIGHT
            weighted_cov_sum += dom.coverage * weight

        domain_coverage_score = weighted_cov_sum * 100

        # === 4. Покрытие по целевой профессии (взвешенное по доменам) ===
        profession_coverage = 0.0
        profession_coverage_detail = {}
        if target_domains and taxonomy:
            profession_coverages = {}
            for dom in target_domains:
                dom_skills = taxonomy.get_domain_skills(dom)
                if dom_skills:
                    user_has = len(set(s.lower() for s in dom_skills) & user_skills_set)
                    cov = user_has / len(dom_skills)
                    profession_coverages[dom] = cov
            prof_weight = 1.0 / len(profession_coverages) if profession_coverages else 0
            for dom, cov in profession_coverages.items():
                profession_coverage += cov * prof_weight
                profession_coverage_detail[dom] = round(cov * 100, 2)
            profession_coverage *= 100

        logger.debug(
            "profession_coverage_calculated",
            target_domains=target_domains,
            profession_coverage=round(profession_coverage, 1),
            detail=profession_coverage_detail,
        )

        # === 4b. KRM competency coverage ===
        krm_coverage = {}
        if taxonomy:
            target_prof = getattr(student, "target_profession", "") or ""
            if target_prof and target_prof in taxonomy.professions:
                krm_coverage = taxonomy.compute_krm_coverage(target_prof, user_skills_list)
                logger.info(
                    "krm_coverage_calculated",
                    profession=target_prof,
                    codes=len(krm_coverage),
                    avg_coverage=round(
                        sum(v["coverage"] for v in krm_coverage.values()) / len(krm_coverage) if krm_coverage else 0, 4
                    ),
                )

        # === 5. Domain bonuses ===
        skill_to_domain_bonus = {}
        for _dom_name, dom in domain_coverages.items():
            for req_skill in dom.required_skills:
                req_norm = req_skill.lower().strip()
                bonus = dom.coverage if req_norm in user_skills_set else dom.coverage * 0.5
                if req_norm not in skill_to_domain_bonus or bonus > skill_to_domain_bonus[req_norm]:
                    skill_to_domain_bonus[req_norm] = bonus

        # === 6. Level weights ===
        level_weights = config.LEVEL_WEIGHTS_MAP.get(user_type, {"junior": 0.33, "middle": 0.34, "senior": 0.33})

        # === 7. Final scores ===
        final_scores = {}
        min_gap_for_fallback = config.GAP_ANALYZER_FALLBACK_MIN_GAP

        for skill, metric in metrics.items():
            skill_norm = skill.lower().strip()
            if skill_norm in user_skills_set:
                continue

            max_gap = max(metric.gap_j, metric.gap_m, metric.gap_s)
            if max_gap > 0.05:
                bonus = skill_to_domain_bonus.get(skill_norm, 0.0)
                final_scores[skill] = metric.score(level_weights, domain_bonus=bonus)

        if not final_scores and metrics:
            logger.warning(
                "all_market_skills_covered_fallback",
                profile_name=student.profile_name,
            )
            fallback_candidates = [
                (s, m)
                for s, m in metrics.items()
                if s.lower().strip() not in user_skills_set and max(m.gap_j, m.gap_m, m.gap_s) > min_gap_for_fallback
            ]
            for skill, metric in sorted(fallback_candidates, key=lambda x: x[1].cluster_relevance, reverse=True)[:15]:
                bonus = skill_to_domain_bonus.get(skill.lower().strip(), 0.0)
                base_score = metric.score(level_weights, domain_bonus=bonus)
                final_scores[skill] = base_score * config.GAP_ANALYZER_FALLBACK_REDUCTION

        # === 8. Итоговые метрики ===
        total_market = len(metrics)

        strong_count = 0
        weak_count = 0
        missing_count = 0
        weighted_cov = 0.0
        max_possible = 0.0

        for m in metrics.values():
            max_gap = max(m.gap_j, m.gap_m, m.gap_s)
            max_demand = max(m.demand_j, m.demand_m, m.demand_s)
            max_possible += max_demand

            if max_gap < config.SKILL_STRONG_GAP_THRESHOLD:
                strong_count += 1
                weighted_cov += 1.0 * max_demand
            elif max_gap < config.SKILL_WEAK_GAP_THRESHOLD:
                weak_count += 1
                weighted_cov += 0.5 * max_demand
            else:
                missing_count += 1

        skill_coverage = weighted_cov / max_possible * 100 if max_possible > 0 else 0.0

        market_coverage_score = 0.60 * skill_coverage + 0.40 * domain_coverage_score

        market_div = total_market if total_market > 0 else 1
        readiness = (
            config.READINESS_MARKET_WEIGHT * market_coverage_score
            + config.READINESS_SKILL_WEIGHT * (strong_count / market_div * 100)
            + config.READINESS_DOMAIN_WEIGHT * (weak_count / market_div * 100)
            + config.READINESS_GAP_PENALTY_WEIGHT * domain_coverage_score
        )

        total_gap = sum((m.gap_j + m.gap_m + m.gap_s) / 3 for m in metrics.values())
        avg_gap = total_gap / max(len(metrics), 1) if metrics else 0.0

        readiness_score = round(max(0.0, min(100.0, readiness)), 2)

        user_skills_norm: set[str] = set()
        for s in user_skills_list:
            match SkillNormalizer.normalize(s):
                case Ok(n):
                    user_skills_norm.add(n)
        all_market_skills = list(metrics.keys())
        covered_market = sum(1 for s in all_market_skills if s in user_skills_norm)
        market_skill_coverage_pct = (
            round(covered_market / len(all_market_skills) * 100, 2) if all_market_skills else 0.0
        )

        from dataclasses import asdict
        gaps = {s: asdict(m) for s, m in metrics.items() if max(m.gap_j, m.gap_m, m.gap_s) > 0.15}

        skill_categories = {"strong": strong_count, "weak": weak_count, "missing": missing_count, "total": total_market}

        eval_result = ProfileEvaluationResult(
            market_coverage_score=round(market_coverage_score, 2),
            skill_coverage=round(skill_coverage, 2),
            domain_coverage_score=round(domain_coverage_score, 2),
            readiness_score=readiness_score,
            avg_gap=round(avg_gap * 100, 2),
            skill_metrics={s: asdict(m) for s, m in metrics.items()},
            domain_coverage={d: asdict(dm) for d, dm in domain_coverages.items()},
            cluster_context=cluster_context_val,
            top_recommendations=sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:15],
            gaps=gaps,
            level_weights_used=level_weights,
            student_skills=user_skills_list,
            market_skill_coverage=market_skill_coverage_pct,
            skill_categories=skill_categories,
        )
        eval_result.profession_coverage = round(profession_coverage, 2)
        eval_result.profession_coverage_detail = profession_coverage_detail
        eval_result.domain_skill_count = domain_skill_count or total_market
        eval_result.krm_coverage = krm_coverage
        result = eval_result.model_dump()
        return Ok(result)

    def _get_cluster_context(self, student: StudentProfile, target_level: str) -> Result[dict, DomainError]:
        if not self.use_clustering:
            return Err(DomainError(message="Clustering disabled"))
        if target_level not in self.cluster_models_loaded or not self.cluster_models_loaded[target_level]:
            return Err(DomainError(message=f"Clusterer not trained for level {target_level}"))

        match self._get_or_compute_student_embedding(student):
            case Ok(student_emb):
                pass
            case Err(e):
                return Err(e)

        match self.clusterer.get_cluster_context(
            profile_embedding=student_emb, level=target_level, top_k_clusters=5, top_k_skills_per_cluster=25
        ):
            case Ok(cluster_context):
                logger.info(
                    "cluster_context_obtained",
                    level=target_level,
                    total_skills=cluster_context["total_skills_in_context"],
                    clusters_count=len(cluster_context.get("closest_clusters", [])),
                )
                return Ok(cluster_context)
            case Err(err):
                return Err(DomainError(message="Cluster context failed", detail=str(err)))

    def _get_or_create_comparator(self, target_level: str, level_analyzer=None) -> CompetencyComparator:
        if target_level in self.comparators:
            return self.comparators[target_level]

        logger.info("creating_level_comparator", level=target_level)
        comparator = CompetencyComparator(use_embeddings=True, level=target_level)
        success = comparator.fit_market(self.vacancies_skills).unwrap_or(False)
        if success:
            logger.info("level_comparator_trained", level=target_level)
        else:
            logger.warning("level_comparator_training_failed", level=target_level)

        self.comparators[target_level] = comparator
        return comparator

    def _get_recommendation(self, readiness_score: float, target_level: str) -> str:
        if readiness_score >= 80:
            return f"Готов к {target_level} уровню"
        elif readiness_score >= 60:
            return f"Неплохо для {target_level}, но есть пробелы"
        elif readiness_score >= 40:
            return f"Нужно подготовиться к {target_level}"
        else:
            return f"Недостаточно готов к {target_level}"

    def _load_cache(self):
        if self._cache_path.exists():
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    def _get_student_hash(self, student: StudentProfile, level: str) -> str:
        skills_str = ",".join(sorted(set(s.lower() for s in student.skills)))
        data = f"{level}:{skills_str}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _compute_student_hash(self, student: StudentProfile) -> str:
        skills_str = ",".join(sorted(set(s.lower() for s in student.skills)))
        return hashlib.sha256(skills_str.encode()).hexdigest()

    def _get_student_cache_path(self, student: StudentProfile) -> Path:
        return config.STUDENT_EMB_CACHE_DIR / f"{student.profile_name}_embedding.json"

    def _load_cached_embedding(self, student: StudentProfile) -> Result[np.ndarray, DomainError]:
        cache_path = self._get_student_cache_path(student)
        data = atomic_read_json(cache_path)
        if data is None:
            return Err(DomainError(message="Cache file not found or invalid"))
        if data.get("hash") == self._compute_student_hash(student):
            return Ok(np.array(data["embedding"]))
        return Err(DomainError(message="Cache hash mismatch"))

    def _save_embedding_cache(self, student: StudentProfile, embedding: np.ndarray) -> None:
        cache_path = self._get_student_cache_path(student)
        data = {"hash": self._compute_student_hash(student), "embedding": embedding.tolist()}
        atomic_write_json(data, cache_path)
        manifest = ArtifactManifest(
            artifact_path=cache_path,
            metrics={"dim": len(embedding)},
        )
        if manifest.save().is_err():
            logger.warning("student_embedding_manifest_save_failed", profile=student.profile_name)

    def _get_or_compute_student_embedding(self, student: StudentProfile) -> Result[np.ndarray, DomainError]:
        match self._load_cached_embedding(student):
            case Ok(cached):
                logger.debug("student_embedding_loaded_from_cache", profile=student.profile_name)
                return Ok(cached)
            case Err(_):
                pass

        embedding_model = get_embedding_model()
        if not student.skills:
            dim = embedding_model.get_sentence_embedding_dimension()
            emb = np.zeros(dim)
        else:
            embs = embedding_model.encode(student.skills, convert_to_numpy=True, show_progress_bar=False)
            emb = np.mean(embs, axis=0)
        self._save_embedding_cache(student, emb)
        return Ok(emb)
