# src/predictors/ltr_recommendation_engine.py

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import structlog
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, ndcg_score, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src import (
    ModelError,
    ModelNotFoundError,
    ModelTrainingError,
    Ok,
    Err,
    Result,
    config,
)
from src.analyzers.skills.skill_filter import SkillFilter
from src.analyzers.skills.skill_level_analyzer import SkillLevelAnalyzer
from src.artifacts import ArtifactManifest
from src.parsing.api.embedding_loader import get_embedding_model
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.parsing.skills.vacancy_parser import VacancyParser
from src.predictors.base import RankingPredictor
from src.predictors.models import SkillImpact

logger = structlog.get_logger(__name__)

def _load_profession_profiles() -> list[list[str]]:
    """Load profession profiles from profession_taxonomy.json + krm_competency_mapping.json."""
    profiles: list[list[str]] = [[]]  # empty profile (no experience)
    try:
        import json
        from pathlib import Path
        from src import config
        prof_path = config.REFERENCE_DIR / "profession_taxonomy.json"
        krm_path = config.REFERENCE_DIR / "krm_competency_mapping.json"
        if not prof_path.exists() or not krm_path.exists():
            raise FileNotFoundError("profession taxonomy files not found")
        taxonomy = json.loads(prof_path.read_text(encoding="utf-8"))
        krm = json.loads(krm_path.read_text(encoding="utf-8"))
        professions = taxonomy.get("professions", {})
        seen: set[str] = set()
        for prof_name, prof_data in professions.items():
            codes = prof_data.get("competency_codes", [])
            skills: set[str] = set()
            for code in codes:
                for kw in krm.get(code, []):
                    kw_norm = kw.lower().strip()
                    if kw_norm and len(kw_norm) > 2 and kw_norm not in seen:
                        seen.add(kw_norm)
                        skills.add(kw_norm)
            if skills:
                profiles.append(sorted(skills)[:30])
        logger.info("loaded_profession_profiles", count=len(profiles) - 1)
    except Exception as exc:
        logger.warning("failed_to_load_profession_profiles_using_defaults", error=str(exc))
        profiles = [[],
            ["python", "sql", "git"],
            ["javascript", "react", "html", "css", "git"],
            ["python", "pandas", "numpy", "jupyter", "sql"],
            ["docker", "kubernetes", "linux", "bash", "terraform"],
            ["java", "spring", "postgresql", "rest api", "git"],
            ["python", "tensorflow", "sklearn", "numpy", "pandas"],
            ["python", "fastapi", "postgresql", "redis", "docker"],
            ["kotlin", "android", "retrofit", "room", "git"],
            ["go", "grpc", "docker", "kafka", "postgresql"],
        ]
    return profiles


_SYNTHETIC_DOMAIN_PROFILES: list[list[str]] | None = None


def _get_domain_profiles() -> list[list[str]]:
    global _SYNTHETIC_DOMAIN_PROFILES
    if _SYNTHETIC_DOMAIN_PROFILES is None:
        _SYNTHETIC_DOMAIN_PROFILES = _load_profession_profiles()
    return _SYNTHETIC_DOMAIN_PROFILES


class LTRRecommendationEngine(RankingPredictor["LTRRecommendationEngine", list[SkillImpact]]):
    """
    LTR-движок на XGBoost для персонализированного ранжирования навыков.
    """

    def __init__(self, model_path: Path | None = None):
        self.vacancy_parser = VacancyParser()
        self.skill_filter = SkillFilter()
        self.level_analyzer = SkillLevelAnalyzer()
        self.embedding_model = get_embedding_model()

        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []
        self.skill_metadata: dict[str, dict[str, Any]] = {}
        self.skill_embeddings: dict[str, np.ndarray] = {}
        self.vacancy_skills_corpus: list[set[str]] = []
        self.category_avg_weight: dict[str, float] = {}
        self.total_vacancies = 0
        self.is_fitted = False

        # SkillTaxonomy — синглтон, кешируем ссылку один раз, не создаём на каждый навык
        self._taxonomy = None

        if model_path is None:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
        else:
            self.model_path = model_path

    @property
    def name(self) -> str:
        return "LTRRanking"

    # ------------------------------------------------------------------
    # ОБУЧЕНИЕ
    # ------------------------------------------------------------------

    def fit(self, vacancies: list[dict]) -> Result["LTRRecommendationEngine", ModelError]:
        if len(vacancies) < 50:
            logger.warning("insufficient_vacancies_for_training")
            return Err(ModelTrainingError(
                message="Недостаточно вакансий для обучения",
                n_samples=len(vacancies),
            ))

        logger.info("ltr_training_started", vacancies=len(vacancies))

        match self.vacancy_parser.extract_skills_from_vacancies(vacancies):
            case Ok(result):
                frequencies = result["frequencies"]
                hybrid_weights = result.get("hybrid_weights", {})
            case Err(e):
                return Err(ModelTrainingError(message=str(e), detail="extract_skills_from_vacancies"))

        processed_vacancies = self._prepare_vacancies_for_levels(vacancies)
        self.level_analyzer.analyze_vacancies(processed_vacancies)

        self.vacancy_skills_corpus = [set(self._extract_skills_from_vacancy(v)) for v in vacancies]
        self.total_vacancies = len(vacancies)

        all_skills = list(frequencies.keys())
        if len(all_skills) < 5:
            logger.error("too_few_skills_for_training")
            return Err(ModelTrainingError(
                message="Слишком мало навыков для обучения LTR-модели",
                n_samples=len(all_skills),
            ))

        embeddings = self.embedding_model.encode(all_skills, convert_to_numpy=True, show_progress_bar=True)
        self.skill_embeddings = {skill: emb for skill, emb in zip(all_skills, embeddings, strict=False)}

        max_hybrid = max(hybrid_weights.values()) if hybrid_weights else 1.0
        max_freq = max(frequencies.values()) if frequencies else 1.0

        for skill, freq in frequencies.items():
            self.skill_metadata[skill] = {
                "frequency": freq,
                "hybrid_weight": hybrid_weights.get(skill, 0.0),
                "level": self.level_analyzer.get_skill_level(skill),
                "category": self._get_skill_category(skill),
                "hybrid_weight_normalized": (
                    hybrid_weights.get(skill, 0.0) / max_hybrid if max_hybrid > 0 else freq / max_freq
                ),
                "freq_normalized": freq / max_freq if max_freq > 0 else 0.0,
            }

        logger.info("generating_training_samples")
        X_rows: list[dict] = []
        y_rows: list[float] = []
        market_emb = np.mean(list(self.skill_embeddings.values()), axis=0) if self.skill_embeddings else None

        for skill in all_skills:
            base_target = self.skill_metadata[skill]["hybrid_weight_normalized"]

            domain_profiles = _get_domain_profiles()
            for domain_profile in domain_profiles:
                student_skills = [s for s in domain_profile if s in self.skill_embeddings]
                student_emb = (
                    np.mean([self.skill_embeddings[s] for s in student_skills], axis=0)
                    if student_skills
                    else market_emb
                )
                features = self._extract_features(skill, student_emb, student_skills)
                if skill in domain_profile:
                    relevance = 1.0
                elif any(skill in other for other in domain_profiles if other is not domain_profile):
                    relevance = 0.5
                else:
                    relevance = 0.3
                target = base_target * relevance
                X_rows.append(features)
                y_rows.append(target)

        X = pd.DataFrame(X_rows)
        y = np.array(y_rows)
        self.feature_names = X.columns.tolist()

        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=config.GLOBAL_RANDOM_SEED)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=config.GLOBAL_RANDOM_SEED
        )

        # Compute category_avg_weight from training data only (prevent data leakage)
        n_profiles = len(domain_profiles)
        train_indices_set = set(X_train.index)
        train_cat_buckets: dict[str, list[float]] = {}
        for skill_idx, skill in enumerate(all_skills):
            skill_start = skill_idx * n_profiles
            skill_end = skill_start + n_profiles
            if any(i in train_indices_set for i in range(skill_start, skill_end)):
                meta = self.skill_metadata.get(skill, {})
                cat = meta.get("category", "other")
                train_cat_buckets.setdefault(cat, []).append(meta.get("hybrid_weight", 0.0))
        self.category_avg_weight = {cat: float(np.mean(vals)) for cat, vals in train_cat_buckets.items()}

        # Update category_avg_weight column in all splits with train-only values
        cat_enc_to_avg = {1: self.category_avg_weight.get("soft_skills", 0.0),
                          2: self.category_avg_weight.get("mobile", 0.0),
                          3: self.category_avg_weight.get("databases", 0.0),
                          4: self.category_avg_weight.get("frameworks", 0.0),
                          5: self.category_avg_weight.get("programming_languages", 0.0)}
        for df in [X_train, X_val, X_test]:
            df["category_avg_weight"] = df["category_encoded"].map(lambda c: cat_enc_to_avg.get(c, 0.0))

        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.GLOBAL_RANDOM_SEED if hasattr(config, "GLOBAL_RANDOM_SEED") else 42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=30,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.is_fitted = True

        pred_test = self.model.predict(X_test)
        r2 = r2_score(y_test, pred_test)
        mae = mean_absolute_error(y_test, pred_test)
        try:
            from scipy.stats import spearmanr
            spear, _ = spearmanr(y_test, pred_test)
        except Exception:
            spear = float("nan")

        try:
            ndcg = ndcg_score(y_test, pred_test, k=min(10, len(y_test)))
        except Exception:
            ndcg = float("nan")

        logger.info(
            "ltr_training_completed",
            r2=round(r2, 4),
            mae=round(mae, 4),
            ndcg_at_10=round(ndcg, 4) if not np.isnan(ndcg) else "n/a",
            spearman_rho=round(spear, 4) if not np.isnan(spear) else "n/a",
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
        )

        # Сохраняем график важности с защитой от ошибок
        try:
            plt.figure(figsize=(12, 8))
            xgb.plot_importance(self.model, max_num_features=15)
            plt.savefig(config.MODELS_DIR / "ltr_feature_importance.png", dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning("failed_to_save_importance_plot", error=str(e))

        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "skill_metadata": self.skill_metadata,
                "skill_embeddings": self.skill_embeddings,
                "vacancy_skills_corpus": self.vacancy_skills_corpus,
                "category_avg_weight": self.category_avg_weight,
                "total_vacancies": self.total_vacancies,
            },
            self.model_path,
        )
        logger.info("model_saved", path=str(self.model_path))

        # ── Сохранение манифеста ──
        data_hash = hashlib.sha256(
            json.dumps([v.get("id", "") for v in vacancies[:200]], sort_keys=True).encode()
        ).hexdigest()
        manifest = ArtifactManifest(
            artifact_path=self.model_path,
            data_hash=data_hash,
            metrics={
                "r2": round(r2, 4),
                "mae": round(mae, 4),
                "ndcg_at_10": round(ndcg, 4) if not np.isnan(ndcg) else 0.0,
            },
        )
        if manifest.save().is_err():
            logger.warning("manifest_save_failed")

        self.last_metrics = {
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "ndcg": round(ndcg, 4) if not np.isnan(ndcg) else 0.0,
        }
        return Ok(self)

    # ------------------------------------------------------------------
    # ПРИЗНАКИ
    # ------------------------------------------------------------------

    def _extract_features(
        self, skill: str, student_emb: np.ndarray | None, student_skills: list[str]
    ) -> dict[str, float]:
        meta = self.skill_metadata.get(skill, {})
        skill_emb = self.skill_embeddings.get(skill)

        sim = 0.0
        if skill_emb is not None and student_emb is not None:
            sim = float(cosine_similarity([skill_emb], [student_emb])[0][0])

        level_map = {"junior": 1, "middle": 2, "senior": 3, "all_levels": 2}
        category = self._get_skill_category(skill)

        category_map = {
            "programming_languages": 5,
            "frameworks": 4,
            "devops": 4,
            "cloud": 4,
            "data_science": 4,
            "ml_advanced": 4,
            "llm_ai": 4,
            "databases": 3,
            "frontend": 3,
            "testing_qa": 3,
            "mobile": 2,
            "security": 2,
            "enterprise": 2,
            "embedded": 2,
            "game_dev": 2,
            "gis": 2,
            "mathematics": 2,
            "methodologies_concepts": 2,
            "soft_skills": 1,
            "management": 1,
            "other": 1,
        }

        cat_avg = self.category_avg_weight.get(category, 0.0) if self.category_avg_weight else 0.0
        return {
            "level_encoded": level_map.get(meta.get("level", "middle"), 2),
            "category_encoded": category_map.get(category, 1),
            "cosine_sim": sim,
            "in_student_profile": 1.0 if skill in student_skills else 0.0,
            "skill_freq_normalized": meta.get("freq_normalized", 0.0),
            "co_occurrence": self._co_occurrence_score(skill, student_skills),
            "category_avg_weight": cat_avg,
            "student_skills_count": min(len(student_skills) / 50.0, 1.0),
        }

    # ------------------------------------------------------------------
    # ПРЕДСКАЗАНИЕ
    # ------------------------------------------------------------------

    def predict_impact(
        self, student_skills: list[str], missing_skills: list[str]
    ) -> Result[list[SkillImpact], ModelError]:
        match self.predict_skill_impact(student_skills, missing_skills):
            case Ok(impacts):
                return Ok([
                    SkillImpact(skill=s, score=sc, explanation=ex)
                    for s, sc, ex in impacts
                ])
            case Err(e):
                return Err(e)

    def predict_skill_impact(
        self, student_skills: list[str], missing_skills: list[str]
    ) -> Result[list[tuple[str, float, str]], ModelError]:
        match self.predict_skill_impact_with_shap(student_skills, missing_skills, compute_shap=True):
            case Ok((recs, _, _)):
                return Ok(recs)
            case Err(e):
                return Err(e)

    def predict_skill_impact_with_shap(
        self, student_skills: list[str], missing_skills: list[str], compute_shap: bool = True
    ) -> Result[tuple[list[tuple[str, float, str]], np.ndarray | None, pd.DataFrame | None], ModelError]:
        if not self.is_fitted or self.model is None:
            logger.warning("model_not_trained")
            return Err(ModelError(model_name="ltr_ranker", message="Model not fitted"))

        match self._get_student_embedding(student_skills):
            case Ok(emb):
                student_emb = emb
            case Err(_):
                student_emb = np.mean(list(self.skill_embeddings.values()), axis=0) if self.skill_embeddings else None
        if student_emb is None:
            return Err(ModelError(model_name="ltr_ranker", message="No embeddings available for prediction"))

        X_rows = []
        valid_missing = []
        for skill in missing_skills:
            if skill not in self.skill_metadata:
                continue
            features = self._extract_features(skill, student_emb, student_skills)
            X_rows.append(features)
            valid_missing.append(skill)

        if not X_rows:
            return Ok(([], None, None))

        X = pd.DataFrame(X_rows)[self.feature_names]
        raw_scores = self.model.predict(X)
        raw_scores = np.clip(raw_scores, 0.0, 1.0)

        if len(raw_scores) > 1:
            exp_scores = np.exp(raw_scores * 5)
            scores = exp_scores / exp_scores.sum()
        else:
            scores = raw_scores

        shap_values = None
        if compute_shap:
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
            except Exception as e:
                logger.warning("shap_computation_failed", error=str(e))

        impacts = []
        for i, skill in enumerate(valid_missing):
            score = float(scores[i])
            explanation = self._generate_explanation(skill, score, shap_values, i, X, student_skills)
            impacts.append((skill, round(score * 100, 2), explanation))

        return Ok((sorted(impacts, key=lambda x: x[1], reverse=True)[:15], shap_values, X))

    # ------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ------------------------------------------------------------------

    def _co_occurrence_score(self, skill: str, student_skills: list[str]) -> float:
        if not student_skills or not self.vacancy_skills_corpus:
            return 0.0
        student_set = set(student_skills)
        hits = 0
        docs_with_skill = 0
        for vac_set in self.vacancy_skills_corpus:
            if skill in vac_set:
                docs_with_skill += 1
                if vac_set & student_set:
                    hits += 1
        return hits / max(docs_with_skill, 1)

    def _get_student_embedding(self, student_skills: list[str]) -> Result[np.ndarray, ModelError]:
        if not student_skills:
            return Err(ModelError(model_name="ltr_ranker", message="No student skills provided"))
        valid = [s for s in student_skills if s in self.skill_embeddings]
        if not valid:
            return Err(ModelError(model_name="ltr_ranker", message="No known skills in student profile"))
        return Ok(np.mean([self.skill_embeddings[s] for s in valid], axis=0))

    def _extract_skills_from_vacancy(self, vac: dict) -> list[str]:
        from src import Ok

        def _normalize(skill: str) -> str | None:
            match SkillNormalizer.normalize(skill):
                case Ok(n):
                    return n
                case _:
                    return None

        skills: set[str] = set()
        for ks in vac.get("key_skills", []):
            name = ks.get("name", "")
            if name:
                norm = _normalize(name)
                if norm:
                    skills.add(norm)
        desc = vac.get("description", "")
        if desc:
            match self.vacancy_parser.extract_skills_from_description(desc):
                case Ok(skills_list):
                    for s in skills_list:
                        norm = _normalize(s)
                        if norm:
                            skills.add(norm)
                case _:
                    pass
        if not skills:
            snippet = vac.get("snippet") or {}
            combined = f"{snippet.get('requirement', '')} {snippet.get('responsibility', '')}"
            if combined.strip():
                match self.vacancy_parser.extract_skills_from_description(combined):
                    case Ok(skills_list):
                        for s in skills_list:
                            norm = _normalize(s)
                            if norm:
                                skills.add(norm)
                    case _:
                        pass
        return list(skills)

    def _prepare_vacancies_for_levels(self, vacancies: list[dict]) -> list[dict]:
        processed = []
        for vac in vacancies:
            skills = self._extract_skills_from_vacancy(vac)
            exp_raw = vac.get("experience", {})
            if isinstance(exp_raw, dict):
                exp_name = exp_raw.get("name", "").lower()
                if "junior" in exp_name or "младший" in exp_name:
                    experience = "junior"
                elif "senior" in exp_name or "старший" in exp_name:
                    experience = "senior"
                else:
                    experience = "middle"
            else:
                experience = exp_raw if isinstance(exp_raw, str) else "middle"
            processed.append({"skills": skills, "experience": experience})
        return processed

    def _get_skill_category(self, skill: str) -> str:
        try:
            if self._taxonomy is None:
                from src.analyzers.skills.skill_taxonomy import SkillTaxonomy

                self._taxonomy = SkillTaxonomy()
            cat = self._taxonomy.get_category(skill)
            if cat and cat != "other":
                return cat
        except Exception as e:
            logger.warning("skill_taxonomy_get_category_failed", skill=skill, error=str(e))
        match self.skill_filter.get_skill_categories([skill]):
            case Ok(cats):
                for cat, cat_skills in cats.items():
                    if skill in cat_skills:
                        return cat
            case _:
                pass
        return "other"

    def _fallback_impacts(self, missing_skills: list[str]) -> list[tuple[str, float, str]]:
        impacts = []
        total = max(self.total_vacancies, 1)
        for skill in missing_skills:
            freq = self.skill_metadata.get(skill, {}).get("frequency", 0)
            score = (freq / total) * 100
            impacts.append((skill, round(score, 2), f"Встречается в {freq} вакансиях"))
        return sorted(impacts, key=lambda x: x[1], reverse=True)[:10]

    def _generate_explanation(
        self,
        skill: str,
        score: float,
        shap_values: np.ndarray | None,
        idx: int,
        X: pd.DataFrame,
        student_skills: list[str] | None = None,
    ) -> str:
        meta = self.skill_metadata.get(skill, {})
        freq = meta.get("frequency", 0)
        level = meta.get("level", "middle")

        base = ""
        if shap_values is not None and idx < len(shap_values):
            top_idx = int(np.argmax(np.abs(shap_values[idx])))
            feat_name = self.feature_names[top_idx]
            feat_val = X.iloc[idx][feat_name]

            if feat_name == "cosine_sim":
                base = (
                    f"🎯 {skill}: хорошо сочетается с вашим профилем "
                    f"(сходство {feat_val:.2f}, важность {score * 100:.1f}%)"
                )
            elif feat_name == "co_occurrence":
                base = (
                    f"🔗 {skill}: часто встречается в вакансиях вместе с вашими навыками "
                    f"(co-occurrence {feat_val:.2f}, важность {score * 100:.1f}%)"
                )
            elif feat_name == "level_encoded":
                level_str = {1: "junior", 2: "middle", 3: "senior"}.get(int(feat_val), "middle")
                base = f"📊 {skill}: востребован на уровне {level_str} (важность {score * 100:.1f}%, частота {freq})"
            elif feat_name in ("category_encoded", "category_avg_weight"):
                cat = self._get_skill_category(skill)
                base = f"📁 {skill}: относится к востребованной категории '{cat}' (важность {score * 100:.1f}%)"

        if not base:
            base = f"{skill}: важность {score * 100:.1f}% (частота {freq}, уровень {level})"

        cross = self._generate_cross_domain_explanation(skill, student_skills or [], score)
        if cross:
            base += "\n" + cross

        return base

    def _get_dominant_student_category(self, student_skills: list[str]) -> str:
        cats = {}
        for s in student_skills:
            c = self._get_skill_category(s)
            cats[c] = cats.get(c, 0) + 1
        return max(cats, key=cats.get) if cats else "other"

    def _generate_cross_domain_explanation(
        self, skill: str, student_skills: list[str], score: float
    ) -> str:
        skill_cat = self._get_skill_category(skill)
        student_cat = self._get_dominant_student_category(student_skills)
        if skill_cat == student_cat or skill_cat == "other":
            return ""

        from src.analyzers.skills.skill_taxonomy import SkillTaxonomy
        tax = SkillTaxonomy()
        skill_cat_label = tax.get_category_label_by_id(skill_cat) or skill_cat
        student_cat_label = tax.get_category_label_by_id(student_cat) or student_cat

        meta = self.skill_metadata.get(skill, {})
        freq = meta.get("frequency", 0)

        return (
            f"🌍 Междисциплинарный навык: {skill} относится к «{skill_cat_label}», "
            f"но востребован и в «{student_cat_label}». "
            f"Встречается в {freq} вакансиях на рынке "
            f"(важность {score * 100:.1f}%). "
            f"Его изучение расширит вашу применимость на стыке доменов."
        )

    def load_model(self, path: Path | None = None) -> Result["LTRRecommendationEngine", ModelError]:
        model_path = path or self.model_path
        if not model_path.exists():
            logger.error("model_file_not_found", path=str(model_path))
            return Err(ModelNotFoundError(
                message="Файл LTR-модели не найден",
                model_name="ltr_ranker_xgb_regressor",
                path=str(model_path),
            ))

        manifest_path = model_path.with_suffix(".manifest.json")
        if manifest_path.exists():
            match ArtifactManifest.load(model_path):
                case Ok(manifest):
                    match manifest.is_compatible():
                        case Ok(False):
                            logger.warning("ltr_model_incompatible_manifest",
                                path=str(model_path),
                                manifest_version=manifest.model_version,
                                current_version=ArtifactManifest._get_embedding_model_version())
                        case Ok(True):
                            logger.info("ltr_manifest_verified", metrics=manifest.metrics)
                        case Err(err):
                            logger.warning("ltr_manifest_compat_check_failed", error=str(err))
                case Err(err):
                    logger.warning("ltr_manifest_load_failed", error=str(err))

        try:
            data = joblib.load(model_path)
        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            return Err(ModelError(
                message=f"Ошибка загрузки LTR-модели: {e}",
                model_name="ltr_ranker_xgb_regressor",
            ))

        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.skill_metadata = data["skill_metadata"]
        self.skill_embeddings = data.get("skill_embeddings", {})
        self.total_vacancies = data["total_vacancies"]
        self.vacancy_skills_corpus = data.get("vacancy_skills_corpus", [])
        self.category_avg_weight = data.get("category_avg_weight", {})
        self.is_fitted = True
        logger.info(
            "model_loaded",
            path=str(model_path),
            features=len(self.feature_names),
            skills=len(self.skill_metadata),
            corpus_size=len(self.vacancy_skills_corpus),
        )
        return Ok(self)


# ------------------------------------------------------------------
# CLI для отладки
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Отладка LTRRecommendationEngine")
    parser.add_argument(
        "--load-raw",
        action="store_true",
        help="Загрузить вакансии из data/raw/hh_vacancies_basic.json",
    )
    parser.add_argument("--train", action="store_true", help="Принудительно обучить модель")
    parser.add_argument("--student", type=str, default="base", help="Профиль студента")
    args = parser.parse_args()

    engine = LTRRecommendationEngine()

    if args.train or not engine.model_path.exists():
        if args.load_raw:
            raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
            if not raw_file.exists():
                logger.error("raw_file_not_found", path=str(raw_file))
                sys.exit(1)
            with open(raw_file, encoding="utf-8") as f:
                vacancies = json.load(f)
            logger.info("raw_vacancies_loaded", count=len(vacancies))
        else:
            logger.warning("using_synthetic_vacancies_for_debug")
            vacancies = [
                {
                    "key_skills": [{"name": "python"}, {"name": "sql"}],
                    "experience": "middle",
                    "description": "Python backend dev",
                },
                {
                    "key_skills": [{"name": "python"}, {"name": "docker"}],
                    "experience": "senior",
                    "description": "Senior Python dev",
                },
            ]
        match engine.fit(vacancies):
            case Ok(_):
                logger.info("ltr_training_successful")
            case Err(err):
                logger.error("ltr_training_failed", error=str(err))
                sys.exit(1)
    else:
        match engine.load_model():
            case Ok(_):
                logger.info("ltr_model_loaded_successfully")
            case Err(err):
                logger.error("ltr_model_load_failed", error=str(err))
                sys.exit(1)

    student_file = config.STUDENTS_DIR / f"{args.student}_competency.json"
    student_skills: list[str] = []
    if student_file.exists():
        with open(student_file, encoding="utf-8") as f:
            data = json.load(f)
            student_skills = data.get("навыки", [])
    else:
        student_skills = ["python", "sql"]

    market_skills = list(engine.skill_metadata.keys())
    missing = [s for s in market_skills if s not in student_skills]

    print("\n" + "=" * 60)
    print(f"LTR-рекомендации для профиля '{args.student}':")
    print("=" * 60)
    recs, shap_vals, X_out = engine.predict_skill_impact_with_shap(student_skills, missing)
    for skill, score, expl in recs[:10]:
        print(f"• {skill:<20} важность: {score:>5.1f}% | {expl}")
