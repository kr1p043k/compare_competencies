from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from src.analyzers.embedding_comparator import EmbeddingComparator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class CompetencyComparator:
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        use_embeddings: bool = False,
        level: str = "middle",
        similarity_threshold: float = 0.5
    ):
        self.use_embeddings = use_embeddings
        self.level = level
        self.tfidf = None
        self.embedding_comparator: Optional[EmbeddingComparator] = None
        self.fitted = False
        self.skill_weights: Optional[Dict[str, float]] = None

        if use_embeddings:
            self.embedding_comparator = EmbeddingComparator(
                similarity_threshold=similarity_threshold
            )
        else:
            self.tfidf = TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                token_pattern=r"(?u)\b\w[\w-]*\b"
            )
            logger.info("✅ TF-IDF Comparator инициализирован (legacy)")

    def set_skill_weights(self, weights: Dict[str, float]) -> None:
        self.skill_weights = weights
        if self.embedding_comparator:
            self.embedding_comparator.skill_weights = weights
        logger.info(f"CompetencyComparator.set_skill_weights: получено {len(weights)} навыков")
        if weights:
            sample = list(weights.items())[:3]
            logger.info(f"   Примеры: {sample}")
        if self.embedding_comparator:
            logger.info("   Проброшено в embedding_comparator")

    def fit_market(self, vacancies_skills: List[List[str]]) -> bool:
        if not vacancies_skills:
            logger.warning("Нет данных для fit_market")
            return False

        if self.use_embeddings and self.embedding_comparator:
            all_skills = [skill for vac in vacancies_skills for skill in vac]
            self.embedding_comparator.build_market_index(all_skills, level=self.level)
            logger.info(f"✅ Market embeddings построены для {self.level} уровня")
        else:
            corpus = [" ".join(skills) for skills in vacancies_skills]
            self.tfidf.fit(corpus)
            logger.info("✅ TF-IDF fitted")

        self.fitted = True
        self._vacancies_skills = vacancies_skills
        return True

    def compare(self, student_skills: List[str]) -> Tuple[float, float]:
        if not self.fitted:
            raise ValueError("Сначала вызови fit_market()")

        if self.skill_weights is not None and len(self.skill_weights) > 0:
            score = self.weighted_coverage(student_skills, self.skill_weights)
            student_norm = [s.lower().strip() for s in student_skills]
            matched = sum(1 for s in student_norm if self.skill_weights.get(s, 0) > 0)
            confidence = matched / max(1, len(student_skills))
            return round(score, 4), round(confidence, 4)

        if self.use_embeddings and self.embedding_comparator:
            result = self.embedding_comparator.compare_student_to_market(student_skills)
            score = result.get("score", result.get("weighted_coverage", 0.0))
            matches = result.get("matches", [])
            confidence = (
                len([m for m in matches if m.get("similarity", 0) >= 0.65])
                / max(1, len(student_skills))
                if student_skills
                else 0.0
            )
            return round(score, 4), round(confidence, 4)

        else:
            if not student_skills:
                return 0.0, 0.0

            student_text = " ".join(student_skills)
            student_vec = self.tfidf.transform([student_text])

            if not hasattr(self, "_market_tfidf_matrix"):
                corpus = [" ".join(skills) for skills in self._vacancies_skills]
                self._market_tfidf_matrix = self.tfidf.transform(corpus)

            similarities = cosine_similarity(student_vec, self._market_tfidf_matrix)[0]
            score = float(np.mean(similarities)) if len(similarities) > 0 else 0.0
            confidence = float(np.max(similarities)) if len(similarities) > 0 else 0.0

            return round(score, 4), round(confidence, 4)

    def get_skill_weights(self) -> Dict[str, float]:
        if self.use_embeddings:
            logger.info("ℹ️  Embedding mode: get_skill_weights() не требуется")
            return {}
        else:
            logger.warning("get_skill_weights() для TF-IDF не реализован в текущей версии")
            return {}

    def get_stats(self) -> Dict:
        return {
            "status": "ready" if self.fitted else "not_fitted",
            "mode": "embeddings" if self.use_embeddings else "tfidf",
            "level": self.level
        }

    def weighted_coverage(
        self,
        student_skills: List[str],
        weights: Dict[str, float],
        use_hybrid: bool = True
    ) -> float:
        if not student_skills or not weights:
            return 0.0

        if not self.use_embeddings or self.embedding_comparator is None:
            student_set = {s.lower().strip() for s in student_skills}
            covered = sum(w for s, w in weights.items() if s.lower().strip() in student_set)
            total = sum(weights.values())
            return covered / total if total > 0 else 0.0

        student_embs = self.embedding_comparator.embed_skills(student_skills)
        if len(student_embs) == 0:
            return 0.0

        matched = 0.0
        total_weight = 0.0

        for skill, weight in weights.items():
            skill_emb = self.embedding_comparator.embed_skills([skill])
            if len(skill_emb) == 0:
                continue
            skill_emb = skill_emb[0]

            sims = cosine_similarity([skill_emb], student_embs)[0]
            sim = max(sims)

            matched += sim * weight
            total_weight += weight

        return matched / total_weight if total_weight > 0 else 0.0