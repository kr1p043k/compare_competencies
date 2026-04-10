from typing import List, Dict, Tuple, Optional
import logging

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
        level: str = "middle"
    ):
        self.use_embeddings = use_embeddings
        self.level = level
        self.tfidf = None
        self.embedding_comparator: Optional[EmbeddingComparator] = None
        self.fitted = False

        if use_embeddings:
            self.embedding_comparator = EmbeddingComparator()
            logger.info(f"✅ EmbeddingComparator инициализирован для уровня {level}")
        else:
            self.tfidf = TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                token_pattern=r"(?u)\b\w[\w-]*\b"
            )
            logger.info("✅ TF-IDF Comparator инициализирован (legacy)")

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
        return True

    def compare(self, student_skills: List[str]) -> Tuple[float, float]:
        if not self.fitted:
            raise ValueError("Сначала вызови fit_market()")

        if self.use_embeddings and self.embedding_comparator:
            result = self.embedding_comparator.compare_student_to_market(student_skills)
            score = result["avg_similarity"]
            confidence = min(1.0, len(result.get("matches", [])) / max(1, len(student_skills) + 5))
            return score, confidence
        else:
            if not student_skills:
                return 0.0, 0.0
            student_text = " ".join(student_skills)
            student_vec = self.tfidf.transform([student_text])
            # legacy fallback
            return 0.5, 0.7  # заглушка, не используется при embeddings
    def get_skill_weights(self) -> Dict[str, float]:
        """Compatibility stub"""
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