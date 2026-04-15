"""
Embedding Comparator с поддержкой уровней опыта и FAISS (опционально)
"""
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import joblib
from pathlib import Path
from src import config
from src.parsing.embedding_loader import get_embedding_model

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def normalize_skills(skills: List[str]) -> List[str]:
    normalized = []
    for skill in skills:
        if not skill:
            continue
        s = skill.lower().strip()
        s = s.replace("'", "").replace('"', "").replace(":", "").replace("-", " ")
        s = " ".join(s.split())
        normalized.append(s)
    return normalized


class EmbeddingComparator:
    def __init__(self, model_name: str = None, cache_dir: str = None, similarity_threshold: float = 0.75, use_faiss: bool = True):
        self.model = get_embedding_model(model_name)
        if cache_dir is None:
            self.cache_dir = config.DATA_EMBEDDINGS_DIR
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.market_embeddings = None
        self.market_skills = None
        self.index = None

        if self.use_faiss:
            logger.info("✅ FAISS доступен, будет использоваться для быстрого поиска")
        else:
            logger.info("ℹ️ FAISS не установлен, используется sklearn cosine_similarity")

    def _get_cache_path(self, name: str, level: str = "middle") -> Path:
        return self.cache_dir / f"{name}_{level}.pkl"

    def embed_skills(self, skills: List[str]) -> np.ndarray:
        normalized = normalize_skills(skills)
        return self.model.encode(normalized, convert_to_numpy=True, show_progress_bar=False)

    def build_market_index(self, all_market_skills: List[str], level: str = "middle"):
        cache_path = self._get_cache_path("market_embeddings", level)
        if cache_path.exists():
            loaded = joblib.load(cache_path)
            if isinstance(loaded, dict):
                self.market_embeddings = loaded["embeddings"]
                self.market_skills = loaded["skills"]
            else:
                self.market_embeddings, self.market_skills = loaded
            logger.info(f"✅ Загружен кэш embeddings для {level}")

            if self.use_faiss:
                self._build_faiss_index()
            return

        self.market_skills = normalize_skills(all_market_skills)
        self.market_embeddings = self.embed_skills(self.market_skills)
        joblib.dump(
            {"embeddings": self.market_embeddings, "skills": self.market_skills},
            cache_path
        )
        logger.info(f"✅ Market embeddings сохранены для level={level}")

        if self.use_faiss:
            self._build_faiss_index()

    def _build_faiss_index(self):
        if self.market_embeddings is None:
            return
        dim = self.market_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.market_embeddings)
        self.index.add(self.market_embeddings)
        logger.info("✅ FAISS индекс построен")

    def compare_student_to_market(self, student_skills: List[str]) -> Dict:
        if self.market_embeddings is None:
            raise ValueError("Сначала вызови build_market_index()")

        student_emb = self.embed_skills(student_skills)
        faiss.normalize_L2(student_emb)

        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(student_emb, len(self.market_skills))
            similarities = scores[0]
            top_indices = indices[0]
            sorted_pairs = sorted(zip(top_indices, similarities), key=lambda x: x[1], reverse=True)
        else:
            similarities = cosine_similarity(student_emb, self.market_embeddings)[0]
            sorted_pairs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

        matches = []
        missing = []

        for idx, sim in sorted_pairs:
            skill = self.market_skills[idx]
            if sim >= self.similarity_threshold:
                matches.append({"skill": skill, "score": float(sim)})
            else:
                missing.append({"skill": skill, "score": float(sim)})

        return {
            "matches": matches,
            "missing": missing[:20],
            "avg_similarity": float(np.mean(similarities))
        }
        
    # embedding_comparator.py (добавить в класс EmbeddingComparator)

    def get_vacancy_embedding(self, skills: List[str]) -> np.ndarray:
        """Средний эмбеддинг навыков вакансии."""
        if not skills:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        embs = self.embed_skills(skills)
        return np.mean(embs, axis=0)

    def find_closest_vacancies(
        self,
        student_skills: List[str],
        vacancies: List[Dict],
        level: str = "middle",
        top_k: int = 50
    ) -> List[Dict]:
        """
        Возвращает top_k вакансий нужного уровня, наиболее близких к студенту.
        """
        student_emb = self.embed_skills(student_skills)
        student_emb = np.mean(student_emb, axis=0) if len(student_emb) > 0 else np.zeros(self.model.get_sentence_embedding_dimension())

        # Фильтруем вакансии по уровню
        level_vacancies = [v for v in vacancies if v.get('experience') == level]
        if not level_vacancies:
            level_vacancies = vacancies  # fallback на все, если пусто

        vac_embs = []
        for vac in level_vacancies:
            vac_skills = vac.get('skills', [])
            emb = self.get_vacancy_embedding(vac_skills)
            vac_embs.append(emb)

        if not vac_embs:
            return []

        vac_embs = np.vstack(vac_embs)
        similarities = cosine_similarity([student_emb], vac_embs)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [level_vacancies[i] for i in top_indices]