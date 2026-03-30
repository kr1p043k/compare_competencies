"""
Embedding Comparator с поддержкой уровней опыта
"""
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_skills(skills: List[str]) -> List[str]:
    """Простая, но рабочая нормализация навыков (можно потом вынести в skill_normalizer.py)"""
    normalized = []
    for skill in skills:
        if not skill:
            continue
        s = skill.lower().strip()
        s = s.replace("'", "").replace('"', "").replace(":", "").replace("-", " ")
        # Убираем лишние пробелы
        s = " ".join(s.split())
        normalized.append(s)
    return normalized


class EmbeddingComparator:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", cache_dir: str = "data/embeddings"):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.market_embeddings = None
        self.market_skills = None

    def _get_cache_path(self, name: str, level: str = "middle") -> Path:
        return self.cache_dir / f"{name}_{level}.pkl"

    def embed_skills(self, skills: List[str]) -> np.ndarray:
        """Нормализуем + эмбедим"""
        normalized = normalize_skills(skills)
        return self.model.encode(normalized, convert_to_numpy=True, show_progress_bar=False)

    def build_market_index(self, all_market_skills: List[str], level: str = "middle"):
        """Один раз строим эмбеддинги рынка для конкретного уровня"""
        cache_path = self._get_cache_path("market_embeddings", level)
        if cache_path.exists():
            self.market_embeddings, self.market_skills = joblib.load(cache_path)
            logger.info(f"✅ Загружен кэш embeddings для {level}")
            return

        self.market_skills = normalize_skills(all_market_skills)
        self.market_embeddings = self.embed_skills(self.market_skills)
        joblib.dump((self.market_embeddings, self.market_skills), cache_path)
        logger.info(f"✅ Market embeddings сохранены для level={level}")

    def compare_student_to_market(self, student_skills: List[str]) -> Dict:
        """Основная функция сравнения"""
        if self.market_embeddings is None:
            raise ValueError("Сначала вызови build_market_index()")

        student_emb = self.embed_skills(student_skills)
        
        similarities = cosine_similarity(student_emb, self.market_embeddings)[0]
        
        matches = []
        missing = []
        
        for idx, sim in enumerate(similarities):
            skill = self.market_skills[idx]
            if sim > 0.75:
                matches.append({"skill": skill, "score": float(sim)})
            else:
                missing.append({"skill": skill, "score": float(sim)})

        return {
            "matches": sorted(matches, key=lambda x: x["score"], reverse=True),
            "missing": sorted(missing, key=lambda x: x["score"], reverse=True)[:20],
            "avg_similarity": float(np.mean(similarities))
        }