"""P2: полнотекстовый IDF-корпус. Мультисловные навыки весуются,
сэмпл детерминирован (random seed=7, не stride)."""

from unittest.mock import patch

from src import Ok
from src.parsing.skills.bm25_ranker import BM25Ranker


def identity_normalize(x):
    return Ok(x.lower() if isinstance(x, str) else x)


def test_multiword_skill_gets_weight_real_bm25():
    """Дефект P2: get_scores(['machine learning']) по однотокенным докам
    не матчился никогда → мультисловные навыки системно без веса."""
    ranker = BM25Ranker()
    # Три дока: одно-доковый корпус даёт отрицательный IDF в rank_bm25
    # (артефакт формулы, не прод — там корпус 50k).
    vacancies = [
        {"description": "machine learning and deep learning Engineer", "key_skills": [], "snippet": {}},
        {"description": "python developer with sql experience", "key_skills": [], "snippet": {}},
        {"description": "java backend developer rest api", "key_skills": [], "snippet": {}},
    ]
    with patch("src.parsing.skills.bm25_ranker.load_it_skills",
               return_value={"python", "machine learning", "deep learning"}), \
         patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize",
               side_effect=identity_normalize):
        weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok()
    w = weights.unwrap()
    assert "machine learning" in w, f"multi-word skill lost weight: {sorted(w)}"


def test_absent_skill_gets_no_weight_real_bm25():
    ranker = BM25Ranker()
    vacancies = [{"description": "python sql", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.load_it_skills",
               return_value={"python", "sql", "machine learning"}), \
         patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize",
               side_effect=identity_normalize):
        weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok()
    w = weights.unwrap()
    assert "machine learning" not in w


def test_sampling_deterministic():
    ranker = BM25Ranker()
    vacancies = [{"description": f"python sql skill {i}", "key_skills": [], "snippet": {}}
                 for i in range(500)]
    with patch("src.parsing.skills.bm25_ranker.load_it_skills",
               return_value={"python", "sql"}), \
         patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize",
               side_effect=identity_normalize):
        w1 = ranker.calculate_weights(vacancies)
    ranker2 = BM25Ranker()
    with patch("src.parsing.skills.bm25_ranker.load_it_skills",
               return_value={"python", "sql"}), \
         patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize",
               side_effect=identity_normalize):
        w2 = ranker2.calculate_weights(vacancies)
    assert w1 == w2
