# tests/analyzers/test_comparator.py
import pytest
import logging
import traceback
import sys
import numpy as np
from unittest.mock import patch, MagicMock
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.embedding_comparator import EmbeddingComparator, normalize_skills

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)
logger = logging.getLogger("test_comparator")


def test_normalize_skills():
    raw = ["  Python  ", "Java-Script", "C++"]
    norm = normalize_skills(raw)
    assert norm == ["python", "java script", "c++"]


class TestCompetencyComparatorExtended:
    def test_fit_market_empty_data(self):
        comparator = CompetencyComparator(use_embeddings=False)
        assert not comparator.fit_market([])

    def test_compare_not_fitted_raises(self):
        comparator = CompetencyComparator(use_embeddings=False)
        with pytest.raises(ValueError, match="Сначала вызови fit_market"):
            comparator.compare(["python"])

    def test_compare_embeddings_mode_mocked(self):
        comparator = CompetencyComparator(use_embeddings=True)
        comparator.fitted = True
        mock_emb = MagicMock()
        mock_emb.compare_student_to_market.return_value = {
            "avg_similarity": 0.8,
            "matches": [{"skill": "python", "score": 0.9}]
        }
        comparator.embedding_comparator = mock_emb
        score, confidence = comparator.compare(["python"])
        assert score == 0.8
        assert 0 <= confidence <= 1

    def test_tfidf_mode_fallback(self):
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["java", "spring"]]
        comparator.fit_market(corpus)
        score, confidence = comparator.compare(["python"])
        assert score == 0.5
        assert confidence == 0.7

    def test_get_stats(self):
        comparator = CompetencyComparator(use_embeddings=True, level="senior")
        stats = comparator.get_stats()
        assert stats["mode"] == "embeddings"
        assert stats["level"] == "senior"
        assert stats["status"] == "not_fitted"
        comparator.fitted = True
        stats = comparator.get_stats()
        assert stats["status"] == "ready"


@pytest.mark.parametrize("use_embeddings,level", [
    (False, "middle"),
    (True, "junior"),
    (True, "middle"),
    (True, "senior")
])
def test_comparator_both_modes(use_embeddings, level):
    """Тест обоих режимов с подробным логированием и отловом ошибок"""
    logger.info("=" * 80)
    logger.info(f"▶️  Запуск теста | embeddings={use_embeddings} | level={level}")
    logger.info("=" * 80)

    try:
        logger.info("Создаём CompetencyComparator...")
        comparator = CompetencyComparator(
            use_embeddings=use_embeddings,
            level=level
        )
        logger.info("✅ Comparator создан")

        vacancies_skills = [
            ["python", "sql", "pandas", "machine learning"],
            ["python", "pytorch", "ml", "rest api"],
            ["sql", "django", "fastapi"]
        ]
        student_skills = ["python", "sql", "pytorch", "ml"]

        logger.info("Вызываем fit_market()...")
        success = comparator.fit_market(vacancies_skills)
        assert success is True, "fit_market вернул False"
        logger.info("✅ fit_market прошёл успешно")

        logger.info("Вызываем compare()...")
        score, confidence = comparator.compare(student_skills)
        logger.info(f"Результат: score={score:.4f} | confidence={confidence:.4f}")

        assert 0.0 <= score <= 1.0, f"Некорректный score: {score}"
        assert 0.0 <= confidence <= 1.0, f"Некорректный confidence: {confidence}"
        assert score > 0.3, f"Слишком низкий score: {score} (ожидали >0.3)"

        stats = comparator.get_stats()
        logger.info(f"Статистика: {stats}")

        logger.info(f"✅ ТЕСТ ПРОШЁЛ УСПЕШНО для {level} (embeddings={use_embeddings})")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("❌ ОШИБКА В ТЕСТЕ")
        logger.error(f"Тип: {type(e).__name__}")
        logger.error(f"Сообщение: {e}")
        logger.error("-" * 60)
        traceback.print_exc(file=sys.stderr)
        logger.error("-" * 60)
        raise


class TestEmbeddingComparatorExtended:
    def test_build_market_index_creates_cache(self, tmp_path):
        comparator = EmbeddingComparator(cache_dir=str(tmp_path))
        skills = ["python", "sql", "docker"]
        with patch.object(comparator, 'embed_skills', return_value=np.random.rand(3, 384)):
            comparator.build_market_index(skills, level="junior")
            cache_path = comparator._get_cache_path("market_embeddings", "junior")
            assert cache_path.exists()
            with patch('joblib.load') as mock_load:
                mock_load.return_value = (None, None)
                comparator2 = EmbeddingComparator(cache_dir=str(tmp_path))
                comparator2.build_market_index([], level="junior")
                mock_load.assert_called_once()

    def test_compare_student_to_market_without_index(self):
        comparator = EmbeddingComparator()
        with pytest.raises(ValueError, match="Сначала вызови build_market_index"):
            comparator.compare_student_to_market(["python"])

    def test_compare_student_to_market_results(self):
        comparator = EmbeddingComparator()
        comparator.market_skills = ["python", "java", "sql"]
        comparator.market_embeddings = np.random.rand(3, 384)
        student_skills = ["python", "pandas"]
        with patch.object(comparator, 'embed_skills', return_value=np.random.rand(2, 384)):
            with patch('src.analyzers.embedding_comparator.cosine_similarity',
                       return_value=np.array([[0.9, 0.2, 0.5]])):
                result = comparator.compare_student_to_market(student_skills)
                assert "matches" in result
                assert "missing" in result
                assert result["avg_similarity"] == pytest.approx(0.5333333, rel=1e-6)