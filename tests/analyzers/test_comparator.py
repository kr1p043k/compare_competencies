# tests/analyzers/test_comparator.py
import pytest
import logging
import traceback
import sys
import numpy as np
from unittest.mock import patch, MagicMock
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.embedding_comparator import EmbeddingComparator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)
logger = logging.getLogger("test_comparator")


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
        # Мокаем embedding_comparator полностью
        mock_emb = MagicMock()
        comparator.embedding_comparator = mock_emb
        # Устанавливаем веса, чтобы попасть в ветку weighted_coverage
        comparator.skill_weights = {"python": 0.9}
        # Мокаем weighted_coverage
        mock_emb.compare_student_to_market = None  # не должен вызываться
        with patch.object(comparator, 'weighted_coverage', return_value=0.8):
            score, confidence = comparator.compare(["python"])
        assert score == 0.8
        assert 0 <= confidence <= 1

    def test_tfidf_mode_fallback(self):
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["java", "spring"]]
        comparator.fit_market(corpus)
        score, confidence = comparator.compare(["python"])
        assert 0.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_get_stats(self):
        comparator = CompetencyComparator(use_embeddings=True, level="senior")
        stats = comparator.get_stats()
        assert stats["mode"] == "embeddings"
        assert stats["level"] == "senior"
        assert stats["status"] == "not_fitted"
        comparator.fitted = True
        stats = comparator.get_stats()
        assert stats["status"] == "ready"

    def test_set_skill_weights(self):
        comparator = CompetencyComparator(use_embeddings=True)
        weights = {"python": 0.9, "sql": 0.7}
        comparator.set_skill_weights(weights)
        assert comparator.skill_weights == weights
        if comparator.embedding_comparator:
            assert comparator.embedding_comparator.skill_weights == weights

    def test_compare_with_weights_no_embeddings(self):
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["python", "docker"]]
        comparator.fit_market(corpus)
        comparator.set_skill_weights({"python": 0.9, "sql": 0.5, "docker": 0.3})
        score, confidence = comparator.compare(["python"])
        assert 0.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_weighted_coverage_basic(self):
        """Прямой тест weighted_coverage"""
        comparator = CompetencyComparator(use_embeddings=False)
        student_skills = ["python", "sql"]
        weights = {"python": 0.9, "sql": 0.5, "docker": 0.3}
        # Без эмбеддингов — точное совпадение
        coverage = comparator.weighted_coverage(student_skills, weights)
        assert coverage > 0


@pytest.mark.parametrize("use_embeddings,level", [
    (False, "middle"),
    (True, "junior"),
    (True, "middle"),
    (True, "senior"),
])
def test_comparator_tfidf_mode(use_embeddings, level):
    """Тест TF-IDF и Embeddings режимов (embeddings через глобальный мок)"""
    logger.info("=" * 80)
    logger.info(f">>>>> Запуск теста | embeddings={use_embeddings} | level={level}")
    logger.info("=" * 80)

    try:
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

        stats = comparator.get_stats()
        logger.info(f"Статистика: {stats}")

        logger.info(f"✅ ТЕСТ ПРОШЁЛ УСПЕШНО")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("❌ ОШИБКА В ТЕСТЕ")
        logger.error(f"Тип: {type(e).__name__}")
        logger.error(f"Сообщение: {e}")
        traceback.print_exc(file=sys.stderr)
        raise

class TestEmbeddingComparatorExtended:
    def test_build_market_index_creates_cache(self, tmp_path):
        comparator = EmbeddingComparator(cache_dir=str(tmp_path), use_faiss=False)
        skills = ["python", "java"]
        comparator.build_market_index(skills, level="middle")
        cache_path = comparator._get_cache_path("market_embeddings", "middle")
        assert cache_path.exists()

    def test_compare_student_to_market_without_index(self):
        comparator = EmbeddingComparator()
        with pytest.raises(ValueError, match="Сначала вызови build_market_index"):
            comparator.compare_student_to_market(["python"])

    def test_compare_student_to_market_results(self):
        comparator = EmbeddingComparator(use_faiss=False)
        market_skills = ["python", "java", "c++"]
        comparator.build_market_index(market_skills, level="middle")
        student_skills = ["python", "c#"]
        result = comparator.compare_student_to_market(student_skills)
        assert "matches" in result
        assert "missing" in result
        assert result["score"] >= 0

    def test_embed_skills_empty(self):
        comparator = EmbeddingComparator()
        result = comparator.embed_skills([])
        assert result.shape[1] == comparator.model.get_sentence_embedding_dimension()

    def test_embed_skills_returns_correct_shape(self):
        comparator = EmbeddingComparator()
        skills = ["python", "java"]
        result = comparator.embed_skills(skills)
        assert result.shape[0] == 2
        assert result.shape[1] == comparator.model.get_sentence_embedding_dimension()

class TestEmbeddingComparatorFull:
    @pytest.fixture
    def comparator(self):
        return EmbeddingComparator(use_faiss=False)

    def test_init_with_faiss(self):
        """Покрытие строк 22-23 (FAISS available)"""
        comp = EmbeddingComparator(use_faiss=True)
        assert comp.use_faiss is True

    def test_init_without_faiss(self):
        comp = EmbeddingComparator(use_faiss=False)
        assert comp.use_faiss is False

    def test_get_vacancy_embedding_empty_skills(self, comparator):
        """Строка 71: пустые навыки → нулевой вектор"""
        emb = comparator.get_vacancy_embedding([])
        assert emb.shape[0] == comparator.model.get_sentence_embedding_dimension()
        assert np.allclose(emb, 0)

    def test_get_vacancy_embedding(self, comparator):
        """Строка 71-73: усреднение эмбеддингов"""
        skills = ["python", "sql"]
        emb = comparator.get_vacancy_embedding(skills)
        assert emb.shape[0] == comparator.model.get_sentence_embedding_dimension()

    def test_build_market_index_with_faiss(self, tmp_path):
        """Строка 87, 91: FAISS индекс"""
        comp = EmbeddingComparator(cache_dir=str(tmp_path), use_faiss=True)
        skills = ["python", "java", "c++"]
        comp.build_market_index(skills, level="test")
        assert comp.index is not None

    def test_compare_student_to_market_empty_skills(self, comparator):
        """Строка 105: пустые навыки студента"""
        comparator.build_market_index(["python", "java"], level="test")
        result = comparator.compare_student_to_market([])
        assert result["score"] == 0.0
        assert result["weighted_coverage"] == 0.0

    def test_compare_student_to_market_no_weights(self, comparator):
        """Строка 136: skill_weights пусты — fallback"""
        comparator.build_market_index(["python", "java", "c++"], level="test")
        comparator.skill_weights = {}
        result = comparator.compare_student_to_market(["python"])
        assert result["score"] >= 0

    def test_compare_student_to_market_with_weights(self, comparator):
        """Строка 140-144: с весами навыков"""
        comparator.build_market_index(["python", "java", "c++"], level="test")
        comparator.skill_weights = {"python": 0.9, "java": 0.7, "c++": 0.5}
        result = comparator.compare_student_to_market(["python"])
        assert result["score"] >= 0

    def test_find_closest_vacancies_empty(self, comparator):
        """Строка 168-171: нет вакансий нужного уровня"""
        student_skills = ["python"]
        vacancies = []
        result = comparator.find_closest_vacancies(student_skills, vacancies)
        assert result == []

    def test_find_closest_vacancies(self, comparator):
        """Строка 168-202: поиск ближайших вакансий"""
        student_skills = ["python"]
        vacancies = [
            {"skills": ["python", "sql"], "experience": "middle"},
            {"skills": ["java", "spring"], "experience": "middle"},
            {"skills": ["python", "docker"], "experience": "senior"},
        ]
        result = comparator.find_closest_vacancies(
            student_skills, vacancies, level="middle", top_k=2
        )
        assert len(result) > 0

    def test_find_closest_vacancies_fallback_level(self, comparator):
        """Строка 180-181: fallback когда нет вакансий нужного уровня"""
        student_skills = ["python"]
        vacancies = [
            {"skills": ["python", "sql"], "experience": "senior"},
        ]
        result = comparator.find_closest_vacancies(
            student_skills, vacancies, level="middle", top_k=2
        )
        assert len(result) > 0

    def test_set_clusterer(self, comparator):
        """Строка 205-206: установка кластеризатора"""
        mock_clusterer = MagicMock()
        comparator.set_clusterer(mock_clusterer, [{"skills": ["python"]}])
        assert comparator.clusterer == mock_clusterer
        assert comparator.vacancies_data == [{"skills": ["python"]}]

    def test_compare_to_clusters_no_clusterer(self, comparator):
        """Строка 213: clusterer не установлен"""
        result = comparator.compare_to_clusters(["python"])
        assert "error" in result

    def test_compare_to_clusters_not_fitted(self, comparator):
        """Строка 213: clusterer не обучен"""
        mock_clusterer = MagicMock()
        mock_clusterer.is_fitted = False
        comparator.set_clusterer(mock_clusterer, [{"skills": ["python"]}])
        result = comparator.compare_to_clusters(["python"])
        assert "error" in result

    def test_compare_to_clusters(self, comparator):
        """Строка 213-228: работа с кластерами"""
        mock_clusterer = MagicMock()
        mock_clusterer.is_fitted = True
        mock_clusterer.find_closest_clusters.return_value = [(0, 0.9), (1, 0.7)]
        mock_clusterer.get_cluster_skills.return_value = ["python", "sql", "docker"]

        comparator.set_clusterer(mock_clusterer, [{"skills": ["python"]}])
        result = comparator.compare_to_clusters(["python"], top_k=2)
        assert "clusters" in result
        assert len(result["clusters"]) == 2

    def test_hybrid_compare(self, comparator):
        """Строка 236-253: гибридное сравнение"""
        comparator.build_market_index(["python", "java", "c++", "sql"], level="test")

        mock_clusterer = MagicMock()
        mock_clusterer.is_fitted = True
        mock_clusterer.find_closest_clusters.return_value = [(0, 0.9)]
        mock_clusterer.get_cluster_skills.return_value = ["python", "sql", "docker"]

        comparator.set_clusterer(mock_clusterer, [{"skills": ["python"]}])
        result = comparator.hybrid_compare(["python", "sql"], {"python": 0.9})
        assert "global_score" in result
        assert "cluster_score" in result
        assert "hybrid_score" in result
        assert 0.0 <= result["hybrid_score"] <= 1.0

    def test_hybrid_compare_no_clusters(self, comparator):
        """Строка 236-253: без кластеров — hybrid_score = global_score"""
        comparator.build_market_index(["python", "java"], level="test")
        result = comparator.hybrid_compare(["python"], {"python": 0.9})
        assert result["cluster_score"] is None
        assert result["hybrid_score"] == result["global_score"]

    def test_faiss_available_check(self, monkeypatch):
        """Строки 22-23: проверка доступности FAISS"""
        # FAISS уже импортирован, проверяем флаг
        from src.analyzers.embedding_comparator import FAISS_AVAILABLE
        assert isinstance(FAISS_AVAILABLE, bool)

    def test_find_closest_vacancies_no_level_match_no_fallback(self, comparator):
        """Строка 182: все вакансии отфильтрованы по уровню, fallback отключен"""
        student_skills = ["python"]
        vacancies = [
            {"skills": ["python", "sql"], "experience": "senior"},
            {"skills": ["java", "spring"], "experience": "senior"},
        ]
        # Указан уровень junior, но все вакансии senior → filter пуст → fallback на все
        result = comparator.find_closest_vacancies(
            student_skills, vacancies, level="junior", top_k=2
        )
        assert len(result) == 2  # fallback возвращает все

class TestCompetencyComparatorFull:
    def test_compare_embeddings_no_weights_no_matches(self):
        """Строки 81-90: embeddings без весов"""
        comparator = CompetencyComparator(use_embeddings=True)
        comparator.fitted = True
        mock_emb = MagicMock()
        mock_emb.compare_student_to_market.return_value = {
            "score": 0.5,
            "weighted_coverage": 0.5,
            "matches": [
                {"skill": "python", "similarity": 0.3},
                {"skill": "java", "similarity": 0.2}
            ]
        }
        comparator.embedding_comparator = mock_emb
        score, confidence = comparator.compare(["python"])
        # confidence = len(matches with sim >= 0.65) / len(student_skills) = 0/1 = 0
        assert score == 0.5
        assert confidence == 0.0

    def test_compare_embeddings_no_student_skills(self):
        """Строка 110: пустые навыки студента в TF-IDF"""
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["java", "spring"]]
        comparator.fit_market(corpus)
        score, confidence = comparator.compare([])
        assert score == 0.0
        assert confidence == 0.0

    def test_compare_tfidf_with_cached_matrix(self):
        """Строка 115: использование кэшированной матрицы"""
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["java", "spring"]]
        comparator.fit_market(corpus)
        # Первый вызов создаст _market_tfidf_matrix
        score1, _ = comparator.compare(["python"])
        # Второй вызов должен использовать кэш
        assert hasattr(comparator, '_market_tfidf_matrix')
        score2, _ = comparator.compare(["python"])
        assert score1 == score2

    def test_fit_market_sets_fitted_flag(self):
        """Проверка что fit_market устанавливает флаг fitted"""
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["java", "spring"]]
        assert comparator.fit_market(corpus) is True
        assert comparator.fitted is True

    def test_weighted_coverage_hybrid(self):
        """Строка 139-158: weighted_coverage с эмбеддингами"""
        comparator = CompetencyComparator(use_embeddings=True)
        mock_emb = MagicMock()
        # Мокаем embed_skills чтобы возвращать эмбеддинги
        mock_emb.embed_skills.return_value = np.random.rand(2, 384)
        comparator.embedding_comparator = mock_emb

        coverage = comparator.weighted_coverage(
            ["python", "sql"],
            {"python": 0.9, "sql": 0.7, "docker": 0.5},
            use_hybrid=True
        )
        assert 0.0 <= coverage <= 1.01

    def test_weighted_coverage_hybrid_empty_embeddings(self):
        """Строка 145: пустые эмбеддинги студента"""
        comparator = CompetencyComparator(use_embeddings=True)
        mock_emb = MagicMock()
        mock_emb.embed_skills.return_value = np.zeros((0, 384))
        comparator.embedding_comparator = mock_emb

        coverage = comparator.weighted_coverage([], {"python": 0.9})
        assert coverage == 0.0

    def test_weighted_coverage_hybrid_empty_market_skill_emb(self):
        """Строка 152: пустой эмбеддинг рыночного навыка"""
        comparator = CompetencyComparator(use_embeddings=True)
        mock_emb = MagicMock()

        def mock_embed(skills):
            if "nonexistent" in skills:
                return np.zeros((0, 384))
            return np.random.rand(len(skills), 384)

        mock_emb.embed_skills.side_effect = mock_embed
        comparator.embedding_comparator = mock_emb

        coverage = comparator.weighted_coverage(
            ["python"],
            {"nonexistent": 0.5}
        )
        assert coverage == 0.0

    def test_get_skill_weights_embeddings(self):
        """Строка 57-59: get_skill_weights в embedding режиме"""
        comparator = CompetencyComparator(use_embeddings=True)
        weights = comparator.get_skill_weights()
        assert weights == {}

    def test_get_skill_weights_tfidf(self):
        """Строка 59: get_skill_weights в TF-IDF режиме"""
        comparator = CompetencyComparator(use_embeddings=False)
        weights = comparator.get_skill_weights()
        assert weights == {}
    
    def test_get_skill_weights_returns_dict(self):
        """Строки 57-59: get_skill_weights возвращает словарь"""
        comparator = CompetencyComparator(use_embeddings=True)
        result = comparator.get_skill_weights()
        assert isinstance(result, dict)

    def test_weighted_coverage_hybrid_disabled(self):
        """Строка 141: use_hybrid=False с эмбеддингами"""
        comparator = CompetencyComparator(use_embeddings=True)
        mock_emb = MagicMock()
        mock_emb.embed_skills.return_value = np.random.rand(2, 384)
        comparator.embedding_comparator = mock_emb

        coverage = comparator.weighted_coverage(
            ["python", "sql"],
            {"python": 0.9, "sql": 0.7},
            use_hybrid=False  # не-hybrid режим
        )
        assert 0.0 <= coverage <= 1.01
        
    def test_weighted_coverage_exact_match_no_hybrid(self):
        """Строка 141: точное совпадение без эмбеддингов"""
        comparator = CompetencyComparator(use_embeddings=False)
        student_skills = ["python", "sql", "docker"]
        weights = {"python": 0.9, "sql": 0.5, "docker": 0.3, "k8s": 0.2}
        coverage = comparator.weighted_coverage(student_skills, weights, use_hybrid=False)
        expected = (0.9 + 0.5 + 0.3) / (0.9 + 0.5 + 0.3 + 0.2)
        assert coverage == pytest.approx(expected)