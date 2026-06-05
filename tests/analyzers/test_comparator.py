# tests/analyzers/test_comparator.py
import logging
import sys
import traceback
from unittest.mock import MagicMock, patch
import joblib
import numpy as np
import pytest

from src import Ok
from src.analyzers.comparison.comparator import CompetencyComparator
from src.analyzers.comparison.embedding_comparator import EmbeddingComparator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)
logger = logging.getLogger("test_comparator")


class TestCompetencyComparatorExtended:
    def test_fit_market_empty_data(self):
        comparator = CompetencyComparator(use_embeddings=False)
        result = comparator.fit_market([])
        assert result.is_ok() and not result.unwrap()

    def test_compare_not_fitted_raises(self):
        comparator = CompetencyComparator(use_embeddings=False)
        result = comparator.compare(["python"])
        assert result.is_err()

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
        with patch.object(comparator, "weighted_coverage", return_value=0.8):
            score, confidence = comparator.compare(["python"]).unwrap()
        assert score == 0.8
        assert 0 <= confidence <= 1

    def test_tfidf_mode_fallback(self):
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["java", "spring"]]
        comparator.fit_market(corpus)
        score, confidence = comparator.compare(["python"]).unwrap()
        assert 0.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_get_stats(self):
        comparator = CompetencyComparator(use_embeddings=True, level="senior")
        stats = comparator.get_stats().unwrap()
        assert stats["mode"] == "embeddings"
        assert stats["level"] == "senior"
        assert stats["status"] == "not_fitted"
        comparator.fitted = True
        stats = comparator.get_stats().unwrap()
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
        score, confidence = comparator.compare(["python"]).unwrap()
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


@pytest.mark.parametrize(
    "use_embeddings,level",
    [
        (False, "middle"),
        (True, "junior"),
        (True, "middle"),
        (True, "senior"),
    ],
)
def test_comparator_tfidf_mode(use_embeddings, level):
    """Тест TF-IDF и Embeddings режимов (embeddings через глобальный мок)"""
    logger.info("=" * 80)
    logger.info(f">>>>> Запуск теста | embeddings={use_embeddings} | level={level}")
    logger.info("=" * 80)

    try:
        comparator = CompetencyComparator(use_embeddings=use_embeddings, level=level)
        logger.info("✅ Comparator создан")

        vacancies_skills = [
            ["python", "sql", "pandas", "machine learning"],
            ["python", "pytorch", "ml", "rest api"],
            ["sql", "django", "fastapi"],
        ]
        student_skills = ["python", "sql", "pytorch", "ml"]

        logger.info("Вызываем fit_market()...")
        success = comparator.fit_market(vacancies_skills)
        assert success.is_ok() and success.unwrap() is True, "fit_market вернул не Ok(True)"
        logger.info("✅ fit_market прошёл успешно")

        logger.info("Вызываем compare()...")
        score, confidence = comparator.compare(student_skills).unwrap()
        logger.info(f"Результат: score={score:.4f} | confidence={confidence:.4f}")

        assert 0.0 <= score <= 1.0, f"Некорректный score: {score}"
        assert 0.0 <= confidence <= 1.0, f"Некорректный confidence: {confidence}"

        stats = comparator.get_stats().unwrap()
        logger.info(f"Статистика: {stats}")

        logger.info("✅ ТЕСТ ПРОШЁЛ УСПЕШНО")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("❌ ОШИБКА В ТЕСТЕ")
        logger.error(f"Тип: {type(e).__name__}")
        logger.error(f"Сообщение: {e}")
        traceback.print_exc(file=sys.stderr)
        raise


class TestEmbeddingComparatorExtended:
    def test_build_market_index_creates_cache(self, tmp_path):
        comparator = EmbeddingComparator(cache_dir=str(tmp_path))
        skills = ["python", "java"]
        comparator.build_market_index(skills, level="middle")
        cache_path = comparator._get_cache_path("market_embeddings", "middle")
        assert cache_path.exists()

    def test_compare_student_to_market_without_index(self):
        comparator = EmbeddingComparator()
        result = comparator.compare_student_to_market(["python"])
        assert result.is_err()

    def test_compare_student_to_market_results(self):
        comparator = EmbeddingComparator()
        market_skills = ["python", "java", "c++"]
        comparator.build_market_index(market_skills, level="middle")
        student_skills = ["python", "c#"]
        result = comparator.compare_student_to_market(student_skills).unwrap()
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
        assert result.ndim == 2
        assert result.shape[1] == comparator.model.get_sentence_embedding_dimension()


class TestEmbeddingComparatorFull:
    @pytest.fixture
    def comparator(self):
        return EmbeddingComparator()

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

    def test_build_market_index_cache_created(self, tmp_path):
        comp = EmbeddingComparator(cache_dir=str(tmp_path))
        skills = ["python", "java", "c++"]
        comp.build_market_index(skills, level="test")
        assert comp.market_skills == skills

    def test_compare_student_to_market_empty_skills(self, comparator):
        """Строка 105: пустые навыки студента"""
        comparator.build_market_index(["python", "java"], level="test")
        result = comparator.compare_student_to_market([]).unwrap()
        assert result["score"] == 0.0
        assert result["weighted_coverage"] == 0.0

    def test_compare_student_to_market_no_weights(self, comparator):
        """Строка 136: skill_weights пусты — fallback"""
        comparator.build_market_index(["python", "java", "c++"], level="test")
        comparator.skill_weights = {}
        result = comparator.compare_student_to_market(["python"]).unwrap()
        assert result["score"] >= 0

    def test_compare_student_to_market_with_weights(self, comparator):
        """Строка 140-144: с весами навыков"""
        comparator.build_market_index(["python", "java", "c++"], level="test")
        comparator.skill_weights = {"python": 0.9, "java": 0.7, "c++": 0.5}
        result = comparator.compare_student_to_market(["python"]).unwrap()
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
        result = comparator.find_closest_vacancies(student_skills, vacancies, level="middle", top_k=2)
        assert len(result) > 0

    def test_find_closest_vacancies_fallback_level(self, comparator):
        """Строка 180-181: fallback когда нет вакансий нужного уровня"""
        student_skills = ["python"]
        vacancies = [
            {"skills": ["python", "sql"], "experience": "senior"},
        ]
        result = comparator.find_closest_vacancies(student_skills, vacancies, level="middle", top_k=2)
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

    def test_find_closest_vacancies_no_level_match_no_fallback(self, comparator):
        """Строка 182: все вакансии отфильтрованы по уровню, fallback отключен"""
        student_skills = ["python"]
        vacancies = [
            {"skills": ["python", "sql"], "experience": "senior"},
            {"skills": ["java", "spring"], "experience": "senior"},
        ]
        # Указан уровень junior, но все вакансии senior → filter пуст → fallback на все
        result = comparator.find_closest_vacancies(student_skills, vacancies, level="junior", top_k=2)
        assert len(result) == 2  # fallback возвращает все


class TestCompetencyComparatorFull:
    def test_compare_embeddings_no_weights_no_matches(self):
        """Строки 81-90: embeddings без весов"""
        comparator = CompetencyComparator(use_embeddings=True)
        comparator.fitted = True
        mock_emb = MagicMock()
        mock_emb.compare_student_to_market_ensemble.return_value = {
            "score": 0.5,
            "weighted_coverage": 0.5,
            "matches": [{"skill": "python", "similarity": 0.3}, {"skill": "java", "similarity": 0.2}],
        }
        comparator.embedding_comparator = mock_emb
        score, confidence = comparator.compare(["python"]).unwrap()
        # confidence = len(matches with sim >= 0.65) / len(student_skills) = 0/1 = 0
        assert score == 0.5
        assert confidence == 0.0

    def test_compare_embeddings_no_student_skills(self):
        """Строка 110: пустые навыки студента в TF-IDF"""
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["java", "spring"]]
        comparator.fit_market(corpus)
        score, confidence = comparator.compare([]).unwrap()
        assert score == 0.0
        assert confidence == 0.0

    def test_compare_tfidf_with_cached_matrix(self):
        """Строка 115: использование кэшированной матрицы"""
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["java", "spring"]]
        comparator.fit_market(corpus)
        # Первый вызов создаст _market_tfidf_matrix
        score1, _ = comparator.compare(["python"]).unwrap()
        # Второй вызов должен использовать кэш
        assert hasattr(comparator, "_market_tfidf_matrix")
        score2, _ = comparator.compare(["python"]).unwrap()
        assert score1 == score2

    def test_fit_market_sets_fitted_flag(self):
        """Проверка что fit_market устанавливает флаг fitted"""
        comparator = CompetencyComparator(use_embeddings=False)
        corpus = [["python", "sql"], ["java", "spring"]]
        result = comparator.fit_market(corpus)
        assert result.is_ok() and result.unwrap() is True
        assert comparator.fitted is True

    def test_weighted_coverage_hybrid(self):
        """Строка 139-158: weighted_coverage с эмбеддингами"""
        comparator = CompetencyComparator(use_embeddings=True)
        mock_emb = MagicMock()
        # Мокаем embed_skills чтобы возвращать эмбеддинги
        mock_emb.embed_skills.return_value = np.random.rand(2, 384)
        comparator.embedding_comparator = mock_emb

        coverage = comparator.weighted_coverage(
            ["python", "sql"], {"python": 0.9, "sql": 0.7, "docker": 0.5}, use_hybrid=True
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

        coverage = comparator.weighted_coverage(["python"], {"nonexistent": 0.5})
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
            use_hybrid=False,  # не-hybrid режим
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

    # ======================== ПОКРЫТИЕ СТРОК 30-31 ============================
    def test_init_loads_embedding_model(self):
        """Проверяет, что при создании объекта загружается модель."""
        fake_model = MagicMock()
        with patch("src.analyzers.comparison.embedding_comparator.EmbeddingProviderFactory.get",
                   return_value=fake_model) as mock_get:
            comp = EmbeddingComparator()
            mock_get.assert_called_once_with(None)
            assert comp.model is fake_model

    # ======================== ПОКРЫТИЕ СТРОК 80-117 ============================
    def test_build_market_index_loads_valid_cache(self, tmp_path):
        """Строки 65-82: успешная загрузка существующего кэша."""
        # Подготовим кэш
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        comp = EmbeddingComparator(cache_dir=str(cache_dir))
        fake_model = MagicMock()
        comp.model = fake_model

        fake_emb = np.array([[0.1, 0.2], [0.3, 0.4]])
        fake_skills = ["python", "java"]
        cache_path = comp._get_cache_path("market_embeddings", "middle")
        joblib.dump({"embeddings": fake_emb, "skills": fake_skills}, cache_path)

        # Манифест пусть будет совместим – подменим его
        with patch("src.analyzers.comparison.embedding_comparator.ArtifactManifest") as MockManifest:
            mock_manifest = MockManifest.load.return_value
            from src import Ok; mock_manifest.is_compatible.return_value = Ok(True)

            comp.build_market_index([], level="middle")  # список не важен, загрузится из кэша

        np.testing.assert_array_equal(comp.market_embeddings, fake_emb)
        assert comp.market_skills == fake_skills

    def test_build_market_index_invalidated_by_model(self, tmp_path):
        """Строки 70-73: манифест несовместим -> пересчёт."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        comp = EmbeddingComparator(cache_dir=str(cache_dir))
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 2
        fake_model.encode.return_value = np.array([[0.5, 0.6]])
        comp.model = fake_model

        # Сохраним старый кэш и манифест
        cache_path = comp._get_cache_path("market_embeddings", "middle")
        joblib.dump({"embeddings": np.array([[1.0, 2.0]]), "skills": ["old"]}, cache_path)
        manifest_path = cache_path.with_suffix(".manifest.json")
        manifest_path.write_text('{"model_version": "old", "metrics": {}}')

        from src import Ok
        from src.analyzers.comparison.embedding_comparator import ArtifactManifest as RealArtifactManifest

        real_manifest = RealArtifactManifest(cache_path)
        from src import Ok; real_manifest.is_compatible = MagicMock(return_value=Ok(False))
        with patch("src.analyzers.comparison.embedding_comparator.ArtifactManifest.load", return_value=Ok(real_manifest)):
            with patch("src.analyzers.comparison.embedding_comparator.ArtifactManifest._get_embedding_model_version", return_value="new"):
                comp.build_market_index(["python"], level="middle")

        # Должен был пересчитать заново
        assert comp.market_skills == ["python"]
        np.testing.assert_array_equal(comp.market_embeddings, np.array([[0.5, 0.6]]))

    def test_build_market_index_corrupted_cache(self, tmp_path):
        """Строки 84-88: битый кэш -> удаляется и пересоздаётся."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        comp = EmbeddingComparator(cache_dir=str(cache_dir))
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 2
        fake_model.encode.return_value = np.array([[0.7, 0.8]])
        comp.model = fake_model

        cache_path = comp._get_cache_path("market_embeddings", "middle")
        # Запишем битый файл (не pickle)
        cache_path.write_text("trash")

        comp.build_market_index(["python"], level="middle")
        assert comp.market_skills == ["python"]

    def test_build_market_index_atomic_write_failure(self, tmp_path):
        """Строки 100-103: ошибка при атомарной записи -> исключение."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        comp = EmbeddingComparator(cache_dir=str(cache_dir))
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 2
        fake_model.encode.return_value = np.array([[0.1, 0.2]])
        comp.model = fake_model

        with patch("joblib.dump", side_effect=IOError("disk full")):
            with pytest.raises(IOError, match="disk full"):
                comp.build_market_index(["python"], level="middle")

    def test_build_market_index_manifest_save_failure(self, tmp_path):
        """Строки 108-109: ошибка сохранения манифеста не роняет процесс."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        comp = EmbeddingComparator(cache_dir=str(cache_dir))
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 2
        fake_model.encode.return_value = np.array([[0.1, 0.2]])
        comp.model = fake_model

        with patch("src.analyzers.comparison.embedding_comparator.ArtifactManifest") as MockManifest:
            instance = MockManifest.return_value
            instance.save.side_effect = Exception("no write access")
            instance.is_compatible.return_value = Ok(True)
            # Не должно упасть
            comp.build_market_index(["python"], level="middle")
            assert comp.market_skills == ["python"]

    # ======================== ПОКРЫТИЕ СТРОК 133-137 ==========================
    def test_compare_student_to_market_logs_empty_weights(self, mocker):
        """Строки 133-134: предупреждение при пустых skill_weights."""
        comp = EmbeddingComparator()
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 3
        fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        comp.model = fake_model
        comp.market_skills = ["python"]
        comp.market_embeddings = np.array([[0.1, 0.2, 0.3]])
        comp.skill_weights = {}

        # Патчим логгер, чтобы проверить вызов warning
        mock_warning = mocker.patch("src.analyzers.comparison.embedding_comparator.logger.warning")
        comp.compare_student_to_market(["python"])
        mock_warning.assert_called_once_with("skill_weights_empty")

    def test_compare_student_to_market_logs_weights_count(self):
        """Строки 135-137: отладка с количеством весов."""
        comp = EmbeddingComparator()
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 3
        fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        comp.model = fake_model
        comp.market_skills = ["python"]
        comp.market_embeddings = np.array([[0.1, 0.2, 0.3]])
        comp.skill_weights = {"python": 0.9}

        with patch("src.analyzers.comparison.embedding_comparator.logger") as mock_logger:
            comp.compare_student_to_market(["python"])
            mock_logger.debug.assert_called_with("skill_weights_count", count=1)

    # ======================== ПОКРЫТИЕ СТРОК 145 и 154 ========================
    def test_compare_student_to_market_weighted_calculation(self):
        """Строка 145: вычисление с весами (effective_sim**2 * weight)."""
        comp = EmbeddingComparator()
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 2
        # student эмбеддинг: [1,0] даст cosine_sim с рыночным [1,0] = 1.0
        fake_model.encode.side_effect = lambda skills, **kwargs: np.array([[1.0, 0.0]])
        comp.model = fake_model
        comp.market_skills = ["python", "java"]
        comp.market_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])  # python, java
        comp.skill_weights = {"python": 0.9, "java": 0.1}

        result = comp.compare_student_to_market(["python"]).unwrap()
        # python: sim=1 -> effective=1, weighted=1*0.9=0.9
        # java: sim=0 -> effective=0, weighted=0*0.1=0
        # total_weighted=0.9, total_weight=1.0, coverage=0.9
        assert result["weighted_coverage"] == pytest.approx(0.9)

    def test_compare_student_to_market_no_weights_calculation(self):
        """Строка 154: вычисление без весов (сумма effective_sim / количество)."""
        comp = EmbeddingComparator()
        fake_model = MagicMock()
        fake_model.get_sentence_embedding_dimension.return_value = 2
        fake_model.encode.side_effect = lambda skills, **kwargs: np.array([[0.6, 0.8]])
        comp.model = fake_model
        comp.market_skills = ["python", "java"]
        comp.market_embeddings = np.array([[0.6, 0.8], [0.0, 1.0]])
        comp.skill_weights = {}   # без весов

        result = comp.compare_student_to_market(["python"]).unwrap()
        # python: sim=1.0 -> effective=1.0, java: sim=0.8 -> effective=0.64
        # total_weighted = 1.0 + 0.64 = 1.64, total_weight = 2.0
        # coverage = 1.64 / 2.0 = 0.82
        assert result["weighted_coverage"] == pytest.approx(0.82)

    # ======================== ПОКРЫТИЕ СТРОКИ 235 ==============================
    def test_hybrid_compare_no_clusters_uses_global(self):
        """Строка 235: если best_cluster is None -> hybrid_score = global_score."""
        comp = EmbeddingComparator()
        # Мокируем compare_student_to_market и compare_to_clusters
        with patch.object(comp, "compare_student_to_market",
                          return_value=Ok({"avg_similarity": 0.7, "weighted_coverage": 0.7})):
            with patch.object(comp, "compare_to_clusters",
                              return_value={"clusters": []}):
                result = comp.hybrid_compare(["python"], {"python": 0.9})
                assert result["global_score"] == 0.7
                assert result["cluster_score"] is None
                assert result["hybrid_score"] == 0.7   # строка 235
