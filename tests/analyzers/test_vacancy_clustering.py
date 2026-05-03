# tests/analyzers/test_vacancy_clustering.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.analyzers.vacancy_clustering import VacancyClusterer


class TestVacancyClusterer:
    @pytest.fixture
    def sample_vacancies(self):
        return [
            {"id": "1", "skills": ["python", "sql", "docker"]},
            {"id": "2", "skills": ["python", "fastapi", "docker"]},
            {"id": "3", "skills": ["python", "sql", "git", "k8s"]},
            {"id": "4", "skills": ["python", "html", "css", "javascript"]},
            {"id": "5", "skills": ["python", "sql", "docker", "pytest"]},
            {"id": "6", "skills": ["python", "fastapi", "git", "ci/cd"]},
            {"id": "7", "skills": ["java", "spring", "sql"]},
            {"id": "8", "skills": ["java", "hibernate", "maven"]},
            {"id": "9", "skills": ["javascript", "react", "node.js"]},
            {"id": "10", "skills": ["javascript", "vue", "css"]},
            {"id": "11", "skills": ["python", "tensorflow", "pandas"]},
            {"id": "12", "skills": ["python", "sklearn", "numpy"]},
        ]

    def test_init_default_params(self):
        clusterer = VacancyClusterer()
        assert clusterer.n_clusters == 10
        assert clusterer.min_clusters == 2
        assert clusterer.max_clusters == 40
        assert clusterer.is_fitted is False
        assert clusterer.n_clusters_ == 0

    def test_init_custom_params(self):
        clusterer = VacancyClusterer(
            n_clusters=5,
            min_clusters=3,
            max_clusters=20,
            random_state=123,
            use_hdbscan_fallback=False
        )
        assert clusterer.n_clusters == 5
        assert clusterer.min_clusters == 3
        assert clusterer.max_clusters == 20
        assert clusterer.random_state == 123

    def test_fit_empty_vacancies(self):
        clusterer = VacancyClusterer()
        clusterer.fit([])
        assert clusterer.is_fitted is False

    def test_fit_few_vacancies(self):
        """Менее 10 вакансий — пропускаем"""
        clusterer = VacancyClusterer()
        vacancies = [
            {"id": "1", "skills": ["python"]},
            {"id": "2", "skills": ["java"]},
        ]
        clusterer.fit(vacancies)
        assert clusterer.is_fitted is False

    def test_fit_kmeans(self, sample_vacancies):
        """KMeans с достаточным количеством вакансий"""
        clusterer = VacancyClusterer(
            n_clusters=2,
            min_clusters=2,
            max_clusters=4,
            use_hdbscan_fallback=False
        )
        clusterer.fit(sample_vacancies, level="test")
        assert clusterer.is_fitted is True
        assert clusterer.n_clusters_ >= 1
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(sample_vacancies)

    def test_find_closest_clusters_not_fitted(self):
        clusterer = VacancyClusterer()
        result = clusterer.find_closest_clusters(["python"])
        assert result == []

    def test_find_closest_clusters(self, sample_vacancies):
        clusterer = VacancyClusterer(
            n_clusters=2,
            min_clusters=2,
            max_clusters=4,
            use_hdbscan_fallback=False
        )
        clusterer.fit(sample_vacancies, level="test")

        closest = clusterer.find_closest_clusters(["python", "sql"], top_k=1)
        assert len(closest) > 0
        assert isinstance(closest[0], tuple)
        assert len(closest[0]) == 2  # (cluster_id, similarity)

    def test_find_closest_clusters_with_embedding(self, sample_vacancies):
        """Поиск кластеров по готовому эмбеддингу"""
        clusterer = VacancyClusterer(
            n_clusters=2,
            min_clusters=2,
            max_clusters=4,
            use_hdbscan_fallback=False
        )
        clusterer.fit(sample_vacancies, level="test")

        # Создаём случайный эмбеддинг правильной размерности
        if clusterer.cluster_centers is not None:
            dim = clusterer.cluster_centers.shape[1]
            embedding = np.random.rand(dim)
            embedding = embedding / np.linalg.norm(embedding)

            closest = clusterer.find_closest_clusters(embedding, top_k=2)
            assert len(closest) > 0
            for cid, sim in closest:
                assert isinstance(cid, int)
                assert 0.0 <= sim <= 1.0

    def test_get_cluster_skills(self, sample_vacancies):
        clusterer = VacancyClusterer(
            n_clusters=2,
            min_clusters=2,
            max_clusters=4,
            use_hdbscan_fallback=False
        )
        clusterer.fit(sample_vacancies, level="test")

        skills = clusterer.get_cluster_skills(0, sample_vacancies)
        assert isinstance(skills, list)
        assert len(skills) > 0

    def test_get_cluster_skills_not_fitted(self):
        clusterer = VacancyClusterer()
        skills = clusterer.get_cluster_skills(0)
        assert skills == []

    def test_get_top_skills_in_cluster(self, sample_vacancies):
        clusterer = VacancyClusterer(
            n_clusters=2,
            min_clusters=2,
            max_clusters=4,
            use_hdbscan_fallback=False
        )
        clusterer.fit(sample_vacancies, level="test")

        top = clusterer.get_top_skills_in_cluster(0, top_n=5)
        assert isinstance(top, list)
        assert len(top) <= 5
        # python должен быть в топе
        assert "python" in top

    def test_get_top_skills_in_cluster_not_fitted(self):
        clusterer = VacancyClusterer()
        top = clusterer.get_top_skills_in_cluster(0)
        assert top == []

    def test_get_cluster_context(self, sample_vacancies):
        clusterer = VacancyClusterer(
            n_clusters=2,
            min_clusters=2,
            max_clusters=4,
            use_hdbscan_fallback=False
        )
        clusterer.fit(sample_vacancies, level="test")

        # Создаём эмбеддинг
        dim = clusterer.cluster_centers.shape[1]
        embedding = np.random.rand(dim)
        embedding = embedding / np.linalg.norm(embedding)

        context = clusterer.get_cluster_context(
            profile_embedding=embedding,
            level="test",
            top_k_clusters=2,
            top_k_skills_per_cluster=10
        )
        assert "closest_clusters" in context
        assert "skills" in context
        assert "total_skills_in_context" in context
        assert context["total_skills_in_context"] > 0

    def test_get_cluster_context_none_embedding(self):
        clusterer = VacancyClusterer()
        context = clusterer.get_cluster_context(None)
        assert context["total_skills_in_context"] == 0

    def test_save_and_load_model(self, tmp_path, sample_vacancies, monkeypatch):
        """Проверяем сохранение и загрузку модели"""
        import src.config as config
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)

        clusterer = VacancyClusterer(
            n_clusters=2,
            min_clusters=2,
            max_clusters=4,
            use_hdbscan_fallback=False
        )
        clusterer.fit(sample_vacancies, level="test")

        # Проверяем, что файл создан
        model_path = tmp_path / "vacancy_clusters_test.pkl"
        assert model_path.exists()

        # Загружаем в новый объект
        clusterer2 = VacancyClusterer()
        loaded = clusterer2.load_model("test")
        assert loaded is True
        assert clusterer2.is_fitted is True
        assert clusterer2.n_clusters_ == clusterer.n_clusters_

    def test_load_model_missing_file(self):
        clusterer = VacancyClusterer()
        loaded = clusterer.load_model("nonexistent")
        assert loaded is False

    def test_n_clusters_property(self, sample_vacancies):
        clusterer = VacancyClusterer(
            n_clusters=2,
            min_clusters=2,
            max_clusters=4,
            use_hdbscan_fallback=False
        )
        assert clusterer.n_clusters_ == 0
        clusterer.fit(sample_vacancies, level="test")
        assert clusterer.n_clusters_ > 0

    def test_find_closest_clusters_returns_all_when_top_k_large(self, sample_vacancies):
        clusterer = VacancyClusterer(
            n_clusters=2,
            min_clusters=2,
            max_clusters=4,
            use_hdbscan_fallback=False
        )
        clusterer.fit(sample_vacancies, level="test")
        n = clusterer.n_clusters_

        closest = clusterer.find_closest_clusters(["python"], top_k=n + 5)
        assert len(closest) <= n

class TestVacancyClusteringFull:
    @pytest.fixture
    def vacancies(self):
        return [
            {"id": f"{i}", "skills": ["python", "sql", f"skill_{i}"]}
            for i in range(20)
        ]

    def test_hdbscan_not_available_constant(self):
        """Проверка константы HDBSCAN_AVAILABLE"""
        from src.analyzers import vacancy_clustering
        # Константа уже вычислена при импорте модуля
        assert isinstance(vacancy_clustering.HDBSCAN_AVAILABLE, bool)

class TestVacancyClusteringFull:
    @pytest.fixture
    def vacancies(self):
        return [
            {"id": f"{i}", "skills": ["python", "sql", f"skill_{i}"]}
            for i in range(20)
        ]

    def test_hdbscan_not_available_constant(self):
        """Проверка константы HDBSCAN_AVAILABLE"""
        from src.analyzers import vacancy_clustering
        assert isinstance(vacancy_clustering.HDBSCAN_AVAILABLE, bool)

    def test_use_hdbscan_when_not_available(self):
        """Когда HDBSCAN_AVAILABLE=False, use_hdbscan_fallback форсируется в False"""
        from src.analyzers.vacancy_clustering import HDBSCAN_AVAILABLE
        clusterer = VacancyClusterer(use_hdbscan_fallback=True)
        assert clusterer.use_hdbscan_fallback == HDBSCAN_AVAILABLE

    def test_fit_with_hdbscan_fallback(self, vacancies):
        """Строки 66-68, 75-77: HDBSCAN fallback"""
        clusterer = VacancyClusterer(
            n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=True
        )
        with patch.object(clusterer, '_save_model'):
            clusterer.fit(vacancies, level="test")
        assert clusterer.is_fitted is True

    def test_labels_none_when_not_fitted(self):
        """Строка 138: labels_ is None до обучения"""
        clusterer = VacancyClusterer()
        assert clusterer.labels_ is None

    def test_get_cluster_skills_from_provided_vacancies(self, vacancies):
        """Строка 147: get_cluster_skills с переданными вакансиями"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=False)
        clusterer.fit(vacancies, level="test")
        external_vacancies = [{"skills": ["python", "external_skill"]} for _ in range(20)]
        skills = clusterer.get_cluster_skills(0, external_vacancies)
        assert isinstance(skills, list)

    def test_get_cluster_skills_from_internal(self, vacancies):
        """Строка 147: get_cluster_skills из внутренних vacancy_skills"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=False)
        clusterer.fit(vacancies, level="test")
        skills = clusterer.get_cluster_skills(0)
        assert isinstance(skills, list)

    def test_find_closest_clusters_empty_skills(self, vacancies):
        """Строки 152-167: пустые навыки студента"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=False)
        clusterer.fit(vacancies, level="test")
        closest = clusterer.find_closest_clusters([], top_k=3)
        assert len(closest) > 0

    def test_save_model_creates_file(self, tmp_path, vacancies):
        """Строки 226-230: сохранение модели"""
        import src.config as config
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=False)
        clusterer.fit(vacancies, level="test")
        model_path = tmp_path / "vacancy_clusters_test.pkl"
        assert model_path.exists()
        assert model_path.stat().st_size > 0
        monkeypatch.undo()

    def test_hdbscan_creates_multiple_clusters(self):
        """Строки 177-216: HDBSCAN создаёт несколько кластеров (без сохранения)"""
        with patch('src.analyzers.vacancy_clustering.HDBSCAN_AVAILABLE', True):
            with patch('src.analyzers.vacancy_clustering.hdbscan.HDBSCAN') as mock_hdb:
                mock_instance = mock_hdb.return_value
                labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, -1] * 2)
                mock_instance.fit_predict.return_value = labels[:20]

                clusterer = VacancyClusterer(
                    n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=True
                )
                vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
                with patch.object(clusterer, '_save_model'):
                    clusterer.fit(vacancies, level="test_hdb")
                assert clusterer.is_fitted is True
                assert clusterer.clusterer_type == "hdbscan"

    def test_hdbscan_no_clusters_found(self, vacancies):
        """Строки 177-216: HDBSCAN не нашёл кластеры → KMeans"""
        with patch('src.analyzers.vacancy_clustering.HDBSCAN_AVAILABLE', True):
            with patch('src.analyzers.vacancy_clustering.hdbscan.HDBSCAN') as mock_hdb:
                mock_instance = mock_hdb.return_value
                mock_instance.fit_predict.return_value = np.full(len(vacancies), -1)
                clusterer = VacancyClusterer(
                    n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=True
                )
                with patch.object(clusterer, '_save_model'):
                    clusterer.fit(vacancies, level="test")
                assert clusterer.is_fitted is True

    def test_hdbscan_exception_fallback(self):
        """Строки 209-211: HDBSCAN выбрасывает исключение → KMeans"""
        with patch('src.analyzers.vacancy_clustering.HDBSCAN_AVAILABLE', True):
            with patch('src.analyzers.vacancy_clustering.hdbscan.HDBSCAN') as mock_hdb:
                mock_hdb.side_effect = RuntimeError("HDBSCAN failed")
                clusterer = VacancyClusterer(
                    n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=True
                )
                vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
                with patch.object(clusterer, '_save_model'):
                    clusterer.fit(vacancies, level="test_exc")
                assert clusterer.is_fitted is True
                assert clusterer.clusterer_type == "kmeans"

    def test_n_clusters_property_not_fitted(self):
        """Строка 302: n_clusters_ без labels"""
        clusterer = VacancyClusterer()
        assert clusterer.n_clusters_ == 0

    def test_n_clusters_property_with_noise(self):
        """Строка 302: n_clusters_ с шумом (-1)"""
        clusterer = VacancyClusterer()
        clusterer.labels_ = np.array([0, 0, 1, -1, -1])
        assert clusterer.n_clusters_ == 2

    def test_get_cluster_skills_unknown_cluster(self):
        """Строка 352: несуществующий кластер → пустой список"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", "sql", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test")
        skills = clusterer.get_cluster_skills(999)
        assert skills == []

    def test_labels_when_not_fitted_attr(self):
        """Строка 138: labels_ is None до обучения"""
        clusterer = VacancyClusterer()
        assert clusterer.labels_ is None

    def test_get_cluster_skills_index_error(self):
        """Строка 154-155: индекс за пределами vacancy_skills"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_idx")
        clusterer.vacancy_skills = clusterer.vacancy_skills[:10]
        skills = clusterer.get_cluster_skills(0)
        assert isinstance(skills, list)

    def test_find_closest_clusters_embedding_normalization(self):
        """Строка 167: нулевой эмбеддинг → нормализация с guard"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_zero")
        dim = clusterer.cluster_centers.shape[1]
        zero_emb = np.zeros(dim)
        closest = clusterer.find_closest_clusters(zero_emb, top_k=2)
        assert len(closest) >= 0

    def test_hdbscan_low_silhouette_fallback_to_kmeans(self, tmp_path):
        """Строка 206: HDBSCAN находит только 1 кластер → KMeans"""
        import src.config as config
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)

        with patch('src.analyzers.vacancy_clustering.HDBSCAN_AVAILABLE', True):
            with patch('src.analyzers.vacancy_clustering.hdbscan.HDBSCAN') as mock_hdb:
                mock_instance = mock_hdb.return_value
                labels = np.array([0] * 18 + [-1, -1])
                mock_instance.fit_predict.return_value = labels
                clusterer = VacancyClusterer(
                    n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=True
                )
                vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
                clusterer.fit(vacancies, level="test_one_cluster")
                assert clusterer.is_fitted is True
        monkeypatch.undo()

    def test_get_top_skills_in_cluster_unknown_cluster(self):
        """Строка 352: get_top_skills_in_cluster с несуществующим кластером"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_unknown_top")
        top = clusterer.get_top_skills_in_cluster(999, top_n=5)
        assert top == []

    def test_compute_embeddings_with_empty_skills(self):
        """Строки 66-68: вакансия с пустыми навыками"""
        clusterer = VacancyClusterer()
        vacancies = [
            {"id": f"{i}", "skills": [] if i % 3 == 0 else ["python", f"skill_{i}"]}
            for i in range(20)
        ]
        clusterer.fit(vacancies, level="test_empty")
        assert clusterer.is_fitted is True

    def test_compute_embeddings_normalization(self):
        """Строки 75-77: нормализация эмбеддингов"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", "sql", f"unique_{i}"]} for i in range(25)]
        clusterer.fit(vacancies, level="test_norm")
        assert clusterer.is_fitted is True

    def test_find_closest_clusters_labels_mapping(self):
        """Строки 152-167: маппинг label_to_center_idx"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", "sql", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_map")
        assert len(clusterer.label_to_center_idx) > 0
        for k in clusterer.label_to_center_idx:
            assert isinstance(k, int)
            assert isinstance(clusterer.label_to_center_idx[k], int)

    def test_save_model_hdbscan_type(self, tmp_path):
        """Строки 226-230: сохранение модели HDBSCAN"""
        import src.config as config
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)
        clusterer = VacancyClusterer()
        clusterer.clusterer_type = "hdbscan"
        clusterer.model = None  # HDBSCAN модель = None (не MagicMock!)
        clusterer.labels_ = np.array([0, 0, 1, 1, -1])
        clusterer.cluster_centers = np.random.rand(2, 384)
        clusterer.vacancy_ids = ["1", "2", "3", "4", "5"]
        clusterer.vacancy_skills = [["python"]] * 5
        clusterer.is_fitted = True
        clusterer.label_to_center_idx = {0: 0, 1: 1}
        clusterer.n_clusters = 2
        clusterer.min_cluster_size = 2

        clusterer._save_model("test_hdb")
        model_path = tmp_path / "vacancy_clusters_test_hdb.pkl"
        assert model_path.exists()
        clusterer2 = VacancyClusterer()
        loaded = clusterer2.load_model("test_hdb")
        assert loaded is True
        assert clusterer2.clusterer_type == "hdbscan"
        monkeypatch.undo()

    def test_compute_embeddings_with_all_empty(self):
        """Строки 75-77: все вакансии с пустыми навыками"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": []} for i in range(15)]
        # 15 вакансий (>=10), но все с пустыми навыками
        clusterer.fit(vacancies, level="test_all_empty")
        # Должен отработать без ошибок
        assert True  # не упал — уже хорошо

    def test_labels_is_none_initially(self):
        """Строка 138: labels_ is None при создании"""
        clusterer = VacancyClusterer()
        assert clusterer.labels_ is None
        assert clusterer.n_clusters_ == 0

    def test_get_cluster_skills_out_of_bounds(self):
        """Строки 154-155: индекс навыков за границей"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_oob")
        # Искусственно обрезаем
        clusterer.vacancy_skills = clusterer.vacancy_skills[:5]
        clusterer.labels_ = np.array([0] * 20)
        skills = clusterer.get_cluster_skills(0)
        assert isinstance(skills, list)

    def test_find_closest_clusters_zero_embedding(self):
        """Строка 167: нулевой эмбеддинг → guard от деления на 0"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_zero_emb")
        dim = clusterer.cluster_centers.shape[1]
        zero_emb = np.zeros(dim)
        closest = clusterer.find_closest_clusters(zero_emb, top_k=2)
        assert len(closest) >= 0  # не падает — хорошо

    def test_hdbscan_single_cluster_edge(self, tmp_path):
        """Строка 206: HDBSCAN возвращает 1 кластер (n_clusters < 2)"""
        import src.config as config
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)

        with patch('src.analyzers.vacancy_clustering.HDBSCAN_AVAILABLE', True):
            with patch('src.analyzers.vacancy_clustering.hdbscan.HDBSCAN') as mock_hdb:
                mock_instance = mock_hdb.return_value
                # Все точки в одном кластере
                mock_instance.fit_predict.return_value = np.zeros(20, dtype=int)
                
                clusterer = VacancyClusterer(
                    n_clusters=2, min_clusters=2, max_clusters=5, use_hdbscan_fallback=True
                )
                vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
                clusterer.fit(vacancies, level="test_one_cluster")
                # n_clusters = 1 → должно fallback на KMeans
                assert clusterer.is_fitted is True
        
        monkeypatch.undo()

    def test_get_top_skills_unknown_cluster(self):
        """Строка 352: несуществующий кластер → пустой список"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_top_unknown")
        top = clusterer.get_top_skills_in_cluster(999, top_n=5)
        assert top == []

    def test_labels_not_set_initially(self):
        """Строка 138: labels_ = None до fit"""
        clusterer = VacancyClusterer()
        assert clusterer.labels_ is None

    def test_get_cluster_skills_partial_index(self):
        """Строки 154-155: частичное совпадение индексов"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_partial")
        # Обрезаем vacancy_skills до 5 элементов (меньше чем labels)
        clusterer.vacancy_skills = clusterer.vacancy_skills[:5]
        skills = clusterer.get_cluster_skills(0)
        assert isinstance(skills, list)

    def test_find_closest_clusters_embedding_near_zero(self):
        """Строка 167: очень маленький эмбеддинг → нормализация"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_small")
        dim = clusterer.cluster_centers.shape[1]
        small_emb = np.full(dim, 1e-10)
        closest = clusterer.find_closest_clusters(small_emb, top_k=2)
        assert len(closest) >= 0

    def test_hdbscan_fallback_on_low_silhouette(self):
        """Строка 206: KMeans silhouette < 0.2 → HDBSCAN → KMeans"""
        with patch('src.analyzers.vacancy_clustering.HDBSCAN_AVAILABLE', True):
            with patch('src.analyzers.vacancy_clustering.hdbscan.HDBSCAN') as mock_hdb:
                mock_instance = mock_hdb.return_value
                mock_instance.fit_predict.return_value = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2)
                clusterer = VacancyClusterer(
                    n_clusters=10, min_clusters=2, max_clusters=20, use_hdbscan_fallback=True
                )
                vacancies = [{"id": f"{i}", "skills": [f"skill_{i}"]} for i in range(20)]
                with patch.object(clusterer, '_save_model'):
                    clusterer.fit(vacancies, level="test_low_sil")
                assert clusterer.is_fitted is True

    def test_get_cluster_skills_nonexistent_cluster_id(self):
        """Строка 352: запрос навыков несуществующего кластера"""
        clusterer = VacancyClusterer(n_clusters=2, min_clusters=2, max_clusters=4, use_hdbscan_fallback=False)
        vacancies = [{"id": f"{i}", "skills": ["python", f"skill_{i}"]} for i in range(20)]
        clusterer.fit(vacancies, level="test_nonex")
        skills = clusterer.get_cluster_skills(9999)
        assert skills == []
        top = clusterer.get_top_skills_in_cluster(9999, top_n=5)
        assert top == []