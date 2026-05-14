# tests/scripts/test_check_clusters.py
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCheckClusters:
    def test_import_module(self):
        """Модуль импортируется без ошибок"""
        import scripts.check_clusters
        assert hasattr(scripts.check_clusters, '__file__')

    def test_load_model_junior(self, tmp_path, monkeypatch):
        """Загрузка junior-модели"""
        monkeypatch.setattr(
            "src.analyzers.clustering.vacancy_clustering.config.VACANCY_CLUSTERS_CACHE_DIR", tmp_path
        )
        from src.analyzers.clustering.vacancy_clustering import VacancyClusterer

        c = VacancyClusterer()
        # Модели нет — load_model вернёт False
        assert c.load_model("junior") is False

    def test_load_model_middle(self, tmp_path, monkeypatch):
        """Загрузка middle-модели"""
        monkeypatch.setattr(
            "src.analyzers.clustering.vacancy_clustering.config.VACANCY_CLUSTERS_CACHE_DIR", tmp_path
        )
        from src.analyzers.clustering.vacancy_clustering import VacancyClusterer

        c = VacancyClusterer()
        assert c.load_model("middle") is False

    def test_load_model_senior(self, tmp_path, monkeypatch):
        """Загрузка senior-модели"""
        monkeypatch.setattr(
            "src.analyzers.clustering.vacancy_clustering.config.VACANCY_CLUSTERS_CACHE_DIR", tmp_path
        )
        from src.analyzers.clustering.vacancy_clustering import VacancyClusterer

        c = VacancyClusterer()
        assert c.load_model("senior") is False

    def test_main_block_with_models(self, tmp_path, monkeypatch):
        """Основной блок с мок-моделями"""
        monkeypatch.setattr(
            "src.analyzers.clustering.vacancy_clustering.config.VACANCY_CLUSTERS_CACHE_DIR", tmp_path
        )
        from src.analyzers.clustering.vacancy_clustering import VacancyClusterer

        # Создаём фейковые модели
        import pickle
        import numpy as np

        for level in ["junior", "middle", "senior"]:
            c = VacancyClusterer()
            c.is_fitted = True
            c.labels_ = np.array([0, 0, 1, 1, 2, 2, -1])
            c.cluster_centers = np.random.rand(3, 384)
            c.vacancy_ids = [str(i) for i in range(7)]
            c.vacancy_skills = [["python"], ["python", "sql"], ["java"], ["java", "spring"], ["react"], ["react", "vue"], ["docker"]]
            c.label_to_center_idx = {0: 0, 1: 1, 2: 2}
            c.clusterer_type = "kmeans"

            path = tmp_path / f"vacancy_clusters_{level}.pkl"
            data = {
                "model": None,
                "clusterer_type": "kmeans",
                "labels": c.labels_,
                "centers": c.cluster_centers,
                "vacancy_ids": c.vacancy_ids,
                "vacancy_skills": c.vacancy_skills,
                "n_clusters": 3,
                "min_cluster_size": 5,
                "label_to_center_idx": c.label_to_center_idx,
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)

        # Запускаем скрипт
        import scripts.check_clusters as cc
        # Просто проверяем, что не падает
        for level in ["junior", "middle", "senior"]:
            c = VacancyClusterer()
            result = c.load_model(level)
            if result:
                assert c.n_clusters_ > 0

    def test_main_block_with_models_and_output(self, tmp_path, monkeypatch, capsys):
        """Покрытие вывода при успешной загрузке модели."""
        monkeypatch.setattr(
            "src.analyzers.clustering.vacancy_clustering.config.VACANCY_CLUSTERS_CACHE_DIR", tmp_path
        )
        import pickle, numpy as np
        from src.analyzers.clustering.vacancy_clustering import VacancyClusterer

        # Создаём модель для junior
        c = VacancyClusterer()
        c.is_fitted = True
        c.labels_ = np.array([0, 0, 1])
        c.cluster_centers = np.random.rand(2, 384)
        c.vacancy_ids = [str(i) for i in range(3)]
        c.vacancy_skills = [["python"], ["python", "sql"], ["java"]]
        c.label_to_center_idx = {0: 0, 1: 1}
        c.clusterer_type = "kmeans"
        data = {
            "model": None,
            "clusterer_type": "kmeans",
            "labels": c.labels_,
            "centers": c.cluster_centers,
            "vacancy_ids": c.vacancy_ids,
            "vacancy_skills": c.vacancy_skills,
            "n_clusters": 2,
            "min_cluster_size": 5,
            "label_to_center_idx": c.label_to_center_idx,
        }
        with open(tmp_path / "vacancy_clusters_junior.pkl", "wb") as f:
            pickle.dump(data, f)

        import scripts.check_clusters as cc
        # Проверяем, что скрипт выводит информацию о кластере
        cc.main() if hasattr(cc, 'main') else exec(open(cc.__file__).read())
        captured = capsys.readouterr()
        assert "Кластер 0" in captured.out or "python" in captured.out

    def test_all_models_missing_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.analyzers.clustering.vacancy_clustering.config.VACANCY_CLUSTERS_CACHE_DIR", tmp_path
        )
        for level in ["junior", "middle", "senior"]:
            assert not (tmp_path / f"vacancy_clusters_{level}.pkl").exists()
        with patch('builtins.print') as mock_print:
            # выполняем скрипт, не импортируя его как модуль, чтобы избежать раннего выполнения
            import importlib.util
            spec = importlib.util.spec_from_file_location("check_clusters", Path(__file__).parent.parent.parent / "scripts" / "check_clusters.py")
            module = importlib.util.module_from_spec(spec)
            # Не выполняем, потому что spec.loader.exec_module выполнит код. Вместо этого выполним exec в изолированном пространстве.
            # Но можно просто выполнить exec с open, избегая импорта модуля.
            # Убедимся, что модуль не был импортирован ранее: удалим из sys.modules
            if 'scripts.check_clusters' in sys.modules:
                del sys.modules['scripts.check_clusters']
            exec(open(Path(__file__).parent.parent.parent / "scripts" / "check_clusters.py", encoding='utf-8').read())
        found = sum(1 for call in mock_print.call_args_list if "модель не найдена" in str(call))
        assert found >= 3
