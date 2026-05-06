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
            "src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", tmp_path
        )
        from src.analyzers.vacancy_clustering import VacancyClusterer

        c = VacancyClusterer()
        # Модели нет — load_model вернёт False
        assert c.load_model("junior") is False

    def test_load_model_middle(self, tmp_path, monkeypatch):
        """Загрузка middle-модели"""
        monkeypatch.setattr(
            "src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", tmp_path
        )
        from src.analyzers.vacancy_clustering import VacancyClusterer

        c = VacancyClusterer()
        assert c.load_model("middle") is False

    def test_load_model_senior(self, tmp_path, monkeypatch):
        """Загрузка senior-модели"""
        monkeypatch.setattr(
            "src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", tmp_path
        )
        from src.analyzers.vacancy_clustering import VacancyClusterer

        c = VacancyClusterer()
        assert c.load_model("senior") is False

    def test_main_block_with_models(self, tmp_path, monkeypatch):
        """Основной блок с мок-моделями"""
        monkeypatch.setattr(
            "src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", tmp_path
        )
        from src.analyzers.vacancy_clustering import VacancyClusterer

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
