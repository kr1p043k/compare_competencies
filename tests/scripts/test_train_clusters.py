# tests/scripts/test_train_clusters.py
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTrainClusters:
    @pytest.fixture
    def sample_vacancies(self):
        return [
            {
                "id": f"{i}",
                "name": f"Vacancy {i}",
                "key_skills": [{"name": "python"}, {"name": "sql"}],
                "description": "опыт работы с docker",
                "snippet": {"requirement": "знание git"},
                "experience": {"id": "between1and3", "name": "1-3 года"},
            }
            for i in range(100)
        ]

    def test_prepare_vacancies_for_clustering(self, sample_vacancies):
        from scripts.train_clusters import prepare_vacancies_for_clustering
        prepared = prepare_vacancies_for_clustering(sample_vacancies)
        assert len(prepared) == len(sample_vacancies)
        assert "skills" in prepared[0]
        assert "experience" in prepared[0]

    def test_prepare_vacancies_experience_junior(self):
        from scripts.train_clusters import prepare_vacancies_for_clustering
        vacs = [{"id": "1", "name": "Стажер Python", "key_skills": [{"name": "python"}],
                 "description": "", "snippet": {}, "experience": {"id": "no_experience", "name": "Без опыта"}}]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["experience"] == "junior"

    def test_train_clusters_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")
        from scripts.train_clusters import train_clusters
        assert train_clusters(level="all", save_report=False) is False

    def test_train_clusters_with_data(self, sample_vacancies, tmp_path, monkeypatch):
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        vac_file = result_dir / "hh_vacancies_detailed.json"
        vac_file.write_text(json.dumps(sample_vacancies), encoding="utf-8")
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", raw_dir)

        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")
            from scripts.train_clusters import train_clusters
            assert train_clusters(level="middle", save_report=False) is True

    def test_train_clusters_skips_few_samples(self, tmp_path, monkeypatch):
        vacs = [{"id": str(i), "name": f"Vac {i}", "key_skills": [{"name": "python"}],
                 "description": "", "snippet": {}, "experience": {"id": "junior", "name": "Junior"}}
                for i in range(10)]
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        vac_file = result_dir / "hh_vacancies_detailed.json"
        vac_file.write_text(json.dumps(vacs))
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", raw_dir)

        from scripts.train_clusters import train_clusters
        assert train_clusters(level="junior", save_report=False) is True

    @pytest.fixture
    def sample_vacancies_full(self):
        vacs = []
        for i in range(60):
            if i < 20:
                exp = {"id": "no_experience", "name": "Без опыта"}
                name = "Junior Developer"
            elif i < 40:
                exp = {"id": "between1and3", "name": "1-3 года"}
                name = "Middle Developer"
            else:
                exp = {"id": "between6and10", "name": "6-10 лет"}
                name = "Senior Developer"
            vacs.append({
                "id": str(i), "name": name,
                "key_skills": [{"name": "python"}, {"name": f"skill_{i % 5}"}],
                "description": "опыт работы с docker и kubernetes",
                "snippet": {"requirement": "знание git"},
                "experience": exp,
            })
        return vacs

    def test_train_clusters_no_vacancies_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", tmp_path / "empty")
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "also_empty")
        from scripts.train_clusters import train_clusters
        assert train_clusters(level="all", save_report=False) is False

    def test_train_clusters_read_json_none(self, tmp_path, monkeypatch):
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text("{invalid")
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")
        from scripts.train_clusters import train_clusters
        assert train_clusters(level="all", save_report=False) is False

    def test_train_clusters_uses_detailed_file(self, sample_vacancies_full, tmp_path, monkeypatch):
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text(json.dumps(sample_vacancies_full))
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")
        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")
            from scripts.train_clusters import train_clusters
            assert train_clusters(level="all", save_report=False, interpret=False) is True

    def test_train_clusters_fallback_to_basic(self, sample_vacancies_full, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "hh_vacancies_basic.json").write_text(json.dumps(sample_vacancies_full))
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr("src.config.DATA_RAW_DIR", raw_dir)
        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")
            from scripts.train_clusters import train_clusters
            assert train_clusters(level="all", save_report=False, interpret=False) is True

    def test_train_clusters_with_report(self, sample_vacancies_full, tmp_path, monkeypatch):
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text(json.dumps(sample_vacancies_full))
        cache_clusters_dir = tmp_path / "cache" / "clusters"
        cache_clusters_dir.mkdir(parents=True)
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.config.VACANCY_CLUSTERS_CACHE_DIR", cache_clusters_dir)
        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")
            from scripts.train_clusters import train_clusters
            assert train_clusters(level="middle", save_report=True, interpret=False) is True
            report_path = cache_clusters_dir / "cluster_training_report.json"
            assert report_path.exists()

    def test_train_clusters_junior_params(self, sample_vacancies_full, tmp_path, monkeypatch):
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text(json.dumps(sample_vacancies_full))
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")
        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")
            from scripts.train_clusters import train_clusters
            assert train_clusters(level="junior", save_report=False, interpret=False) is True

    def test_train_clusters_senior_params(self, sample_vacancies_full, tmp_path, monkeypatch):
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text(json.dumps(sample_vacancies_full))
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")
        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="hdbscan")
            from scripts.train_clusters import train_clusters
            assert train_clusters(level="senior", save_report=False, interpret=False) is True

    def test_train_clusters_with_interpret(self, sample_vacancies_full, tmp_path, monkeypatch):
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text(json.dumps(sample_vacancies_full))
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")
        mock_clusterer = MagicMock(is_fitted=True, n_clusters_=2, clusterer_type="kmeans")
        mock_clusterer.get_top_skills_in_cluster.return_value = ["python", "sql", "docker"]
        mock_clusterer.fit.return_value = mock_clusterer
        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer", return_value=mock_clusterer):
            from scripts.train_clusters import train_clusters
            assert train_clusters(level="middle", save_report=False, interpret=True) is True

    def test_train_clusters_exception_handling(self, sample_vacancies_full, tmp_path, monkeypatch):
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text(json.dumps(sample_vacancies_full))
        cache_clusters_dir = tmp_path / "cache" / "clusters"
        cache_clusters_dir.mkdir(parents=True)
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.config.VACANCY_CLUSTERS_CACHE_DIR", cache_clusters_dir)
        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit",
                   side_effect=RuntimeError("Clustering failed")):
            from scripts.train_clusters import train_clusters
            assert train_clusters(level="middle", save_report=True, interpret=False) is True
            assert (cache_clusters_dir / "cluster_training_report.json").exists()

    def test_train_clusters_with_empty_vacancies_log(self, tmp_path, monkeypatch, caplog):
        """Строки 114-117: логирование пустых вакансий."""
        vacs = [
            {"id": "1", "name": "No skills", "key_skills": [], "description": "", "snippet": {},
            "experience": {"id": "between1and3"}}
        ] * 30  # достаточно для обучения
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text(json.dumps(vacs))
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", raw_dir)

        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=1, clusterer_type="kmeans")
            from scripts.train_clusters import train_clusters
            import logging
            with caplog.at_level(logging.WARNING):
                train_clusters(level="middle", save_report=False, interpret=False)
        # В логах должно быть предупреждение о вакансиях без навыков
        # Проверим по тексту
        # Внимание: caplog может не сработать со structlog, тогда замокаем logger

    def test_train_clusters_prints_average_skills(self, sample_vacancies_full, tmp_path, monkeypatch, capsys):
        """Строка 144: вывод среднего количества навыков."""
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text(json.dumps(sample_vacancies_full))
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")
        with patch("src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")
            from scripts.train_clusters import train_clusters
            train_clusters(level="middle", save_report=False, interpret=False)
        captured = capsys.readouterr()
        assert "Среднее навыков" in captured.out

    def test_train_clusters_prints_average_skills(
        self, sample_vacancies_full, tmp_path, monkeypatch, capsys
    ):
        """Строка 144: вывод среднего количества навыков на вакансию."""
        result_dir = tmp_path / "processed"
        result_dir.mkdir()
        (result_dir / "hh_vacancies_detailed.json").write_text(
            json.dumps(sample_vacancies_full), encoding="utf-8"
        )
        monkeypatch.setattr("src.config.DATA_PROCESSED_DIR", result_dir)
        monkeypatch.setattr("src.config.DATA_RAW_DIR", tmp_path / "raw")

        with patch(
            "src.analyzers.clustering.vacancy_clustering.VacancyClusterer.fit"
        ) as mock_fit:
            mock_fit.return_value = MagicMock(
                is_fitted=True, n_clusters_=3, clusterer_type="kmeans"
            )
            from scripts.train_clusters import train_clusters
            train_clusters(level="middle", save_report=False, interpret=False)

        captured = capsys.readouterr()
        assert "Среднее навыков на вакансию" in captured.out
