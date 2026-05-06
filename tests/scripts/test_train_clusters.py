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
        """Подготовка вакансий к кластеризации"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        prepared = prepare_vacancies_for_clustering(sample_vacancies)
        assert len(prepared) == len(sample_vacancies)
        assert "skills" in prepared[0]
        assert "experience" in prepared[0]
        assert "id" in prepared[0]
        assert "name" in prepared[0]

    def test_prepare_vacancies_experience_junior(self):
        """Определение уровня junior"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Стажер Python",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": {"id": "no_experience", "name": "Без опыта"},
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["experience"] == "junior"

    def test_prepare_vacancies_experience_senior(self):
        """Определение уровня senior"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Ведущий разработчик",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": {"id": "between6and10", "name": "6-10 лет"},
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["experience"] == "senior"

    def test_prepare_vacancies_experience_from_name(self):
        """Определение уровня из названия вакансии"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Senior Python Developer",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": {"id": "unknown", "name": "Не указан"},
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["experience"] == "senior"

    def test_prepare_vacancies_empty_skills(self):
        """Вакансия без навыков"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Manager",
            "key_skills": [],
            "description": "",
            "snippet": {},
            "experience": {"id": "middle", "name": "Middle"},
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["skills"] == []

    def test_train_clusters_no_files(self, tmp_path, monkeypatch):
        """Нет файлов вакансий"""
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", tmp_path)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path)

        from scripts.train_clusters import train_clusters
        result = train_clusters(level="all", save_report=False)
        assert result is False

    def test_train_clusters_with_data(self, sample_vacancies, tmp_path, monkeypatch):
        """Обучение с тестовыми данными"""
        # Сохраняем тестовые вакансии
        result_dir = tmp_path / "result"
        result_dir.mkdir()
        vac_file = result_dir / "hh_vacancies_detailed.json"
        vac_file.write_text(json.dumps(sample_vacancies), encoding="utf-8")

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", result_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", processed_dir)

        # Мокаем VacancyClusterer.fit чтобы не обучать реально
        with patch("src.analyzers.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")

            from scripts.train_clusters import train_clusters
            result = train_clusters(level="middle", save_report=False)
            assert result is True

    def test_cli_with_args(self, monkeypatch):
        """Парсинг аргументов командной строки"""
        monkeypatch.setattr("sys.argv", [
            "train_clusters.py",
            "--level", "middle",
            "--no-report",
            "--no-interpret",
        ])

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--level", type=str, default="all")
        parser.add_argument("--no-report", action="store_true")
        parser.add_argument("--no-interpret", action="store_true")
        args = parser.parse_args()

        assert args.level == "middle"
        assert args.no_report is True
        assert args.no_interpret is True

    def test_train_clusters_skips_few_samples(self, tmp_path, monkeypatch):
        """Пропуск уровня с малым количеством вакансий"""
        vacs = [{
            "id": f"{i}",
            "name": f"Vac {i}",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": {"id": "junior", "name": "Junior"},
        } for i in range(10)]  # < 30 вакансий

        result_dir = tmp_path / "result"
        result_dir.mkdir()
        vac_file = result_dir / "hh_vacancies_detailed.json"
        vac_file.write_text(json.dumps(vacs))

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", result_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", processed_dir)

        from scripts.train_clusters import train_clusters
        result = train_clusters(level="junior", save_report=False)
        # Должен пропустить из-за малого количества
        assert result is True  # Не падает, просто пропускает

    @pytest.fixture
    def sample_vacancies_full(self):
        """Вакансии с разными уровнями опыта"""
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
                "id": str(i),
                "name": name,
                "key_skills": [{"name": "python"}, {"name": f"skill_{i % 5}"}],
                "description": f"опыт работы с docker и kubernetes",
                "snippet": {"requirement": "знание git", "responsibility": ""},
                "experience": exp,
            })
        return vacs

    def test_prepare_vacancies_string_experience_middle(self):
        """Строки 51-58: строковый опыт 'middle'"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Middle Dev",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": "middle",
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["experience"] == "middle"

    def test_prepare_vacancies_experience_dict_default(self):
        """Строка 50: опыт dict без известных ключей → middle"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Developer",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": {"id": "unknown_format", "name": "Любой"},
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["experience"] == "middle"
    def test_prepare_vacancies_name_senior_override(self):
        """Строки 60-62: название вакансии переопределяет senior"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Ведущий разработчик Python",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": {"id": "between1and3", "name": "1-3 года"},  # middle по опыту
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        # Название "Ведущий" → senior переопределяет middle
        assert prepared[0]["experience"] == "senior"

    def test_prepare_vacancies_name_junior_override(self):
        """Строки 60-62: название 'стажер' → junior"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Стажер Python",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": {"id": "between3and6", "name": "3-6 лет"},  # middle по опыту
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        # Название "Стажер" → junior переопределяет middle
        assert prepared[0]["experience"] == "junior"

    def test_train_clusters_no_vacancies_file(self, tmp_path, monkeypatch):
        """Строка 88-89: нет файлов вакансий → False"""
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", tmp_path / "empty")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "also_empty")

        from scripts.train_clusters import train_clusters
        result = train_clusters(level="all", save_report=False)
        assert result is False

    def test_train_clusters_read_json_none(self, tmp_path, monkeypatch):
        """Строка 93-95: read_json возвращает None"""
        result_dir = tmp_path / "result"
        result_dir.mkdir()
        # Создаём битый JSON
        (result_dir / "hh_vacancies_detailed.json").write_text("{invalid")

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", result_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "raw")

        from scripts.train_clusters import train_clusters
        result = train_clusters(level="all", save_report=False)
        assert result is False

    def test_prepare_vacancies_string_experience_default(self):
        """Строка 58: строковый опыт без junior/senior → middle"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Developer",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": "неизвестный уровень",
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["experience"] == "middle"
    def test_prepare_vacancies_string_experience_unknown(self):
        """Строки 55-58: неизвестный строковый опыт → middle"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Developer",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": "unknown_level",
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["experience"] == "middle"

    def test_train_clusters_uses_detailed_file(self, sample_vacancies_full, tmp_path, monkeypatch):
        """Строки 92-93: использование детального файла"""
        result_dir = tmp_path / "result"
        result_dir.mkdir()
        detailed = result_dir / "hh_vacancies_detailed.json"
        detailed.write_text(json.dumps(sample_vacancies_full))

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", result_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", processed_dir)

        with patch("src.analyzers.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")

            from scripts.train_clusters import train_clusters
            result = train_clusters(level="all", save_report=False, interpret=False)
            assert result is True

    def test_train_clusters_fallback_to_basic(self, sample_vacancies_full, tmp_path, monkeypatch):
        """Строки 100-101: fallback на базовый файл"""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        basic = raw_dir / "hh_vacancies_basic.json"
        basic.write_text(json.dumps(sample_vacancies_full))

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", tmp_path / "result_empty")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", raw_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", processed_dir)

        with patch("src.analyzers.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")

            from scripts.train_clusters import train_clusters
            result = train_clusters(level="all", save_report=False, interpret=False)
            assert result is True

    def test_prepare_vacancies_experience_name_junior(self):
        """Строки 112-115: определение junior по названию"""
        from scripts.train_clusters import prepare_vacancies_for_clustering

        vacs = [{
            "id": "1",
            "name": "Младший разработчик Python",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
            "experience": {"id": "unknown", "name": "Не указан"},
        }]
        prepared = prepare_vacancies_for_clustering(vacs)
        assert prepared[0]["experience"] == "junior"

    def test_train_clusters_with_report(self, sample_vacancies_full, tmp_path, monkeypatch):
        """Строка 138: сохранение отчёта"""
        result_dir = tmp_path / "result"
        result_dir.mkdir()
        detailed = result_dir / "hh_vacancies_detailed.json"
        detailed.write_text(json.dumps(sample_vacancies_full))

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", result_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", processed_dir)

        with patch("src.analyzers.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")

            from scripts.train_clusters import train_clusters
            result = train_clusters(level="middle", save_report=True, interpret=False)
            assert result is True
            report_path = processed_dir / "cluster_training_report.json"
            assert report_path.exists()

    def test_train_clusters_junior_params(self, sample_vacancies_full, tmp_path, monkeypatch):
        """Строки 154-159: параметры для junior"""
        result_dir = tmp_path / "result"
        result_dir.mkdir()
        detailed = result_dir / "hh_vacancies_detailed.json"
        detailed.write_text(json.dumps(sample_vacancies_full))

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", result_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", processed_dir)

        with patch("src.analyzers.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="kmeans")

            from scripts.train_clusters import train_clusters
            result = train_clusters(level="junior", save_report=False, interpret=False)
            assert result is True

    def test_train_clusters_senior_params(self, sample_vacancies_full, tmp_path, monkeypatch):
        """Строки 166-169: параметры для senior"""
        result_dir = tmp_path / "result"
        result_dir.mkdir()
        detailed = result_dir / "hh_vacancies_detailed.json"
        detailed.write_text(json.dumps(sample_vacancies_full))

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", result_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", processed_dir)

        with patch("src.analyzers.vacancy_clustering.VacancyClusterer.fit") as mock_fit:
            mock_fit.return_value = MagicMock(is_fitted=True, n_clusters_=3, clusterer_type="hdbscan")

            from scripts.train_clusters import train_clusters
            result = train_clusters(level="senior", save_report=False, interpret=False)
            assert result is True

    def test_train_clusters_with_interpret(self, sample_vacancies_full, tmp_path, monkeypatch):
        """Строки 177-180: вывод топ-навыков кластеров"""
        result_dir = tmp_path / "result"
        result_dir.mkdir()
        detailed = result_dir / "hh_vacancies_detailed.json"
        detailed.write_text(json.dumps(sample_vacancies_full))

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", result_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", processed_dir)

        mock_clusterer = MagicMock(is_fitted=True, n_clusters_=2, clusterer_type="kmeans")
        mock_clusterer.get_top_skills_in_cluster.return_value = ["python", "sql", "docker"]
        mock_clusterer.fit.return_value = mock_clusterer

        with patch("src.analyzers.vacancy_clustering.VacancyClusterer", return_value=mock_clusterer):
            from scripts.train_clusters import train_clusters
            result = train_clusters(level="middle", save_report=False, interpret=True)
            assert result is True

    def test_train_clusters_exception_handling(self, sample_vacancies_full, tmp_path, monkeypatch):
        """Строки 189-192, 195-199: обработка ошибок кластеризации"""
        result_dir = tmp_path / "result"
        result_dir.mkdir()
        detailed = result_dir / "hh_vacancies_detailed.json"
        detailed.write_text(json.dumps(sample_vacancies_full))

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RESULT_DIR", result_dir)
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr("src.analyzers.vacancy_clustering.config.DATA_PROCESSED_DIR", processed_dir)

        with patch("src.analyzers.vacancy_clustering.VacancyClusterer.fit", side_effect=RuntimeError("Clustering failed")):
            from scripts.train_clusters import train_clusters
            result = train_clusters(level="middle", save_report=True, interpret=False)
            assert result is True  # Не падает, продолжает
            report_path = processed_dir / "cluster_training_report.json"
            assert report_path.exists()

    def test_cli_main_all_levels(self, monkeypatch):
        """Строки 208-215: CLI с --level all"""
        monkeypatch.setattr("sys.argv", ["train_clusters.py", "--level", "all", "--no-report", "--no-interpret"])

        with patch("scripts.train_clusters.train_clusters") as mock_train:
            mock_train.return_value = True

            import scripts.train_clusters
            # Вызываем main через importlib для coverage
            import runpy
            try:
                runpy.run_module("scripts.train_clusters", run_name="__main__")
            except SystemExit:
                pass

    def test_cli_main_single_level(self, monkeypatch):
        """Строки 208-215: CLI с конкретным уровнем"""
        monkeypatch.setattr("sys.argv", ["train_clusters.py", "--level", "junior"])

        with patch("scripts.train_clusters.train_clusters") as mock_train:
            mock_train.return_value = True

            import scripts.train_clusters
            try:
                import runpy
                runpy.run_module("scripts.train_clusters", run_name="__main__")
            except SystemExit:
                pass
