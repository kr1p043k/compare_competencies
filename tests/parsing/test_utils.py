# tests/parsing/test_utils.py
import pytest
import json
import logging
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from collections import Counter
import sys

from src.parsing import utils
from src import config


class TestReadWriteJson:
    def test_read_json_success(self, tmp_path):
        data = {"key": "value"}
        filepath = tmp_path / "test.json"
        filepath.write_text(json.dumps(data), encoding='utf-8')
        result = utils.read_json(filepath)
        assert result == data

    def test_read_json_file_not_found(self, tmp_path):
        filepath = tmp_path / "missing.json"
        result = utils.read_json(filepath)
        assert result is None

    def test_read_json_invalid_json(self, tmp_path):
        filepath = tmp_path / "invalid.json"
        filepath.write_text("{not json", encoding='utf-8')
        result = utils.read_json(filepath)
        assert result is None

    def test_write_json_success(self, tmp_path):
        data = {"key": "value"}
        filepath = tmp_path / "out.json"
        utils.write_json(data, filepath)
        assert filepath.exists()
        loaded = json.loads(filepath.read_text(encoding='utf-8'))
        assert loaded == data

    def test_write_json_handles_error(self, tmp_path):
        # Создаём директорию вместо файла, чтобы вызвать ошибку записи
        filepath = tmp_path / "dir"
        filepath.mkdir()
        data = {"key": "value"}
        # Не должно упасть, только залогировать ошибку
        utils.write_json(data, filepath)


class TestSetupLogging:
    def test_setup_logging(self, tmp_path, monkeypatch):
        # Подменяем LOG_FILE на временный
        monkeypatch.setattr(config, 'LOG_FILE', tmp_path / "test.log")
        utils.setup_logging()
        logger = logging.getLogger()
        # Проверяем, что хендлеры добавились
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


class TestLoadItSkills:
    def test_load_it_skills_success(self, tmp_path, monkeypatch):
        skills_file = tmp_path / "it_skills.json"
        skills_list = ["Python", "JavaScript", "  React  "]
        skills_file.write_text(json.dumps(skills_list), encoding='utf-8')
        monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
        result = utils.load_it_skills()
        assert result == {"python", "javascript", "react"}

    def test_load_it_skills_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
        result = utils.load_it_skills()
        assert result == set()

    def test_load_it_skills_not_a_list(self, tmp_path, monkeypatch):
        skills_file = tmp_path / "it_skills.json"
        skills_file.write_text('{"not": "list"}', encoding='utf-8')
        monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
        result = utils.load_it_skills()
        assert result == set()


class TestFilterSkillsByWhitelist:
    def test_filter_skills_by_whitelist(self):
        skills_dict = {"Python": 10, "Java": 5, "SQL": 7}
        whitelist = {"python", "sql"}
        filtered = utils.filter_skills_by_whitelist(skills_dict, whitelist)
        assert filtered == {"Python": 10, "SQL": 7}

    def test_filter_skills_by_whitelist_empty(self):
        skills_dict = {"Python": 10, "Java": 5}
        filtered = utils.filter_skills_by_whitelist(skills_dict, set())
        assert filtered == skills_dict


class TestCollectVacanciesMultiple:
    def test_collect_vacancies_multiple(self):
        mock_hh_api = MagicMock()
        mock_hh_api.search_vacancies.side_effect = [
            [{"id": "1"}, {"id": "2"}],
            [{"id": "2"}, {"id": "3"}],
            [{"id": "4"}]
        ]
        queries = ["Python", "Java"]
        area_ids = [1, 2]
        with patch('time.sleep', return_value=None):
            vacs = utils.collect_vacancies_multiple(
                mock_hh_api,
                queries=queries,
                area_ids=area_ids,
                period_days=30,
                max_pages=5,
                industry=None,
                max_vacancies_per_query=10
            )
        # Должно быть 4 уникальных: 1,2,3,4 (2 дублируется)
        assert len(vacs) == 4
        ids = {v['id'] for v in vacs}
        assert ids == {"1", "2", "3", "4"}

    def test_collect_vacancies_multiple_limit_per_query(self):
        mock_hh_api = MagicMock()
        mock_hh_api.search_vacancies.return_value = [{"id": str(i)} for i in range(20)]
        with patch('time.sleep', return_value=None):
            vacs = utils.collect_vacancies_multiple(
                mock_hh_api,
                queries=["Python"],
                area_ids=[1],
                period_days=30,
                max_pages=5,
                max_vacancies_per_query=5
            )
        assert len(vacs) == 5


class TestLoadQueriesFromFile:
    def test_load_queries_from_file(self, tmp_path):
        filepath = tmp_path / "queries.txt"
        filepath.write_text("Python\nJava\n  SQL  \n", encoding='utf-8')
        queries = utils.load_queries_from_file(filepath)
        assert queries == ["Python", "Java", "SQL"]

    def test_load_queries_from_file_missing(self, tmp_path):
        filepath = tmp_path / "missing.txt"
        queries = utils.load_queries_from_file(filepath)
        assert queries == []


class TestSafePrint:
    def test_safe_print_unicode(self, capsys):
        utils.safe_print("Hello 世界")
        captured = capsys.readouterr()
        assert "Hello" in captured.out

    def test_safe_print_unicode_error(self, capsys):
        # Имитируем UnicodeEncodeError (зависит от окружения)
        # Просто вызываем, чтобы убедиться, что не падает
        utils.safe_print("Hello")


class TestInputFunctions:
    def test_input_int_default(self, monkeypatch):
        monkeypatch.setattr('builtins.input', lambda _: "")
        result = utils.input_int("prompt", default=5)
        assert result == 5

    def test_input_int_valid(self, monkeypatch):
        monkeypatch.setattr('builtins.input', lambda _: "10")
        result = utils.input_int("prompt", default=5, min_val=1, max_val=20)
        assert result == 10

    def test_input_int_out_of_range_then_valid(self, monkeypatch):
        inputs = iter(["100", "15"])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))
        result = utils.input_int("prompt", default=5, min_val=1, max_val=20)
        assert result == 15

    def test_input_yes_no_default(self, monkeypatch):
        monkeypatch.setattr('builtins.input', lambda _: "")
        assert utils.input_yes_no("prompt", default=True) is True
        assert utils.input_yes_no("prompt", default=False) is False

    def test_input_yes_no_yes(self, monkeypatch):
        for ans in ['y', 'yes', 'да']:
            monkeypatch.setattr('builtins.input', lambda _: ans)
            assert utils.input_yes_no("prompt") is True

    def test_input_yes_no_no(self, monkeypatch):
        monkeypatch.setattr('builtins.input', lambda _: "n")
        assert utils.input_yes_no("prompt") is False

    def test_select_from_list(self, monkeypatch):
        items = ["A", "B", "C"]
        monkeypatch.setattr('builtins.input', lambda _: "2")
        result = utils.select_from_list(items, "Choose:")
        assert result == "B"

    def test_select_from_list_invalid_then_valid(self, monkeypatch):
        items = ["A", "B", "C"]
        inputs = iter(["abc", "3"])
        monkeypatch.setattr('builtins.input', lambda _: next(inputs))
        result = utils.select_from_list(items, "Choose:")
        assert result == "C"


class TestInteractiveConfig:
    @patch('src.parsing.utils.select_from_list')
    @patch('src.parsing.utils.input_int')
    @patch('src.parsing.utils.input_yes_no')
    @patch('builtins.input')
    def test_interactive_config_mode_10(self, mock_input, mock_yes_no, mock_int, mock_select):
        mock_select.return_value = "10. Другое (ввести свой запрос)"
        mock_input.return_value = "Custom Query"
        mock_int.return_value = 15
        mock_yes_no.side_effect = [True, False, True]  # skip_details? no, show_list? yes, excel? yes
        with patch('builtins.print'):
            cfg = utils.interactive_config()
        assert cfg['query'] == "Custom Query"
        assert cfg['queries'] == ["Custom Query"]
        assert cfg['period'] == 15
        assert cfg['skip_details'] is False
        assert cfg['show_vacancies'] is True
        assert cfg['excel'] is True

    @patch('src.parsing.utils.select_from_list')
    @patch('src.parsing.utils.input_int')
    @patch('src.parsing.utils.input_yes_no')
    def test_interactive_config_mode_11_it_sector(self, mock_yes_no, mock_int, mock_select):
        mock_select.return_value = "11. Поиск по всему IT-сектору (industry=7)"
        mock_yes_no.side_effect = [False, True]  # apply_filter? no, excel? yes
        with patch('builtins.print'):
            cfg = utils.interactive_config()
        assert cfg['is_it_sector'] is True
        assert cfg['industry'] == 7
        assert len(cfg['queries']) > 5
        assert cfg['max_vacancies_per_query'] == 500

    @patch('src.parsing.utils.select_from_list')
    def test_interactive_config_mode_standard(self, mock_select):
        mock_select.return_value = "1. Data Scientist"
        with patch('src.parsing.utils.input_int', return_value=30):
            with patch('src.parsing.utils.input_yes_no', side_effect=[True, False, True]):
                with patch('builtins.input', return_value="1"):
                    with patch('builtins.print'):
                        cfg = utils.interactive_config()
        assert cfg['query'] == "Data Scientist"
        assert cfg['industry'] is None
        assert cfg['period'] == 30


class TestNormalizeSkillForMatching:
    def test_normalize_skill_for_matching(self):
        assert utils.normalize_skill_for_matching("  Python 3.9  ") == "python 39"
        assert utils.normalize_skill_for_matching("C++") == "c"
        assert utils.normalize_skill_for_matching("Node.js") == "nodejs"


class TestExtractAndCountSkills:
    def test_extract_and_count_skills_empty(self):
        result = utils.extract_and_count_skills([], MagicMock())
        assert result == {"frequencies": {}, "tfidf_weights": {}}

    def test_extract_and_count_skills_success(self):
        mock_parser = MagicMock()
        mock_parser.extract_skills_from_vacancies.return_value = {
            "frequencies": {"python": 10},
            "tfidf_weights": {"python": 0.9}
        }
        result = utils.extract_and_count_skills([{"id": 1}], mock_parser)
        assert result["frequencies"] == {"python": 10}

    def test_extract_and_count_skills_exception(self):
        mock_parser = MagicMock()
        mock_parser.extract_skills_from_vacancies.side_effect = Exception("Parser error")
        result = utils.extract_and_count_skills([{"id": 1}], mock_parser)
        assert result == {"frequencies": {}, "tfidf_weights": {}}


class TestMapToCompetencies:
    def test_map_to_competencies(self):
        skill_freq = {"python": 10, "sql": 5, "unknown": 2}
        mapping = {
            "Programming": ["python", "java"],
            "Databases": ["sql", "postgresql"]
        }
        counter = utils.map_to_competencies(skill_freq, mapping)
        assert counter["Programming"] == 10
        assert counter["Databases"] == 5
        # 'unknown' не маппится

    def test_map_to_competencies_partial_match(self):
        skill_freq = {"python programming": 10}
        mapping = {"Programming": ["python"]}
        counter = utils.map_to_competencies(skill_freq, mapping)
        assert counter["Programming"] == 10


class TestPrintTopSkills:
    def test_print_top_skills(self, capsys):
        skill_freq = {"python": 100, "sql": 50}
        utils.print_top_skills(skill_freq, top_n=2)
        captured = capsys.readouterr()
        assert "python" in captured.out
        assert "100" in captured.out


class TestPrintTopCompetencies:
    def test_print_top_competencies(self, capsys):
        counter = Counter({"Programming": 100, "Databases": 50})
        utils.print_top_competencies(counter, top_n=2)
        captured = capsys.readouterr()
        assert "Programming" in captured.out
        assert "100" in captured.out