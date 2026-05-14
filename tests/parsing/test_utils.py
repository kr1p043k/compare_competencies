# tests/parsing/test_utils.py
import json
import logging
import pickle
from collections import Counter
from unittest.mock import MagicMock, patch
import pytest

from src import config
from src.parsing import utils as parsing_utils
from src import utils as base_utils

# Чтобы существующие тесты продолжали работать без изменений,
# назначаем utils = parsing_utils.
utils = parsing_utils


class TestReadWriteJson:
    def test_read_json_success(self, tmp_path):
        data = {"key": "value"}
        filepath = tmp_path / "test.json"
        filepath.write_text(json.dumps(data), encoding="utf-8")
        result = utils.read_json(filepath)
        assert result == data

    def test_read_json_file_not_found(self, tmp_path):
        filepath = tmp_path / "missing.json"
        result = utils.read_json(filepath)
        assert result is None

    def test_read_json_invalid_json(self, tmp_path):
        filepath = tmp_path / "invalid.json"
        filepath.write_text("{not json", encoding="utf-8")
        result = utils.read_json(filepath)
        assert result is None

    def test_write_json_success(self, tmp_path):
        data = {"key": "value"}
        filepath = tmp_path / "out.json"
        utils.write_json(data, filepath)
        assert filepath.exists()
        loaded = json.loads(filepath.read_text(encoding="utf-8"))
        assert loaded == data

    def test_write_json_handles_error(self, tmp_path):
        filepath = tmp_path / "dir"
        filepath.mkdir()
        data = {"key": "value"}
        utils.write_json(data, filepath)


class TestSetupLogging:
    def test_setup_logging(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "LOG_FILE", tmp_path / "test.log")
        utils.setup_logging()
        logger = logging.getLogger()
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


class TestLoadItSkills:
    def test_load_it_skills_success(self, tmp_path, monkeypatch):
        skills_file = tmp_path / "it_skills.json"
        skills_list = ["Python", "JavaScript", "  React  "]
        skills_file.write_text(json.dumps(skills_list), encoding="utf-8")
        monkeypatch.setattr(config, "IT_SKILLS_PATH", tmp_path / "it_skills.json")
        result = utils.load_it_skills()
        assert result == {"python", "javascript", "react"}

    def test_load_it_skills_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IT_SKILLS_PATH", tmp_path / "it_skills.json")
        result = utils.load_it_skills()
        assert result == set()

    def test_load_it_skills_not_a_list(self, tmp_path, monkeypatch):
        skills_file = tmp_path / "it_skills.json"
        skills_file.write_text('{"not": "list"}', encoding="utf-8")
        monkeypatch.setattr(config, "IT_SKILLS_PATH", tmp_path / "it_skills.json")
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
        mock_hh_api.last_response = {"found": 10, "pages": 1}
        mock_hh_api.search_vacancies.return_value = [{"id": "1"}, {"id": "2"}]
        with patch("time.sleep", return_value=None):
            vacs = utils.collect_vacancies_multiple(
                mock_hh_api,
                queries=["Python"],
                area_ids=[1],
                period_days=30,
                max_pages=5,
                industry=None,
                max_vacancies_per_query=20,
            )
        assert len(vacs) == 2


class TestLoadQueriesFromFile:
    def test_load_queries_from_file(self, tmp_path):
        filepath = tmp_path / "queries.txt"
        filepath.write_text("Python\nJava\n  SQL  \n", encoding="utf-8")
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
        utils.safe_print("Hello")


class TestInputFunctions:
    def test_input_int_default(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        result = utils.input_int("prompt", default=5)
        assert result == 5

    def test_input_int_valid(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "10")
        result = utils.input_int("prompt", default=5, min_val=1, max_val=20)
        assert result == 10

    def test_input_int_out_of_range_then_valid(self, monkeypatch):
        inputs = iter(["100", "15"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = utils.input_int("prompt", default=5, min_val=1, max_val=20)
        assert result == 15

    def test_input_yes_no_default(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert utils.input_yes_no("prompt", default=True) is True
        assert utils.input_yes_no("prompt", default=False) is False

    def test_input_yes_no_yes(self, monkeypatch):
        for ans in ["y", "yes", "да"]:
            monkeypatch.setattr("builtins.input", lambda _, a=ans: a)
            assert utils.input_yes_no("prompt") is True

    def test_input_yes_no_no(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        assert utils.input_yes_no("prompt") is False

    def test_select_from_list(self, monkeypatch):
        items = ["A", "B", "C"]
        monkeypatch.setattr("builtins.input", lambda _: "2")
        result = utils.select_from_list(items, "Choose:")
        assert result == "B"

    def test_select_from_list_invalid_then_valid(self, monkeypatch):
        items = ["A", "B", "C"]
        inputs = iter(["abc", "3"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = utils.select_from_list(items, "Choose:")
        assert result == "C"


class TestInteractiveConfig:
    @patch("src.parsing.utils.select_from_list")
    @patch("src.parsing.utils.input_int")
    @patch("src.parsing.utils.input_yes_no")
    @patch("builtins.input")
    def test_interactive_config_mode_10(self, mock_input, mock_yes_no, mock_int, mock_select):
        mock_select.return_value = "10. Другое (ввести свой запрос)"
        mock_input.return_value = "Custom Query"
        mock_int.return_value = 15
        mock_yes_no.side_effect = [True, False, True, True]
        with patch("builtins.print"):
            cfg = utils.interactive_config()
        assert cfg["query"] == "Custom Query"

    @patch("builtins.input")
    @patch("src.parsing.utils.select_from_list")
    @patch("src.parsing.utils.input_yes_no")
    def test_interactive_config_mode_11_it_sector(self, mock_yes_no, mock_select, mock_input):
        mock_select.return_value = "11. Поиск по всему IT-сектору (industry=7)"
        mock_yes_no.return_value = False
        mock_input.return_value = "1"
        with patch("builtins.print"):
            cfg = utils.interactive_config()
        assert cfg["is_it_sector"] is True
        assert cfg["industry"] == 7

    @patch("src.parsing.utils.select_from_list")
    def test_interactive_config_mode_standard(self, mock_select):
        mock_select.return_value = "1. Data Scientist"
        with (
            patch("src.parsing.utils.input_int", return_value=30),
            patch("src.parsing.utils.input_yes_no", side_effect=[True, False, True, False]),
            patch("builtins.input", return_value="1"),
            patch("builtins.print"),
        ):
            cfg = utils.interactive_config()
        assert cfg["query"] == "Data Scientist"


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
            "tfidf_weights": {"python": 0.9},
        }
        result = utils.extract_and_count_skills([{"id": 1}], mock_parser)
        assert result["frequencies"] == {"python": 10}

    def test_extract_and_count_skills_exception(self):
        mock_parser = MagicMock()
        mock_parser.extract_skills_from_vacancies.side_effect = Exception("Parser error")
        result = utils.extract_and_count_skills([{"id": 1}], mock_parser)
        assert result == {"frequencies": {}, "tfidf_weights": {}}


class TestPrintTopCompetencies:
    def test_print_top_competencies(self, capsys):
        counter = Counter({"Programming": 100, "Databases": 50})
        utils.print_top_competencies(counter, top_n=2)
        captured = capsys.readouterr()
        assert "Programming" in captured.out
        assert "100" in captured.out


# --- дополнительные тесты для покрытия пропущенных строк ---

def test_read_json_permission_error(tmp_path):
    """Строки 90-92: ошибка чтения JSON"""
    filepath = tmp_path / "test.json"
    filepath.write_text('{"key": "value"}', encoding="utf-8")
    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        result = utils.read_json(filepath)
        assert result is None


def test_date_chunks():
    """Строки 473-479: разбивка периода на чанки"""
    chunks = utils.date_chunks(20, chunk_size=5)
    assert len(chunks) == 4


def test_date_chunks_short_period():
    """Период меньше размера чанка"""
    chunks = utils.date_chunks(3, chunk_size=5)
    assert len(chunks) == 1


def test_collect_vacancies_multiple_large_found(monkeypatch):
    """Строки 166, 168-192: разбивка по датам при >2000 вакансий"""
    mock_hh_api = MagicMock()
    mock_hh_api.last_response = {"found": 3000, "pages": 50}
    mock_hh_api.search_vacancies.return_value = [{"id": f"{i}"} for i in range(10)]
    monkeypatch.setattr(utils, "date_chunks", lambda days, chunk_size: [(100, 200), (200, 300)])
    with patch("time.sleep", return_value=None):
        vacs = utils.collect_vacancies_multiple(
            mock_hh_api, queries=["Python"], area_ids=[1], period_days=60,
            max_pages=5, max_vacancies_per_query=5
        )
    assert len(vacs) == 5


def test_interactive_config_region_parsing():
    """Строки 219-220, 233-234: парсинг регионов"""
    with (
        patch("src.parsing.utils.select_from_list", return_value="1. Data Scientist"),
        patch("src.parsing.utils.input_int", return_value=15),
        patch("src.parsing.utils.input_yes_no", side_effect=[True, False, True, False]),
        patch("builtins.input", return_value="1 2"),
        patch("builtins.print"),
    ):
        cfg = utils.interactive_config()
    assert len(cfg["area_ids"]) >= 1


def test_extract_and_count_skills_restore():
    """Строка 343: восстановление после ошибки"""
    mock_parser = MagicMock()
    mock_parser.extract_skills_from_vacancies.side_effect = Exception("Boom")
    result = utils.extract_and_count_skills([{"id": 1}], mock_parser)
    assert result == {"frequencies": {}, "tfidf_weights": {}}


def test_filter_skills_by_whitelist_empty_whitelist():
    from src.parsing.utils import filter_skills_by_whitelist
    d = {"python": 3, "java": 2}
    assert filter_skills_by_whitelist(d, set()) == d


def test_date_chunks_already_tested():
    from src.parsing.utils import date_chunks
    chunks = date_chunks(10, chunk_size=3)
    assert len(chunks) == 4


def test_print_top_skills(capsys):
    from src.parsing.utils import print_top_skills
    print_top_skills({"python": 10, "java": 5}, top_n=2)
    captured = capsys.readouterr()
    assert "python" in captured.out


def test_map_to_competencies(mocker):
    from src.parsing.utils import map_to_competencies
    mocker.patch("src.parsing.utils.logger")
    freq = {"python": 5, "sql": 3}
    mapping = {"comp1": ["python"], "comp2": ["sql"]}
    res = map_to_competencies(freq, mapping)
    assert res["comp1"] == 5
    assert res["comp2"] == 3


def test_collect_vacancies_multiple_max_vacancies_limit():
    """Строки 152, 154: ограничение max_vacancies_per_query"""
    mock_hh_api = MagicMock()
    mock_hh_api.last_response = {"found": 1000, "pages": 10}
    mock_hh_api.search_vacancies.return_value = [{"id": str(i)} for i in range(100)]
    with patch("time.sleep", return_value=None):
        vacs = utils.collect_vacancies_multiple(
            mock_hh_api, queries=["Python"], area_ids=[1], period_days=30,
            max_pages=10, max_vacancies_per_query=50
        )
    assert len(vacs) <= 50


def test_collect_vacancies_multiple_date_chunk_loop():
    """Строки 177, 206-207: разбивка на чанки и выход при достижении лимита"""
    mock_hh_api = MagicMock()
    mock_hh_api.last_response = {"found": 3000, "pages": 50}
    mock_hh_api.search_vacancies.return_value = [{"id": str(i)} for i in range(50)]
    with patch("src.parsing.utils.date_chunks", return_value=[(100, 200), (200, 300)]), \
         patch("time.sleep", return_value=None):
        vacs = utils.collect_vacancies_multiple(
            mock_hh_api, queries=["Python"], area_ids=[1], period_days=60,
            max_pages=5, max_vacancies_per_query=30
        )
    assert len(vacs) == 30


def test_load_queries_from_file_error(tmp_path):
    """Строки 220-221: ошибка чтения файла запросов"""
    filepath = tmp_path / "queries.txt"
    with patch("builtins.open", side_effect=OSError("error")):
        queries = utils.load_queries_from_file(filepath)
        assert queries == []


def test_interactive_config_region_selection_fallback():
    """Строки 233-234: некорректный ввод региона → область по умолчанию"""
    with patch("src.parsing.utils.select_from_list", return_value="1. Data Scientist"), \
         patch("src.parsing.utils.input_int", return_value=30), \
         patch("src.parsing.utils.input_yes_no", side_effect=[True, False, True, False]), \
         patch("builtins.input", return_value="abc 1"), \
         patch("builtins.print"):
        cfg = utils.interactive_config()
    assert 1 in cfg["area_ids"]


def test_read_json_os_error(tmp_path):
    """Ошибка открытия файла (PermissionError и т.п.)"""
    filepath = tmp_path / "test.json"
    filepath.write_text('{"key":"value"}', encoding="utf-8")
    with patch("builtins.open", side_effect=OSError("Permission denied")):
        result = utils.read_json(filepath)
        assert result is None


def test_collect_vacancies_multiple_date_limit():
    """Строки 206-207: остановка при достижении лимита в цикле по датам"""
    mock_hh_api = MagicMock()
    mock_hh_api.last_response = {"found": 3000, "pages": 50}
    mock_hh_api.search_vacancies.return_value = [{"id": str(i)} for i in range(50)]
    with patch("src.parsing.utils.date_chunks", return_value=[(100, 200), (200, 300)]), \
         patch("time.sleep", return_value=None):
        vacs = utils.collect_vacancies_multiple(
            mock_hh_api, queries=["Python"], area_ids=[1], period_days=60,
            max_pages=5, max_vacancies_per_query=30
        )
    assert len(vacs) == 30


def test_map_to_competencies_unmapped_skill():
    """Ненайденный навык не попадает в результат (строка 353)"""
    from src.parsing.utils import map_to_competencies
    freq = {"python": 5, "unknown_skill": 3}
    mapping = {"comp1": ["python"]}
    res = map_to_competencies(freq, mapping)
    assert "comp1" in res
    assert "unknown_skill" not in res


# --- тесты для функций из src.utils (base_utils) ---

def test_validate_safe_path_inside(tmp_path):
    """Путь внутри разрешённой директории"""
    base = tmp_path
    user_path = "subdir/file.txt"
    result = base_utils.validate_safe_path(user_path, base_dir=base)
    expected = (base / user_path).resolve()
    assert result == expected


def test_validate_safe_path_outside(tmp_path):
    """Путь выходит за пределы – ошибка"""
    base = tmp_path
    user_path = "../outside.txt"
    with pytest.raises(ValueError, match="выходит за пределы"):
        base_utils.validate_safe_path(user_path, base_dir=base)


def test_safe_load_pickle_success(tmp_path):
    """Успешная загрузка pickle из разрешённой директории"""
    data = {"key": "value"}
    filepath = tmp_path / "test.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    result = base_utils.safe_load_pickle(filepath, allowed_dirs=[tmp_path])
    assert result == data
