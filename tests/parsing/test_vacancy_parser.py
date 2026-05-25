# tests/parsing/test_vacancy_parser.py
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from src.models.vacancy import Area, Employer, KeySkill, Vacancy
from src.parsing.skills.skill_parser import ExtractedSkill, SkillSource
from src.parsing.skills.vacancy_parser import VacancyParser


@pytest.fixture
def parser_with_mocks():
    """Parser с замоканными зависимостями."""
    parser = VacancyParser()
    parser.skill_parser = MagicMock()
    parser.skill_validator = MagicMock()
    parser.bm25_ranker = MagicMock()
    parser.embedding_cache = MagicMock()
    parser.hybrid_calc = MagicMock()
    return parser


class TestExtractSkillsFromDescription:
    def test_empty_description(self, parser_with_mocks):
        assert parser_with_mocks.extract_skills_from_description("") == []

    def test_calls_parser(self, parser_with_mocks):
        parser_with_mocks.skill_parser._extract_from_text.return_value = [
            ExtractedSkill("python", SkillSource.DESCRIPTION),
            ExtractedSkill("sql", SkillSource.DESCRIPTION),
        ]
        skills = parser_with_mocks.extract_skills_from_description("Python and SQL")
        assert skills == ["python", "sql"]
        parser_with_mocks.skill_parser._extract_from_text.assert_called_once_with(
            "Python and SQL", source=SkillSource.DESCRIPTION
        )


class TestExtractSkillsFromVacancies:
    def test_integration_with_mocks(self, parser_with_mocks):
        # Настройка моков
        parser_with_mocks.skill_parser.parse_vacancy.return_value = [
            ExtractedSkill("python", SkillSource.KEY_SKILLS, 1.0),
            ExtractedSkill("sql", SkillSource.DESCRIPTION, 0.9),
        ]
        parser_with_mocks.skill_validator.validate_batch.return_value = (
            ["python", "sql"], [MagicMock(is_valid=True)] * 2
        )
        # Для одиночной валидации тоже замокаем
        parser_with_mocks.skill_validator.validate.return_value = MagicMock(is_valid=True)
        parser_with_mocks.hybrid_calc.calculate.return_value = {"python": 0.8, "sql": 0.6}
        parser_with_mocks.embedding_cache.get_embeddings.return_value = {
            "python": np.array([0.1, 0.2]),
            "sql": np.array([0.3, 0.4]),
        }

        # Создаём тестовую вакансию
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(
            id="1", name="Dev", area=area, employer=employer,
            key_skills=[KeySkill("Python")], description="SQL required"
        )
        vacancies = [vac]

        result = parser_with_mocks.extract_skills_from_vacancies(vacancies)

        assert "frequencies" in result
        assert result["frequencies"]["python"] == 1
        assert result["frequencies"]["sql"] == 1

    def test_validation_thread_pool(self, parser_with_mocks):
        # Навыков >200, должен использоваться ThreadPoolExecutor
        parser_with_mocks.skill_parser.parse_vacancy.return_value = []
        # Готовим список навыков
        all_skills = [f"skill_{i}" for i in range(250)]
        parser_with_mocks.skill_validator.validate.return_value = MagicMock(is_valid=True)
        # Создаём одну вакансию, чтобы запустить подсчёт
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="1", name="Dev", area=area, employer=employer, key_skills=[], description="")
        # Мокаем parse_vacancy, чтобы он вернул 250 ExtractedSkill
        parser_with_mocks.skill_parser.parse_vacancy.return_value = [
            ExtractedSkill(s, SkillSource.KEY_SKILLS, 1.0) for s in all_skills
        ]
        parser_with_mocks.skill_validator.validate_batch.side_effect = None  # чтобы не сработал
        parser_with_mocks.hybrid_calc.calculate.return_value = {}
        parser_with_mocks.embedding_cache.get_embeddings.return_value = {}

        result = parser_with_mocks.extract_skills_from_vacancies([vac])
        assert "frequencies" in result


class TestSaveMethods:
    def test_save_raw_vacancies(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.parsing.skills.vacancy_parser.config.DATA_RAW_DIR", tmp_path)
        parser = VacancyParser()
        parser.save_raw_vacancies([{"id": "1"}], filename="test.json")
        assert (tmp_path / "test.json").exists()

    def test_save_processed_frequencies_with_filter(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.parsing.skills.vacancy_parser.config.DATA_PROCESSED_DIR", tmp_path)
        parser = VacancyParser()
        with patch("src.parsing.skills.vacancy_parser.load_it_skills", return_value={"python"}), \
             patch("src.parsing.skills.vacancy_parser.filter_skills_by_whitelist") as mock_filter:
            mock_filter.return_value = {"python": 10}
            parser.save_processed_frequencies({"python": 10, "java": 5}, apply_filter=True)
            mock_filter.assert_called_once()
        assert (tmp_path / "competency_frequency.json").exists()


class TestStaticMethods:
    def test_strip_html(self):
        assert VacancyParser._strip_html("<p>text</p>") == "text"
        assert VacancyParser._strip_html("") == ""
        assert VacancyParser._strip_html(None) == ""

    def test_clean_highlighttext(self):
        assert VacancyParser.clean_highlighttext("<highlighttext>Python</highlighttext>") == "Python"
        assert VacancyParser.clean_highlighttext("") == ""
        assert VacancyParser.clean_highlighttext(None) == ""


class TestExcel:
    def test_aggregate_to_dataframe(self, parser_with_mocks):
        parser_with_mocks.skill_parser.parse_vacancy.return_value = []
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="1", name="Dev", area=area, employer=employer,
                      key_skills=[KeySkill("Python")])
        df = parser_with_mocks.aggregate_to_dataframe([vac])
        assert df.shape[0] == 1
        assert "python" in df.loc[0, "Навыки"]

    def test_save_to_excel(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.parsing.skills.vacancy_parser.config.DATA_RESULT_DIR", tmp_path)
        parser = VacancyParser()
        df = pd.DataFrame({"col": [1, 2]})
        parser.save_to_excel(df, "test.xlsx")
        assert (tmp_path / "test.xlsx").exists()


class TestPrintVacancies:
    def test_print_vacancies_list(self, capsys, parser_with_mocks):
        parser_with_mocks.skill_parser.parse_vacancy.return_value = [
            ExtractedSkill("python", SkillSource.KEY_SKILLS)
        ]
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="1", name="Dev", area=area, employer=employer,
                      key_skills=[KeySkill("Python")])
        parser_with_mocks.print_vacancies_list([vac])
        captured = capsys.readouterr()
        assert "Dev" in captured.out
        assert "python" in captured.out.lower()

    def test_print_vacancies_list_dict(self, capsys, parser_with_mocks):
        parser_with_mocks.skill_parser.parse_vacancy.return_value = []
        parser_with_mocks.print_vacancies_list(
            [{"id": "1", "name": "Job", "employer": {"name": "Corp"}, "area": {"name": "MSK"}, "key_skills": []}]
        )
        captured = capsys.readouterr()
        assert "Job" in captured.out

def test_extract_skills_from_vacancies_mixed(parser_with_mocks):
    # смесь dict и Vacancy, проверка, что невалидный dict пропускается
    parser_with_mocks.skill_parser.parse_vacancy.return_value = []
    parser_with_mocks.skill_validator.validate_batch.return_value = ([], [])
    parser_with_mocks.hybrid_calc.calculate.return_value = {}
    parser_with_mocks.embedding_cache.get_embeddings.return_value = {}

    invalid_dict = {"id": "bad"}
    area = Area(1, "MSK")
    employer = Employer("1", "Corp")
    vac = Vacancy(id="good", name="Job", area=area, employer=employer,
                  key_skills=[KeySkill("Python")])

    with patch("src.models.vacancy.Vacancy.from_api", side_effect=[ValueError, vac]):
        result = parser_with_mocks.extract_skills_from_vacancies([invalid_dict, vac])
    assert "frequencies" in result

def test_extract_skills_thread_pool_activated(parser_with_mocks, monkeypatch):
    # Более 200 навыков → ThreadPoolExecutor. Нормализация возвращает все навыки без изменений
    monkeypatch.setattr("src.parsing.skills.vacancy_parser.SkillNormalizer.normalize_batch", lambda x: x)
    parser_with_mocks.skill_parser.parse_vacancy.return_value = [
        ExtractedSkill(f"skill_{i}", SkillSource.KEY_SKILLS, 1.0) for i in range(250)
    ]
    parser_with_mocks.skill_validator.validate.return_value = MagicMock(is_valid=True)
    parser_with_mocks.hybrid_calc.calculate.return_value = {}
    parser_with_mocks.embedding_cache.get_embeddings.return_value = {}
    area = Area(1, "MSK")
    employer = Employer("1", "Corp")
    vac = Vacancy(id="1", name="Big", area=area, employer=employer, key_skills=[])
    result = parser_with_mocks.extract_skills_from_vacancies([vac])
    assert "frequencies" in result
    assert len(result["frequencies"]) == 250

def test_aggregate_to_dataframe_with_dict_vacancy(parser_with_mocks):
    """Строки 142-155: обработка словаря вакансии"""
    parser_with_mocks.skill_parser.parse_vacancy.return_value = []
    vac_dict = {
        "id": "1",
        "name": "Dev",
        "employer": {"name": "Corp"},
        "area": {"name": "MSK"},
        "key_skills": [{"name": "Python"}],
        "description": "SQL",
        "snippet": {"requirement": "Django", "responsibility": ""},
    }
    df = parser_with_mocks.aggregate_to_dataframe([vac_dict])
    assert df.shape[0] == 1
    # Навыки должны включать Python и SQL
    skills_str = df.loc[0, "Навыки"].lower()
    assert "python" in skills_str

def test_aggregate_to_dataframe_missing_employer(parser_with_mocks):
    """Строки 162-163: employer = None в словаре"""
    parser_with_mocks.skill_parser.parse_vacancy.return_value = []
    vac_dict = {
            "id": "1",
            "name": "Dev",
            "employer": None,
            "area": {"name": "MSK"},
            "key_skills": [],
            "description": "",
            "snippet": {},
    }
    df = parser_with_mocks.aggregate_to_dataframe([vac_dict])
    assert df.loc[0, "Компания"] == "Unknown"

def test_aggregate_to_dataframe_with_quality_report_spam(parser_with_mocks):
    """Добавление колонок 'Спам' и 'Причина спама' через quality_report"""
    parser_with_mocks.skill_parser.parse_vacancy.return_value = []
    vac = {
        "id": "1",
        "name": "Test",
        "employer": {"name": "Corp"},
        "area": {"name": "MSK"},
        "key_skills": [],
        "description": "",
        "snippet": {},
    }
    report = {
        "spam_vacancies": [
            {"id": "1", "name": "Test", "employer": "Corp", "score": 0.3,
             "flags": [{"reason": "NO_SKILLS", "detail": "No key skills"}]}
        ]
    }
    df = parser_with_mocks.aggregate_to_dataframe([vac], quality_report=report)
    assert "Спам" in df.columns
    assert "Причина спама" in df.columns
    assert df.loc[0, "Спам"] == "Да"

def test_aggregate_to_dataframe_with_quality_report_clean(parser_with_mocks):
    """Чистая вакансия в quality_report — колонка 'Спам' = 'Нет'"""
    parser_with_mocks.skill_parser.parse_vacancy.return_value = []
    vac = {
        "id": "1",
        "name": "Test",
        "employer": {"name": "Corp"},
        "area": {"name": "MSK"},
        "key_skills": [],
        "description": "",
        "snippet": {},
    }
    report = {
        "spam_vacancies": [
            {"id": "1", "name": "Test", "employer": "Corp", "score": 0.8,
             "flags": [{"reason": "OK", "detail": ""}]}
        ]
    }
    df = parser_with_mocks.aggregate_to_dataframe([vac], quality_report=report)
    assert df.loc[0, "Спам"] == "Нет"


def test_save_processed_frequencies_without_filter(tmp_path, monkeypatch):
    """Строки 217-218: apply_filter=False"""
    monkeypatch.setattr("src.parsing.skills.vacancy_parser.config.DATA_PROCESSED_DIR", tmp_path)
    parser = VacancyParser()
    with patch("src.parsing.skills.vacancy_parser.atomic_write_json") as mock_write:
        parser.save_processed_frequencies({"python": 10}, apply_filter=False)
        mock_write.assert_called_once_with({"python": 10}, tmp_path / "competency_frequency.json")
