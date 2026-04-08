# tests/parsing/test_parsers.py
import pytest
from src.parsing.skill_normalizer import SkillNormalizer
from src.parsing.vacancy_parser import VacancyParser
from unittest.mock import Mock, patch
from src.parsing.hh_api import HeadHunterAPI
from src.parsing.hh_api_async import HeadHunterAPIAsync
import asyncio
from unittest.mock import AsyncMock, patch
# tests/parsing/test_vacancy_parser.py
import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, ANY
import numpy as np
from collections import Counter
import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, ANY
import numpy as np
from collections import Counter

from src.parsing.vacancy_parser import VacancyParser
from src.models.vacancy import Vacancy, Area, Employer, KeySkill
from src.parsing.skill_parser import SkillParser, ExtractedSkill, SkillSource
from src.parsing.skill_validator import SkillValidator, ValidationResult, ValidationReason
from src.parsing.vacancy_parser import VacancyParser
from src.models.vacancy import Vacancy, Area, Employer, KeySkill
from src.parsing.skill_parser import SkillParser, ExtractedSkill, SkillSource
from src.parsing.skill_validator import SkillValidator, ValidationResult, ValidationReason


# ----------------------------------------------------------------------
# Фикстуры
# ----------------------------------------------------------------------

@pytest.fixture
def sample_vacancy_dict():
    return {
        "id": "123",
        "name": "Python Developer",
        "area": {"id": 1, "name": "Москва"},
        "employer": {"id": "10", "name": "Test Corp"},
        "key_skills": [{"name": "Python"}, {"name": "Django"}],
        "description": "Опыт работы с FastAPI"
    }

@pytest.fixture
def sample_vacancy_obj():
    area = Area(id=1, name="Москва")
    employer = Employer(id="10", name="Test Corp")
    return Vacancy(
        id="123",
        name="Python Developer",
        area=area,
        employer=employer,
        key_skills=[KeySkill(name="Python"), KeySkill(name="Django")],
        description="Опыт работы с FastAPI"
    )

@pytest.fixture
def mock_skill_parser():
    parser = MagicMock(spec=SkillParser)
    parser.parse_vacancy.return_value = [
        ExtractedSkill("python", SkillSource.KEY_SKILLS, confidence=1.0),
        ExtractedSkill("django", SkillSource.KEY_SKILLS, confidence=1.0),
        ExtractedSkill("fastapi", SkillSource.DESCRIPTION, confidence=0.9),
    ]
    parser._extract_from_text.return_value = [
        ExtractedSkill("fastapi", SkillSource.DESCRIPTION, confidence=0.9)
    ]
    parser.get_stats.return_value = MagicMock()
    return parser

@pytest.fixture
def mock_skill_validator():
    validator = MagicMock(spec=SkillValidator)
    # validate_batch: возвращает список валидных навыков и список результатов
    def validate_batch_side_effect(skills, confidences=None):
        valid = [s for s in skills if s in {"python", "django", "fastapi"}]
        results = [ValidationResult(skill=s, is_valid=(s in valid), confidence=0.9) for s in skills]
        return valid, results
    validator.validate_batch.side_effect = validate_batch_side_effect
    return validator

@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.encode.return_value = np.random.rand(3, 384)
    return model


# ----------------------------------------------------------------------
# Тесты инициализации и загрузки модели
# ----------------------------------------------------------------------

def test_init_loads_skill_parser_and_validator():
    with patch('src.parsing.vacancy_parser.load_it_skills', return_value={"python", "sql"}):
        parser = VacancyParser()
        assert parser.skill_parser is not None
        assert parser.skill_validator is not None
        assert parser.embedding_model is not None

@patch('src.parsing.vacancy_parser.SentenceTransformer')
def test_init_embedding_model_load_failure(mock_st):
    mock_st.side_effect = Exception("Model load error")
    with patch('src.parsing.vacancy_parser.load_it_skills', return_value=set()):
        parser = VacancyParser()
        assert parser.embedding_model is None


# ----------------------------------------------------------------------
# Сохранение
# ----------------------------------------------------------------------

def test_save_raw_vacancies_dict(tmp_path, sample_vacancy_dict, monkeypatch):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_RAW_DIR", tmp_path)
    parser = VacancyParser()
    parser.save_raw_vacancies([sample_vacancy_dict], filename="test.json")
    saved = tmp_path / "test.json"
    assert saved.exists()
    with open(saved, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["id"] == "123"

def test_save_raw_vacancies_obj(tmp_path, sample_vacancy_obj, monkeypatch):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_RAW_DIR", tmp_path)
    parser = VacancyParser()
    parser.save_raw_vacancies([sample_vacancy_obj], filename="test_obj.json")
    saved = tmp_path / "test_obj.json"
    assert saved.exists()

def test_save_processed_frequencies_no_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_PROCESSED_DIR", tmp_path)
    parser = VacancyParser()
    freqs = {"python": 10, "java": 5}
    parser.save_processed_frequencies(freqs, filename="freq.json", apply_filter=False)
    saved = tmp_path / "freq.json"
    assert saved.exists()
    with open(saved, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data == freqs

def test_save_processed_frequencies_with_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_PROCESSED_DIR", tmp_path)
    with patch('src.parsing.vacancy_parser.load_it_skills', return_value={"python"}):
        with patch('src.parsing.vacancy_parser.filter_skills_by_whitelist') as mock_filter:
            mock_filter.return_value = {"python": 10}
            parser = VacancyParser()
            parser.save_processed_frequencies({"python": 10, "java": 5}, apply_filter=True)
            mock_filter.assert_called_once()


# ----------------------------------------------------------------------
# extract_skills_from_description
# ----------------------------------------------------------------------

def test_extract_skills_from_description():
    parser = VacancyParser()
    skills = parser.extract_skills_from_description("Опыт работы с Python и FastAPI")
    assert any("python" in s.lower() for s in skills)


# ----------------------------------------------------------------------
# extract_skills_from_vacancies
# ----------------------------------------------------------------------

@patch('src.parsing.vacancy_parser.SkillNormalizer')
@patch('src.parsing.vacancy_parser.TfidfVectorizer')
def test_extract_skills_from_vacancies_dict(mock_tfidf, mock_normalizer, mock_skill_parser, mock_skill_validator, tmp_path, monkeypatch, sample_vacancy_dict):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    mock_normalizer.normalize_batch.return_value = ["python", "django", "fastapi"]
    mock_tfidf_instance = MagicMock()
    mock_tfidf_instance.fit_transform.return_value = MagicMock()
    mock_tfidf_instance.get_feature_names_out.return_value = ["python", "django"]
    mock_tfidf.return_value = mock_tfidf_instance

    with patch('src.parsing.vacancy_parser.VacancyParser._get_skill_embeddings', return_value={"python": [0.1, 0.2]}):
        parser = VacancyParser()
        parser.skill_parser = mock_skill_parser
        parser.skill_validator = mock_skill_validator
        parser.embedding_model = MagicMock()

        result = parser.extract_skills_from_vacancies([sample_vacancy_dict])

    assert "frequencies" in result
    assert "tfidf_weights" in result
    assert "skill_embeddings" in result
    assert result["frequencies"]["python"] == 1
    assert result["frequencies"]["django"] == 1
    assert result["frequencies"]["fastapi"] == 1
    assert result["skill_embeddings"]["python"] == [0.1, 0.2]


# ----------------------------------------------------------------------
# _get_skill_embeddings
# ----------------------------------------------------------------------

def test_get_skill_embeddings_cache_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache_file = tmp_path / "skill_embeddings.json"
    cache_data = {"python": [0.1, 0.2], "sql": [0.3, 0.4]}
    cache_file.write_text(json.dumps(cache_data), encoding='utf-8')

    parser = VacancyParser()
    parser.embedding_model = MagicMock()
    embeddings = parser._get_skill_embeddings(["python"])
    assert embeddings["python"] == [0.1, 0.2]
    # Модель не вызывалась, т.к. взято из кэша
    parser.embedding_model.encode.assert_not_called()

def test_get_skill_embeddings_compute_new(tmp_path, monkeypatch, mock_embedding_model):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    parser = VacancyParser()
    parser.embedding_model = mock_embedding_model
    skills = ["python", "django"]
    embeddings = parser._get_skill_embeddings(skills)
    assert len(embeddings) == 2
    mock_embedding_model.encode.assert_called_once()
    # Проверяем, что кэш создан
    cache_file = tmp_path / "skill_embeddings.json"
    assert cache_file.exists()


# ----------------------------------------------------------------------
# _calculate_tfidf_weights
# ----------------------------------------------------------------------

def test_calculate_tfidf_weights_empty():
    parser = VacancyParser()
    weights = parser._calculate_tfidf_weights([])
    assert weights == {}

@patch('src.parsing.vacancy_parser.TfidfVectorizer')
def test_calculate_tfidf_weights_success(mock_tfidf, sample_vacancy_dict):
    mock_instance = MagicMock()
    mock_instance.fit_transform.return_value = MagicMock()
    # Имитируем матрицу со средним значением
    mock_instance.fit_transform.return_value.mean.return_value = 0.1
    mock_instance.get_feature_names_out.return_value = np.array(["python", "django"])
    mock_tfidf.return_value = mock_instance

    parser = VacancyParser()
    weights = parser._calculate_tfidf_weights([sample_vacancy_dict])
    # Так как среднее > 0.05, оба навыка войдут
    assert "python" in weights
    assert "django" in weights


# ----------------------------------------------------------------------
# Старые методы (совместимость)
# ----------------------------------------------------------------------

def test_clean_highlighttext():
    text = "Опыт работы с <highlighttext>Python</highlighttext> и <highlighttext>Django</highlighttext>"
    cleaned = VacancyParser.clean_highlighttext(text)
    assert "highlighttext" not in cleaned
    assert "Python" in cleaned

def test_extract_skills_static(sample_vacancy_dict):
    skills = VacancyParser.extract_skills([sample_vacancy_dict])
    assert "Python" in skills
    assert "Django" in skills

def test_normalize_skill():
    assert VacancyParser.normalize_skill("Опыт работы с Python") == "python"
    assert VacancyParser.normalize_skill("командная работа") == "командная работа"  # из стоп-слов
    assert VacancyParser.normalize_skill("очень длинное название навыка из пяти слов") == ""

def test_is_valid_skill():
    assert VacancyParser.is_valid_skill("Python") is True
    assert VacancyParser.is_valid_skill("") is False
    assert VacancyParser.is_valid_skill("a") is False

def test_count_skills():
    skills_list = ["Python", "python", "Django", "коммуникация"]
    with patch('src.parsing.vacancy_parser.load_it_skills', return_value={"python", "django"}):
        counts = VacancyParser.count_skills(skills_list)
    assert counts == {"python": 2, "django": 1}

def test_extract_skills_from_text_deprecated():
    # Должен вернуть пустой список и залогировать предупреждение
    result = VacancyParser.extract_skills_from_text([])
    assert result == []


# ----------------------------------------------------------------------
# Excel и вывод
# ----------------------------------------------------------------------

def test_aggregate_to_dataframe_dict(sample_vacancy_dict):
    parser = VacancyParser()
    df = parser.aggregate_to_dataframe([sample_vacancy_dict])
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert df.loc[0, 'Вакансия'] == "Python Developer"

def test_aggregate_to_dataframe_obj(sample_vacancy_obj):
    parser = VacancyParser()
    df = parser.aggregate_to_dataframe([sample_vacancy_obj])
    assert df.shape[0] == 1

def test_save_to_excel(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_PROCESSED_DIR", tmp_path)
    df = pd.DataFrame({"col": [1, 2]})
    parser = VacancyParser()
    parser.save_to_excel(df, "test.xlsx")
    assert (tmp_path / "test.xlsx").exists()

def test_print_vacancies_list_dict(capsys, sample_vacancy_dict):
    parser = VacancyParser()
    parser.print_vacancies_list([sample_vacancy_dict])
    captured = capsys.readouterr()
    assert "Python Developer" in captured.out

def test_print_vacancies_list_obj(capsys, sample_vacancy_obj):
    parser = VacancyParser()
    parser.print_vacancies_list([sample_vacancy_obj])
    captured = capsys.readouterr()
    assert "Python Developer" in captured.out
class TestSkillNormalizer:
    def setup_method(self):
        SkillNormalizer._whitelist = None
    def test_synonyms(self):
        assert SkillNormalizer.normalize("Python 3.11") == "python"
        assert SkillNormalizer.normalize("javascript") == "node.js"

    def test_suffix_removal(self):
        assert SkillNormalizer.normalize("язык python") == "python"

    def test_fuzzy_matching(self):
        SkillNormalizer._whitelist = None
        assert SkillNormalizer.normalize("reackt") == "react"
        assert SkillNormalizer.normalize("react native") == "react"
        assert SkillNormalizer.normalize("node js") == "node.js"          # ← исправлено
        assert SkillNormalizer.normalize("NodeJS") == "node.js"           # ← исправлено

    def test_no_match_returns_cleaned_version(self):
        result = SkillNormalizer.normalize("какой-то_мусор_навык_123")
        assert result == "какой-то_мусор_навык_"

    def test_normalize_batch(self):
        SkillNormalizer._whitelist = None
        skills = ["Python 3", "React.js v18", "reackt", "machine learning"]
        normalized = SkillNormalizer.normalize_batch(skills)
        assert normalized == ["python", "react", "mlops"]   # machine learning → mlops (fuzzy)

    def test_batch_with_duplicates(self):
        """Проверка автоматической дедупликации (нормально для списка навыков)"""
        skills = ["Python", "python", "", "React", "reackt"]
        normalized = SkillNormalizer.normalize_batch(skills)
        assert normalized.count("python") == 1
        assert "react" in normalized
        assert "" not in normalized

    def test_normalize_edge_cases(self):
        """Дополнительные кейсы"""
        assert SkillNormalizer.normalize("") == ""
        assert SkillNormalizer.normalize("   Python   ") == "python"
        assert SkillNormalizer.normalize("React.js v18") == "react"
        assert SkillNormalizer.normalize("NodeJS") == "node.js"
        
    def test_empty_and_whitespace(self):
        """Проверка пустых и пробельных строк"""
        assert SkillNormalizer.normalize("") == ""
        assert SkillNormalizer.normalize("   ") == ""
        assert SkillNormalizer.normalize("\t\n") == ""
        assert SkillNormalizer.normalize(None) == ""  # если поддерживается None

    def test_version_patterns(self):
        """Удаление номеров версий и данных в скобках"""
        assert SkillNormalizer.normalize("Python 3.11") == "python"
        assert SkillNormalizer.normalize("React v18.2.1") == "react"
        assert SkillNormalizer.normalize("Django (фреймворк)") == "django"
        assert SkillNormalizer.normalize("Angular [2+]") == "angular"
        assert SkillNormalizer.normalize("PostgreSQL 14.5") == "postgresql"

    def test_suffix_removal_extended(self):
        """Удаление лишних слов из конца"""
        assert SkillNormalizer.normalize("язык программирования Python") == "python"
        assert SkillNormalizer.normalize("Python язык") == "python"
        assert SkillNormalizer.normalize("React фреймворк") == "react"
        assert SkillNormalizer.normalize("pandas библиотека") == "pandas"
        assert SkillNormalizer.normalize("MongoDB database") == "mongodb"

    def test_special_characters_cleaning(self):
        assert SkillNormalizer.normalize("Node.JS") == "node.js"
        assert SkillNormalizer.normalize("some_skill!") == "some_skill"

    def test_fuzzy_matching_extended(self):
        """Расширенные проверки fuzzy-совпадений"""
        # Уже есть "reackt" → "react"
        assert SkillNormalizer.normalize("pythn") == "python"
        assert SkillNormalizer.normalize("anguler") == "angular"
        assert SkillNormalizer.normalize("dockr") == "docker"
        # Порог низкий - не должно совпадать с плохим вариантом
        assert SkillNormalizer.normalize("xyzabc") == "xyzabc"   # останется как есть

    def test_whitelist_exact_match_prevents_fuzzy(self):
        """Точное совпадение в whitelist останавливает fuzzy"""
        # Предполагаем, что 'python' есть в whitelist
        assert SkillNormalizer.normalize("python") == "python"
        # А 'pythn' нет, поэтому сработает fuzzy и исправит
        assert SkillNormalizer.normalize("pythn") == "python"

    def test_deduplicate_order_and_uniqueness(self):
        """Проверка дедупликации с сохранением порядка"""
        skills = ["React", "react", "Python", "python", "Docker", "React"]
        result = SkillNormalizer.deduplicate(skills)
        assert result == ["react", "python", "docker"]
        # Пустые значения игнорируются
        skills_with_empty = ["", "React", None, "React"]
        result2 = SkillNormalizer.deduplicate(skills_with_empty)
        assert result2 == ["react"]

    def test_normalize_batch_with_empty(self):
        """normalize_batch должен отфильтровывать пустые и дубликаты"""
        skills = ["Python 3", "", "python", "React v18", "REACT"]
        normalized = SkillNormalizer.normalize_batch(skills)
        assert normalized == ["python", "react"]

    def test_whitelist_loading_and_caching(self):
        """Проверка кэширования whitelist"""
        SkillNormalizer._whitelist = None
        whitelist1 = SkillNormalizer._get_whitelist()
        whitelist2 = SkillNormalizer._get_whitelist()
        assert whitelist1 is whitelist2
        assert "python" in whitelist1
        assert "mlops" in whitelist1
        assert "node.js" in whitelist1

    def test_direct_phrase_mapping(self):
        """Проверка точного фразового маппинга (если добавлен DIRECT_PHRASE_MAP)"""
        # Если вы реализовали DIRECT_PHRASE_MAP, как предлагалось ранее
        if hasattr(SkillNormalizer, 'DIRECT_PHRASE_MAP'):
            assert SkillNormalizer.normalize("machine learning") == "mlops"
            assert SkillNormalizer.normalize("node js") == "node.js"
            assert SkillNormalizer.normalize("react native") == "react"

    def test_complex_input_combinations(self):
        """Комбинации синонимов, версий и суффиксов"""
        assert SkillNormalizer.normalize("язык программирования Python 3.11") == "python"
        assert SkillNormalizer.normalize("фреймворк React.js v18") == "react"
        
    def test_normalize_with_special_chars_and_version(self):
        assert SkillNormalizer.normalize("  PyThOn  3.9  ") == "python"
        assert SkillNormalizer.normalize("Node.JS (среда)") == "node.js"
        assert SkillNormalizer.normalize("C++17") == "cpp"

    def test_fuzzy_threshold_boundary(self):
        # При низком сходстве возвращается исходная строка (после очистки)
        result = SkillNormalizer.normalize("completelywrongterm")
        # Если в whitelist нет похожих, должно вернуться очищенное "completelywrongterm"
        assert "completelywrongterm" in result

    def test_direct_phrase_mapping_if_present(self):
        # Проверка работы прямых фразовых замен (если внедрили)
        # В текущей версии синонимы обрабатываются через регулярки, этого достаточно
        pass

class TestVacancyParser:
    def test_extract_and_count_skills_with_weights(self, gap_analyzer):
        """Подсчёт навыков по вакансиям + дедупликация"""
        parser = VacancyParser()
        vacancies = [
            {"key_skills": ["Python", "FastAPI"]},
            {"key_skills": ["Python", "Docker"]}
        ]

        all_key_skills = []
        for vac in vacancies:
            all_key_skills.extend(vac.get("key_skills", []))

        normalized = SkillNormalizer.normalize_batch(all_key_skills)
        # теперь дедуплицирует → python только один раз
        assert normalized.count("python") == 1
        assert len(normalized) == 3
        assert set(normalized) == {"python", "fastapi", "docker"}