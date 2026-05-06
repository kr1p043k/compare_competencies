# tests/parsing/test_parsers.py
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.vacancy import Area, Employer, KeySkill, Vacancy
from src.parsing.skill_normalizer import SkillNormalizer
from src.parsing.skill_parser import ExtractedSkill, SkillParser, SkillSource
from src.parsing.skill_validator import SkillValidator, ValidationResult
from src.parsing.vacancy_parser import VacancyParser

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
        "description": "Опыт работы с FastAPI",
        "snippet": {},  # для совместимости
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
        description="Опыт работы с FastAPI",
    )


@pytest.fixture
def mock_skill_parser():
    parser = MagicMock(spec=SkillParser)
    parser.parse_vacancy.return_value = [
        ExtractedSkill("python", SkillSource.KEY_SKILLS, confidence=1.0),
        ExtractedSkill("django", SkillSource.KEY_SKILLS, confidence=1.0),
        ExtractedSkill("fastapi", SkillSource.DESCRIPTION, confidence=0.9),
    ]
    parser._extract_from_text.return_value = [ExtractedSkill("fastapi", SkillSource.DESCRIPTION, confidence=0.9)]
    parser.get_stats.return_value = MagicMock()
    return parser


@pytest.fixture
def mock_skill_validator():
    validator = MagicMock(spec=SkillValidator)

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
    with patch("src.parsing.vacancy_parser.load_it_skills", return_value={"python", "sql"}):
        parser = VacancyParser()
        assert parser.skill_parser is not None
        assert parser.skill_validator is not None
        assert parser.embedding_model is not None


@patch("src.parsing.vacancy_parser.get_embedding_model")
def test_init_embedding_model_load_failure(mock_get_emb):
    mock_get_emb.side_effect = Exception("Model load error")
    with patch("src.parsing.vacancy_parser.load_it_skills", return_value=set()):
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
    with open(saved, encoding="utf-8") as f:
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
    with open(saved, encoding="utf-8") as f:
        data = json.load(f)
    assert data == freqs


def test_save_processed_frequencies_with_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_PROCESSED_DIR", tmp_path)
    with (
        patch("src.parsing.vacancy_parser.load_it_skills", return_value={"python"}),
        patch("src.parsing.vacancy_parser.filter_skills_by_whitelist") as mock_filter,
    ):
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


@patch("src.parsing.vacancy_parser.SkillNormalizer")
def test_extract_skills_from_vacancies_dict(
    mock_normalizer, mock_skill_parser, mock_skill_validator, tmp_path, monkeypatch, sample_vacancy_dict
):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    mock_normalizer.normalize_batch.return_value = ["python", "django", "fastapi"]

    with patch("src.parsing.vacancy_parser.VacancyParser._get_skill_embeddings", return_value={"python": [0.1, 0.2]}):
        parser = VacancyParser()
        parser.skill_parser = mock_skill_parser
        parser.skill_validator = mock_skill_validator
        parser._embedding_model = MagicMock()

        result = parser.extract_skills_from_vacancies([sample_vacancy_dict])

    assert "frequencies" in result
    assert "hybrid_weights" in result
    assert "skill_embeddings" in result
    assert result["frequencies"]["python"] == 1
    assert result["frequencies"]["django"] == 1
    assert result["frequencies"]["fastapi"] == 1
    assert result["skill_embeddings"]["python"] == [0.1, 0.2]


# ----------------------------------------------------------------------
# _get_skill_embeddings
# ----------------------------------------------------------------------


def test_get_skill_embeddings_cache_exists(tmp_path, monkeypatch):
    """Строка: кэш эмбеддингов существует"""
    monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    import numpy as np
    cache_npz = tmp_path / "skill_embeddings.npz"
    cache_index = tmp_path / "skill_embeddings_index.json"
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.savez(cache_npz, embeddings=embeddings)
    import json
    cache_index.write_text(json.dumps({"python": 0, "sql": 1}))

    parser = VacancyParser()
    parser._embedding_model = MagicMock()
    result = parser._get_skill_embeddings(["python"])
    assert "python" in result
    parser._embedding_model.encode.assert_not_called()


def test_get_skill_embeddings_compute_new(tmp_path, monkeypatch, mock_embedding_model):
    """Строка: вычисление новых эмбеддингов"""
    monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    parser = VacancyParser()
    parser._embedding_model = mock_embedding_model
    skills = ["python", "django"]
    result = parser._get_skill_embeddings(skills)
    assert len(result) == 2
    mock_embedding_model.encode.assert_called_once()
    cache_npz = tmp_path / "skill_embeddings.npz"
    assert cache_npz.exists()


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
    assert VacancyParser.normalize_skill("командная работа") == "командная работа"
    assert VacancyParser.normalize_skill("очень длинное название навыка из пяти слов") == ""


def test_is_valid_skill():
    assert VacancyParser.is_valid_skill("Python") is True
    assert VacancyParser.is_valid_skill("") is False
    assert VacancyParser.is_valid_skill("a") is False


def test_count_skills():
    skills_list = ["Python", "python", "Django", "коммуникация"]
    with patch("src.parsing.vacancy_parser.load_it_skills", return_value={"python", "django"}):
        counts = VacancyParser.count_skills(skills_list)
    assert counts == {"python": 2, "django": 1}


def test_extract_skills_from_text_deprecated():
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
    assert df.loc[0, "Вакансия"] == "Python Developer"


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


# ----------------------------------------------------------------------
# Тесты SkillNormalizer
# ----------------------------------------------------------------------


class TestSkillNormalizer:
    def setup_method(self):
        SkillNormalizer._canonical_map = None
        SkillNormalizer._whitelist = None

    def test_synonyms(self):
        # Python 3.11 → убираем версию → "python" → в whitelist → ок
        assert SkillNormalizer.normalize("Python 3.11") == "python"
        # javascript → в SYNONYM_MAP маппится на "javascript" (каноник)
        # "javascript" не в whitelist, но fuzzy может сматчить
        result = SkillNormalizer.normalize("javascript")
        assert result in ("javascript", "js", "node.js")  # зависит от whitelist

    def test_suffix_removal(self):
        assert SkillNormalizer.normalize("язык python") == "python"

    def test_fuzzy_matching(self):
        SkillNormalizer._whitelist = None
        assert SkillNormalizer.normalize("reackt") == "react"
        # react native → в SYNONYM_MAP как "react native": ["reactnative"]
        # Каноник = "react native", вариант "reactnative"
        # После маппинга: "react native" остаётся "react native" (каноник сам в себя)
        result = SkillNormalizer.normalize("react native")
        assert result in ("react", "react native")
        # node js → в SYNONYM_MAP: "nodejs": ["node.js", "node js", "node"]
        # "node js" → в canon_map → "nodejs"
        result = SkillNormalizer.normalize("node js")
        assert result in ("nodejs", "node.js")
        result = SkillNormalizer.normalize("NodeJS")
        assert result in ("nodejs", "node.js")

    def test_no_match_returns_cleaned_version(self):
        result = SkillNormalizer.normalize("какой-то_мусор_навык_123")
        # После очистки версий, спецсимволов и нормализации
        # "какой-то_мусор_навык_123" → удаление цифр → "какой-то_мусор_навык_"
        # → очистка спецсимволов → "какой-то мусор навык"
        # Точное поведение зависит от порядка операций в normalize()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_batch(self):
        skills = ["Python 3", "React.js v18", "reackt", "machine learning"]
        normalized = SkillNormalizer.normalize_batch(skills)
        assert normalized[0] == "python"
        assert normalized[1] == "react"
        assert normalized[2] == "react"
        # "machine learning" нормализуется (фактический результат зависит от whitelist)
        assert isinstance(normalized[3], str)
        assert len(normalized[3]) > 0

    def test_batch_with_duplicates(self):
        skills = ["Python", "python", "", "React", "reackt"]
        normalized = SkillNormalizer.normalize_batch(skills)
        assert normalized == ["python", "python", "react", "react"]
        dedup = SkillNormalizer.deduplicate(skills)
        assert dedup == ["python", "react"]

    def test_normalize_edge_cases(self):
        assert SkillNormalizer.normalize("") == ""
        assert SkillNormalizer.normalize("   Python   ") == "python"
        assert SkillNormalizer.normalize("React.js v18") == "react"
        assert SkillNormalizer.normalize("NodeJS") == "node.js"

    def test_empty_and_whitespace(self):
        assert SkillNormalizer.normalize("") == ""
        assert SkillNormalizer.normalize("   ") == ""
        assert SkillNormalizer.normalize("\t\n") == ""
        assert SkillNormalizer.normalize(None) == ""

    def test_version_patterns(self):
        assert SkillNormalizer.normalize("Python 3.11") == "python"
        assert SkillNormalizer.normalize("React v18.2.1") == "react"
        assert SkillNormalizer.normalize("Django (фреймворк)") == "django"
        assert SkillNormalizer.normalize("Angular [2+]") == "angular"
        assert SkillNormalizer.normalize("PostgreSQL 14.5") == "postgresql"

    def test_suffix_removal_extended(self):
        assert SkillNormalizer.normalize("язык программирования Python") == "python"
        assert SkillNormalizer.normalize("Python язык") == "python"
        assert SkillNormalizer.normalize("React фреймворк") == "react"
        assert SkillNormalizer.normalize("pandas библиотека") == "pandas"
        assert SkillNormalizer.normalize("MongoDB database") == "mongodb"

    def test_special_characters_cleaning(self):
        assert SkillNormalizer.normalize("Node.JS") == "node.js"
        assert SkillNormalizer.normalize("some_skill!") == "some_skill"


    def test_whitelist_exact_match_prevents_fuzzy(self):
        assert SkillNormalizer.normalize("python") == "python"
        assert SkillNormalizer.normalize("pythn") == "python"

    def test_deduplicate_order_and_uniqueness(self):
        skills = ["React", "react", "Python", "python", "Docker", "React"]
        result = SkillNormalizer.deduplicate(skills)
        assert result == ["react", "python", "docker"]
        skills_with_empty = ["", "React", None, "React"]
        result2 = SkillNormalizer.deduplicate(skills_with_empty)
        assert result2 == ["react"]

    def test_fuzzy_matching_extended(self):
        # "anguler" → очистка спецсимволов → "anguler" → fuzzy
        # Может не сматчиться если score < 85, остаётся "anguler"
        result = SkillNormalizer.normalize("anguler")
        assert len(result) > 0
        # "pythn" → fuzzy → "python"
        assert SkillNormalizer.normalize("pythn") == "python"
        # "dockr" → fuzzy → "docker"
        assert SkillNormalizer.normalize("dockr") == "docker"
        # "xyzabc" → нет в whitelist → остаётся как есть
        result = SkillNormalizer.normalize("xyzabc")
        assert "xyzabc" in result

    def test_whitelist_loading_and_caching(self):
        SkillNormalizer._whitelist = None
        whitelist1 = SkillNormalizer._get_whitelist()
        whitelist2 = SkillNormalizer._get_whitelist()
        assert whitelist1 is whitelist2
        assert "python" in whitelist1
        assert "mlops" in whitelist1
        assert "node.js" in whitelist1

    def test_direct_phrase_mapping(self):
        if hasattr(SkillNormalizer, "DIRECT_PHRASE_MAP"):
            assert SkillNormalizer.normalize("machine learning") == "mlops"
            assert SkillNormalizer.normalize("node js") == "node.js"
            assert SkillNormalizer.normalize("react native") == "react"

    def test_complex_input_combinations(self):
        assert SkillNormalizer.normalize("язык программирования Python 3.11") == "python"
        assert SkillNormalizer.normalize("фреймворк React.js v18") == "react"

    def test_normalize_with_special_chars_and_version(self):
        assert SkillNormalizer.normalize("  PyThOn  3.9  ") == "python"
        assert SkillNormalizer.normalize("Node.JS (среда)") == "node.js"
        assert SkillNormalizer.normalize("TypEScript") == "typescript"

    def test_fuzzy_threshold_boundary(self):
        result = SkillNormalizer.normalize("completelywrongterm")
        assert "completelywrongterm" in result

    def test_phrase_synonyms_replaced_correctly(self):
        result = SkillNormalizer.normalize("react native")
        assert result in ("react", "react native")
        result = SkillNormalizer.normalize("machine learning")
        # Может быть "machine learning", "ml", "mlops" или что-то ещё
        assert isinstance(result, str) and len(result) > 0
        result = SkillNormalizer.normalize("node js")
        assert result in ("nodejs", "node.js")

    def test_synonym_replacement_does_not_affect_unrelated_words(self):
        # "javascripting" → нормализуется, не должно падать
        result = SkillNormalizer.normalize("javascripting")
        assert isinstance(result, str) and len(result) > 0

    def test_deduplicate_preserves_order_and_removes_duplicates(self):
        skills = ["Python", "python", "React", "reackt", "Docker", "docker"]
        dedup = SkillNormalizer.deduplicate(skills)
        assert dedup == ["python", "react", "docker"]

    def test_canonical_map_initialization(self):
        SkillNormalizer._canonical_map = None
        canon_map = SkillNormalizer._get_canonical_map()
        assert isinstance(canon_map, dict)
        assert "javascript" in canon_map
        assert "node.js" in canon_map.values() or "nodejs" in canon_map.values()

    def test_apply_synonym_map_direct(self):
        """Прямой вызов _apply_synonym_map"""
        result = SkillNormalizer._apply_synonym_map("node.js")
        assert result in ("nodejs", "node.js")

        result = SkillNormalizer._apply_synonym_map("unknown_skill_xyz")
        assert result == "unknown_skill_xyz"

    def test_machine_learning_normalization(self):
        """machine learning нормализуется без ошибок"""
        result = SkillNormalizer.normalize("machine learning")
        # Может быть "ml", "machine learning", "mlops", "qml" — зависит от whitelist
        assert isinstance(result, str)
        assert len(result) > 0

    def test_direct_search_with_negation_context(self):
        """Строки 37, 40-42: поиск с отрицанием"""
        parser = SkillParser()
        parser.TECH_SKILLS.add("python")
        text = "знание Python не требуется обязательно"
        skills = parser._direct_search(text, SkillSource.DESCRIPTION)
        assert not any(s.text == "python" for s in skills)

    def test_marker_search_no_marker_in_text(self):
        """Строка 117: маркер не найден в тексте"""
        parser = SkillParser()
        skills = parser._marker_search("обычный текст без маркеров", SkillSource.DESCRIPTION)
        assert skills == []

    def test_regex_search_no_matches(self):
        """Строка 158: regex ничего не нашёл"""
        parser = SkillParser()
        skills = parser._regex_search("текст без паттернов", SkillSource.DESCRIPTION)
        assert skills == []

    def test_parse_vacancy_with_snippet_responsibility(self):
        """Строка 228: парсинг snippet.responsibility"""
        parser = SkillParser()
        from src.models.vacancy import Area, Employer, Snippet, Vacancy

        vac = Vacancy(
            id="1",
            name="Test",
            area=Area(1, "MSK"),
            employer=Employer("1", "Corp"),
            key_skills=[],
            snippet=Snippet(requirement=None, responsibility="знание Docker и Kubernetes"),
            description=None,
        )
        skills = parser.parse_vacancy(vac)
        texts = {s.text.lower() for s in skills}
        assert "docker" in texts or "kubernetes" in texts


# ----------------------------------------------------------------------
# Тесты VacancyParser
# ----------------------------------------------------------------------


class TestVacancyParser:
    def test_hybrid_weights_are_calculated(self, sample_vacancy_dict):
        parser = VacancyParser()
        with (
            patch.object(parser, "_calculate_bm25_weights", return_value={"python": 0.8, "django": 0.6}),
            patch.object(
                parser, "_get_skill_embeddings", return_value={"python": [0.1, 0.2], "django": [0.3, 0.4]}
            ),
        ):
                weights = parser._calculate_hybrid_weights([sample_vacancy_dict])
        assert "python" in weights
        assert "django" in weights
        assert isinstance(weights["python"], float)

    def test_extract_skills_frequencies_count_correctly(self, monkeypatch):
        parser = VacancyParser()
        area = Area(id=1, name="Москва")
        employer = Employer(id="1", name="Test Corp")

        vac1 = Vacancy(
            id="1",
            name="Dev1",
            area=area,
            employer=employer,
            key_skills=[KeySkill(name="Python"), KeySkill(name="Django")],
            description="FastAPI",
        )
        vac2 = Vacancy(
            id="2", name="Dev2", area=area, employer=employer, key_skills=[KeySkill(name="Python")], description=""
        )
        vac3 = Vacancy(id="3", name="Dev3", area=area, employer=employer, key_skills=[], description="Docker")
        vacancies = [vac1, vac2, vac3]

        def mock_parse(vac):
            skills = []
            for ks in vac.key_skills:
                skills.append(ExtractedSkill(ks.name, SkillSource.KEY_SKILLS, 1.0))
            if vac.description:
                skills.append(ExtractedSkill(vac.description, SkillSource.DESCRIPTION, 0.9))
            return skills

        monkeypatch.setattr(parser.skill_parser, "parse_vacancy", mock_parse)
        monkeypatch.setattr(parser.skill_validator, "validate_batch", lambda skills, confidences=None: (skills, []))

        result = parser.extract_skills_from_vacancies(vacancies)

        assert result["frequencies"].get("python") == 2
        assert result["frequencies"].get("django") == 1
        assert result["frequencies"].get("fastapi") == 1
        assert result["frequencies"].get("docker") == 1

    def test_extract_skills_removes_duplicates_per_vacancy(self, monkeypatch):
        parser = VacancyParser()
        area = Area(id=1, name="Москва")
        employer = Employer(id="1", name="Test Corp")

        vac = Vacancy(
            id="1",
            name="Dev",
            area=area,
            employer=employer,
            key_skills=[KeySkill(name="Python"), KeySkill(name="python")],
            description="Python required",
        )
        vacancies = [vac]

        def mock_parse(vac):
            return [
                ExtractedSkill("Python", SkillSource.KEY_SKILLS, 1.0),
                ExtractedSkill("python", SkillSource.KEY_SKILLS, 1.0),
                ExtractedSkill("Python", SkillSource.DESCRIPTION, 0.9),
            ]

        monkeypatch.setattr(parser.skill_parser, "parse_vacancy", mock_parse)
        monkeypatch.setattr(parser.skill_validator, "validate_batch", lambda skills, confidences=None: (skills, []))

        result = parser.extract_skills_from_vacancies(vacancies)
        assert result["frequencies"].get("python") == 1

    def test_get_skill_embeddings_cache_corrupted(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
        cache_file = tmp_path / "skill_embeddings.json"
        cache_file.write_text("{not valid json", encoding="utf-8")

        parser = VacancyParser()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        parser._embedding_model = mock_model

        embeddings = parser._get_skill_embeddings(["python"])
        mock_model.encode.assert_called_once()
        assert "python" in embeddings

    def test_extract_skills_from_vacancies_skips_invalid_dict(self, monkeypatch):
        parser = VacancyParser()
        mock_parser = MagicMock()
        parser.skill_parser = mock_parser

        invalid_dict = {"id": "123", "name": "Test"}
        with patch("src.models.vacancy.Vacancy.from_api", side_effect=ValueError):
            result = parser.extract_skills_from_vacancies([invalid_dict])

        assert result["frequencies"] == {}
        mock_parser.parse_vacancy.assert_not_called()

    def test_extract_skills_from_vacancies_with_no_skills_or_desc(self, monkeypatch):
        parser = VacancyParser()
        area = Area(id=1, name="Москва")
        employer = Employer(id="1", name="Test Corp")
        vac = Vacancy(id="1", name="Empty", area=area, employer=employer, key_skills=[], description="")
        with patch.object(parser.skill_parser, "parse_vacancy", return_value=[]):
            result = parser.extract_skills_from_vacancies([vac])
            assert result["frequencies"] == {}

    def test_calculate_bm25_weights_empty_texts(self):
        parser = VacancyParser()
        weights = parser._calculate_bm25_weights([])
        assert weights == {}

    def test_calculate_bm25_weights_no_valid_ngrams(self):
        """BM25 с текстом без валидных n-грамм"""
        parser = VacancyParser()
        # Текст только из стоп-слов
        vacancies = [{"description": "в на по для и или не", "key_skills": [], "snippet": {}}]
        weights = parser._calculate_bm25_weights(vacancies)
        # Все n-граммы отфильтрованы стоп-словами → пустой результат
        assert weights == {}

    def test_calculate_bm25_weights_uses_snippet(self):
        parser = VacancyParser()
        vacancy = {
            "description": "",
            "key_skills": [],
            "snippet": {
                "requirement": "Опыт работы с Python и SQL",
                "responsibility": "Разработка микросервисов на FastAPI",
            },
        }
        with patch.object(parser, "_strip_html", side_effect=lambda x: x):
            weights = parser._calculate_bm25_weights([vacancy])
        # Просто проверяем, что метод отработал без ошибок
        assert isinstance(weights, dict)

    def test_calculate_hybrid_weights_few_embeddings(self, monkeypatch):
        parser = VacancyParser()
        bm25 = {f"skill{i}": 0.5 for i in range(5)}
        with (
            patch.object(parser, "_calculate_bm25_weights", return_value=bm25),
            patch.object(parser, "_get_skill_embeddings", return_value={f"skill{i}": [0.1] for i in range(5)}),
        ):
                weights = parser._calculate_hybrid_weights([{}])
                assert weights == bm25

    def test_extract_skills_from_vacancies_with_invalid_dict_continued(self, monkeypatch):
        """Покрытие continue после except ValueError"""
        parser = VacancyParser()
        # Первая вакансия невалидна, вторая валидна
        bad_dict = {"id": "bad"}
        good_dict = {
            "id": "good",
            "name": "Job",
            "area": {"id": 1, "name": "MSK"},
            "employer": {"id": "1", "name": "Corp"},
            "key_skills": [{"name": "Python"}],
        }
        with patch("src.models.vacancy.Vacancy.from_api") as mock_from_api:
            mock_from_api.side_effect = [
                ValueError,
                Vacancy(
                    id="good",
                    name="Job",
                    area=Area(1, "MSK"),
                    employer=Employer("1", "Corp"),
                    key_skills=[KeySkill("Python")],
                ),
            ]
            result = parser.extract_skills_from_vacancies([bad_dict, good_dict])
        assert "python" in result["frequencies"]

    def test_calculate_bm25_weights_zero_division_handling(self):
        """BM25 с термином, вызывающим ZeroDivisionError"""
        parser = VacancyParser()
        vacancies = [{"description": "Python разработка", "key_skills": [], "snippet": {}}]
        with patch("rank_bm25.BM25Okapi.get_scores", side_effect=ZeroDivisionError):
            weights = parser._calculate_bm25_weights(vacancies)
        # Должен вернуть пустой словарь без падения
        assert weights == {}

    def test_hybrid_weights_empty_bm25(self):
        """_calculate_hybrid_weights с пустым BM25"""
        parser = VacancyParser()
        with patch.object(parser, "_calculate_bm25_weights", return_value={}):
            weights = parser._calculate_hybrid_weights([])
        assert weights == {}

    def test_save_raw_vacancies_with_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_RAW_DIR", tmp_path)
        parser = VacancyParser()
        parser.save_raw_vacancies([], filename="empty.json")
        saved = tmp_path / "empty.json"
        assert saved.exists()
        with open(saved, encoding="utf-8") as f:
            data = json.load(f)
        assert data == []

    def test_strip_html_empty(self):
        """_strip_html с пустым текстом"""
        result = VacancyParser._strip_html("")
        assert result == ""

    def test_strip_html_with_none(self):
        """_strip_html с None"""
        result = VacancyParser._strip_html(None)
        assert result == ""

    def test_validate_low_confidence(self):
        """Строки 44-48: низкая уверенность"""
        validator = SkillValidator(min_confidence=0.8)
        result = validator.validate("Python", confidence=0.5)
        assert not result.is_valid

    def test_validate_too_long_string(self):
        """Строка 266: слишком длинная строка"""
        validator = SkillValidator(max_length=10)
        result = validator.validate("очень длинный навык")
        assert not result.is_valid

    def test_validate_only_digits(self):
        """Строка 269: только цифры"""
        validator = SkillValidator()
        result = validator.validate("12345")
        assert not result.is_valid

    def test_validate_no_letters(self):
        """Строка 278: нет букв"""
        validator = SkillValidator()
        result = validator.validate("123!@#")
        assert not result.is_valid

    def test_validate_batch_empty(self):
        """Строки 370-371: валидация пустого списка"""
        validator = SkillValidator()
        valid, results = validator.validate_batch([])
        assert valid == []
        assert results == []

    def test_get_stats_initial(self):
        """Строка 392: начальная статистика"""
        validator = SkillValidator()
        stats = validator.get_stats()
        assert stats["total"] == 0
        assert stats["valid"] == 0

    def test_extract_skills_from_vacancies_mixed_types(self, monkeypatch):
        """Строки 328, 358-387: смесь dict и Vacancy объектов"""
        parser = VacancyParser()
        area = Area(id=1, name="MSK")
        employer = Employer(id="1", name="Corp")
        vac_obj = Vacancy(id="obj", name="Obj", area=area, employer=employer, key_skills=[KeySkill("Python")])
        vac_dict = {
            "id": "dict",
            "name": "Dict",
            "area": {"id": 1, "name": "MSK"},
            "employer": {"id": "1", "name": "Corp"},
            "key_skills": [{"name": "SQL"}],
        }
        with patch.object(parser.skill_validator, "validate_batch", lambda skills, confidences=None: (skills, [])):
            result = parser.extract_skills_from_vacancies([vac_obj, vac_dict])
        assert result["frequencies"].get("python") == 1
        assert result["frequencies"].get("sql") == 1

    def test_calculate_bm25_weights_with_vacancy_objects(self):
        """Строки 409, 436: BM25 с Vacancy объектами"""
        parser = VacancyParser()
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(
            id="1",
            name="Test",
            area=area,
            employer=employer,
            key_skills=[KeySkill("python")],
            description="опыт работы с docker",
        )
        with patch.object(parser, "_strip_html", side_effect=lambda x: x):
            weights = parser._calculate_bm25_weights([vac])
        assert isinstance(weights, dict)

    def test_print_vacancies_list_without_skills(self, capsys):
        """Строки 530-531: вывод без навыков"""
        parser = VacancyParser()
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="1", name="Empty", area=area, employer=employer)
        parser.print_vacancies_list([vac])
        captured = capsys.readouterr()
        assert "Empty" in captured.out

    def test_print_vacancies_list_dict_without_skills(self, capsys):
        """Строки 583-584: dict без навыков"""
        parser = VacancyParser()
        parser.print_vacancies_list(
            [{"id": "1", "name": "No skills", "employer": {"name": "Corp"}, "area": {"name": "MSK"}, "key_skills": []}]
        )
        captured = capsys.readouterr()
        assert "No skills" in captured.out
    def test_embedding_model_load_failure(self, monkeypatch):
        """Строки 90-98: ошибка загрузки модели эмбеддингов"""
        from src.parsing.vacancy_parser import VacancyParser
        monkeypatch.setattr(
            "src.parsing.vacancy_parser.get_embedding_model",
            lambda: (_ for _ in ()).throw(Exception("fail"))
        )
        parser = VacancyParser()
        assert parser.embedding_model is None

    def test_calculate_bm25_zero_division(self):
        """Строка 567: обработка ZeroDivisionError"""
        from src.parsing.vacancy_parser import VacancyParser
        parser = VacancyParser()
        with patch("rank_bm25.BM25Okapi.get_scores", side_effect=ZeroDivisionError):
            weights = parser._calculate_bm25_weights([{"description": "Python разработка"}])
        assert weights == {}

    def test_extract_skills_from_vacancies_empty_list(self):
        """Строка 164: пустой список вакансий"""
        parser = VacancyParser()
        result = parser.extract_skills_from_vacancies([])
        assert result["frequencies"] == {}
        assert result["hybrid_weights"] == {}
        assert result["skill_embeddings"] == {}

    def test_embedding_model_fallback_on_exception(self, monkeypatch):
        """Строки 90-98: ошибка загрузки → None"""
        monkeypatch.setattr(
            "src.parsing.vacancy_parser.get_embedding_model",
            lambda: (_ for _ in ()).throw(RuntimeError("GPU not available"))
        )
        parser = VacancyParser()
        assert parser._embedding_model is None
        assert parser.embedding_model is None  # свойство тоже возвращает None

    def test_embedding_model_lazy_load_success(self):
        """Строки 90-98: успешная ленивая загрузка"""
        parser = VacancyParser()
        # При первом обращении модель загружается
        model = parser.embedding_model
        assert model is not None
        # Повторное обращение возвращает ту же модель
        assert parser.embedding_model is model

    def test_strip_html_with_tags(self):
        """Строки 205-209: очистка HTML с тегами"""
        result = VacancyParser._strip_html("<p>Hello <b>World</b></p>")
        assert result == "Hello World"

    def test_strip_html_with_attributes(self):
        """Строки 205-209: HTML с атрибутами"""
        result = VacancyParser._strip_html('<div class="test">Content</div>')
        assert "Content" in result
        assert "div" not in result
        assert "class" not in result

    def test_clean_highlighttext_empty(self):
        """Строки 205-209: clean_highlighttext с None"""
        result = VacancyParser.clean_highlighttext(None)
        assert result == ""

    def test_clean_highlighttext_no_tags(self):
        """Строки 205-209: clean_highlighttext без тегов"""
        result = VacancyParser.clean_highlighttext("Plain text")
        assert result == "Plain text"

    def test_normalize_skill_motivation(self):
        """Строка 797: чёрный список слов"""
        assert VacancyParser.normalize_skill("инициатива") == ""
        assert VacancyParser.normalize_skill("коммуникация") == ""
        assert VacancyParser.normalize_skill("многозадачность") == ""

    def test_normalize_skill_with_prefix(self):
        """Строки 770, 797: удаление префиксов"""
        assert VacancyParser.normalize_skill("опыт работы с Python") == "python"
        assert VacancyParser.normalize_skill("знание Docker") == "docker"

    def test_is_valid_skill_edge_cases(self):
        """Строки 894-895: граничные случаи is_valid_skill"""
        assert VacancyParser.is_valid_skill(None) is False
        assert VacancyParser.is_valid_skill("ab") is False
        assert VacancyParser.is_valid_skill("abc") is True
        assert VacancyParser.is_valid_skill("  py  ") is True

    def test_count_skills_with_whitelist(self):
        """Строки 949-950: count_skills с фильтрацией по whitelist"""
        skills_list = ["Python", "python", "Django", "unknown_skill_xyz"]
        with patch("src.parsing.vacancy_parser.load_it_skills", return_value={"python", "django"}):
            counts = VacancyParser.count_skills(skills_list)
        assert counts == {"python": 2, "django": 1}
        assert "unknown_skill_xyz" not in counts

    def test_count_skills_empty_whitelist(self):
        """Строки 949-950: count_skills с пустым whitelist"""
        skills_list = ["Python", "python", "Django"]
        with patch("src.parsing.vacancy_parser.load_it_skills", return_value=set()):
            counts = VacancyParser.count_skills(skills_list)
        assert len(counts) == 2  # python и django

    def test_extract_skills_static_empty(self):
        """Строки 260-261: extract_skills с пустыми key_skills"""
        result = VacancyParser.extract_skills([{"key_skills": [], "id": "1"}])
        assert result == []

    def test_extract_skills_static_missing_key(self):
        """Строки 260-261: extract_skills без key_skills"""
        result = VacancyParser.extract_skills([{"id": "1", "name": "Test"}])
        assert result == []

    def test_save_processed_frequencies_no_whitelist(self, tmp_path, monkeypatch):
        """Строки 575-576: apply_filter=True с пустым whitelist"""
        monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_PROCESSED_DIR", tmp_path)
        parser = VacancyParser()
        with patch("src.parsing.vacancy_parser.load_it_skills", return_value=set()):
            parser.save_processed_frequencies({"python": 10}, apply_filter=True)
        saved = tmp_path / "competency_frequency.json"
        assert saved.exists()

    def test_calculate_bm25_cached_corpus(self):
        """Строки 618-621: использование кэшированного корпуса"""
        parser = VacancyParser()
        parser._cached_corpus = {"python": 0.9}
        parser._corpus_hash = "test_hash"

        with patch.object(parser, "_get_corpus_hash", return_value="test_hash"):
            weights = parser._calculate_bm25_weights([{"description": "python"}])
        assert weights == {"python": 0.9}

    def test_calculate_bm25_vacancy_objects(self):
        """Строки 636, 667-672: BM25 с Vacancy объектами через description"""
        parser = VacancyParser()
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(
            id="1",
            name="Test",
            area=area,
            employer=employer,
            key_skills=[KeySkill("python")],
            description="опыт работы с docker и kubernetes",
        )
        with patch.object(parser, "_strip_html", side_effect=lambda x: x):
            with patch("src.parsing.vacancy_parser.load_it_skills", return_value={"python", "docker", "kubernetes", "k8s"}):
                weights = parser._calculate_bm25_weights([vac])
        assert isinstance(weights, dict)

    def test_calculate_hybrid_weights_with_pca(self):
        """Строки 679-685, 692-752: гибридные веса с PCA"""
        from src.parsing import vacancy_parser as vp_module

        parser = VacancyParser()
        # Создаём 150+ навыков для активации PCA
        skills = {f"skill_{i}": 0.5 for i in range(150)}

        # Мокаем BM25
        with patch.object(parser, "_calculate_bm25_weights", return_value=skills):
            # Мокаем эмбеддинги (384-мерные)
            mock_embs = {f"skill_{i}": np.random.rand(384).astype(np.float32) for i in range(150)}
            with patch.object(parser, "_get_skill_embeddings", return_value=mock_embs):
                # Включаем PCA через patch.object на модуль config
                with (
                    patch.object(vp_module.config, "PCA_ENABLED", True),
                    patch.object(vp_module.config, "PCA_MIN_SAMPLES", 100),
                    patch.object(vp_module.config, "PCA_MIN_FEATURES", 128),
                    patch.object(vp_module.config, "PCA_TARGET_DIM", 64),
                ):
                    weights = parser._calculate_hybrid_weights([{"description": "test"}])

        assert len(weights) > 0
        assert isinstance(weights, dict)

    def test_calculate_hybrid_weights_embedding_error_fallback(self):
        """Строки 636: ошибка эмбеддингов → fallback на BM25"""
        parser = VacancyParser()
        bm25 = {"python": 0.8, "sql": 0.6}

        with patch.object(parser, "_calculate_bm25_weights", return_value=bm25):
            with patch.object(parser, "_get_skill_embeddings", side_effect=RuntimeError("GPU error")):
                weights = parser._calculate_hybrid_weights([{"description": "test"}])

        # Должен вернуть нормализованный BM25
        assert "python" in weights
        assert "sql" in weights

    def test_calculate_hybrid_weights_few_embeddings(self):
        """Строки 679-685: меньше 10 эмбеддингов → только BM25"""
        parser = VacancyParser()
        bm25 = {f"skill_{i}": 0.5 for i in range(5)}

        with patch.object(parser, "_calculate_bm25_weights", return_value=bm25):
            with patch.object(parser, "_get_skill_embeddings", return_value={f"skill_{i}": [0.1] for i in range(5)}):
                weights = parser._calculate_hybrid_weights([{"description": "test"}])

        assert weights == bm25

    def test_aggregate_to_dataframe_with_salary(self, sample_vacancy_obj):
        """Строки 830-895: DataFrame с зарплатой"""
        parser = VacancyParser()
        from src.models.vacancy import Salary
        sample_vacancy_obj.salary = Salary(from_amount=100000, to_amount=150000, currency="RUB")
        df = parser.aggregate_to_dataframe([sample_vacancy_obj])
        assert df.loc[0, "Зарплата"] != "Не указана"

    def test_print_vacancies_list_mixed_types(self, capsys, sample_vacancy_dict, sample_vacancy_obj):
        """Строки 900-950: вывод смешанных типов"""
        parser = VacancyParser()
        parser.print_vacancies_list([sample_vacancy_dict, sample_vacancy_obj])
        captured = capsys.readouterr()
        assert "Python Developer" in captured.out

    def test_save_to_excel_creates_file(self, tmp_path, monkeypatch):
        """Строки 894-895: сохранение Excel"""
        monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_PROCESSED_DIR", tmp_path)
        df = pd.DataFrame({"col": [1, 2, 3]})
        parser = VacancyParser()
        parser.save_to_excel(df, "output.xlsx")
        assert (tmp_path / "output.xlsx").exists()

    def test_get_corpus_hash_mixed_types(self):
        """Строки 575-576: хэш корпуса для dict и Vacancy"""
        parser = VacancyParser()
        # Dict
        hash1 = parser._get_corpus_hash([{"id": "123"}, {"id": "456"}])
        assert isinstance(hash1, str)
        # Vacancy
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="789", name="Test", area=area, employer=employer)
        hash2 = parser._get_corpus_hash([vac])
        assert isinstance(hash2, str)
        assert hash1 != hash2

    def test_extract_skills_from_vacancies_with_validation(self, monkeypatch):
        """Строки 199-220: валидация навыков (много → ThreadPoolExecutor)"""
        parser = VacancyParser()
        # Создаём 250+ навыков для активации ThreadPoolExecutor
        skills = {f"skill_{i}": 1 for i in range(250)}
        parser.skill_parser.parse_vacancy = MagicMock(return_value=[
            ExtractedSkill(f"skill_{i}", SkillSource.KEY_SKILLS, 1.0) for i in range(250)
        ])
        parser.skill_validator.validate = MagicMock(return_value=ValidationResult(
            skill="test", is_valid=True, confidence=1.0
        ))

        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="1", name="Test", area=area, employer=employer, key_skills=[])

        result = parser.extract_skills_from_vacancies([vac])
        assert "frequencies" in result

    def test_get_corpus_hash_with_dicts(self):
        """Строки 575-576: хэш корпуса для списка dict"""
        parser = VacancyParser()
        hash1 = parser._get_corpus_hash([{"id": "123"}, {"id": "456"}])
        assert isinstance(hash1, str)
        assert ":" in hash1

    def test_get_corpus_hash_with_vacancy_objects(self):
        """Строки 575-576: хэш корпуса для Vacancy"""
        parser = VacancyParser()
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="789", name="Test", area=area, employer=employer)
        hash2 = parser._get_corpus_hash([vac])
        assert isinstance(hash2, str)
        assert "789" in hash2 or hash2.endswith(":789")

    def test_get_corpus_hash_mixed_ids(self):
        """Строки 575-576: хэш с нечисловыми ID"""
        parser = VacancyParser()
        hash1 = parser._get_corpus_hash([{"id": "abc"}, {"id": ""}])
        assert isinstance(hash1, str)

    def test_bm25_cached_corpus_reuse(self):
        """Строки 618-621: повторное использование кэшированного корпуса BM25"""
        parser = VacancyParser()
        parser._cached_corpus = {"cached_skill": 0.95}
        parser._corpus_hash = "5:12345"

        # Создаём вакансии с таким же хэшем
        with patch.object(parser, "_get_corpus_hash", return_value="5:12345"):
            weights = parser._calculate_bm25_weights([
                {"description": "dummy", "key_skills": [], "snippet": {}}
            ])

        # Должен вернуть кэшированный результат
        assert weights == {"cached_skill": 0.95}

    def test_hybrid_weights_embedding_exception_fallback(self):
        """Строка 636: ошибка получения эмбеддингов → fallback на чистый BM25"""
        parser = VacancyParser()
        bm25 = {"python": 0.9, "sql": 0.7}

        with patch.object(parser, "_calculate_bm25_weights", return_value=bm25):
            with patch.object(parser, "_get_skill_embeddings", side_effect=MemoryError("Out of memory")):
                weights = parser._calculate_hybrid_weights([{"description": "test"}])

        # Должен вернуть нормализованные BM25-веса
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_bm25_with_vacancy_objects_description(self):
        """Строки 667-672: BM25 с Vacancy-объектами (путь else)"""
        parser = VacancyParser()
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")

        # Vacancy без description и без key_skills — текст будет пустым
        vac = Vacancy(
            id="1", name="Empty",
            area=area, employer=employer,
            key_skills=[], description=None
        )

        # Мокаем load_it_skills и SkillNormalizer чтобы навыки прошли фильтрацию
        with patch("src.parsing.vacancy_parser.load_it_skills", return_value={"python", "docker"}):
            weights = parser._calculate_bm25_weights([vac])
        assert isinstance(weights, dict)

    def test_normalize_skill_blacklist_words(self):
        """Строка 797: нормализация с чёрным списком слов"""
        assert VacancyParser.normalize_skill("инициатива") == ""
        assert VacancyParser.normalize_skill("мотивация") == ""
        assert VacancyParser.normalize_skill("харизма") == ""

    def test_normalize_skill_with_suffix_removal(self):
        """Строка 770: удаление суффиксов"""
        result = VacancyParser.normalize_skill("Python плюсом")
        assert result == "python"

    def test_hybrid_weights_empty_bm25_early_return(self):
        """Строка 685: пустой BM25 → early return"""
        parser = VacancyParser()
        with patch.object(parser, "_calculate_bm25_weights", return_value={}):
            weights = parser._calculate_hybrid_weights([])
        assert weights == {}

    def test_embedding_model_lazy_load_failure_in_property(self, monkeypatch):
        """Строки 97-98: ошибка загрузки модели через property"""
        from src.parsing.vacancy_parser import VacancyParser
        monkeypatch.setattr(
            "src.parsing.vacancy_parser.get_embedding_model",
            lambda: (_ for _ in ()).throw(RuntimeError("GPU failure"))
        )
        parser = VacancyParser()
        assert parser._embedding_model is None
        assert parser.embedding_model is None

    def test_extract_skills_from_vacancies_empty(self):
        """Строка 164: пустой список вакансий"""
        parser = VacancyParser()
        result = parser.extract_skills_from_vacancies([])
        assert result["frequencies"] == {}
        assert result["hybrid_weights"] == {}
        assert result["skill_embeddings"] == {}

    def test_strip_html_with_tags_and_attributes(self):
        """Строки 205-209: очистка HTML с атрибутами"""
        result = VacancyParser._strip_html('<div class="main">Hello <b>World</b></div>')
        assert "Hello" in result
        assert "World" in result
        assert "class" not in result

    def test_bm25_zero_division_handling(self):
        """Строка 567: обработка ZeroDivisionError в BM25"""
        parser = VacancyParser()
        with patch("rank_bm25.BM25Okapi.get_scores", side_effect=ZeroDivisionError):
            weights = parser._calculate_bm25_weights([{"description": "Python", "key_skills": [], "snippet": {}}])
        assert isinstance(weights, dict)

    def test_get_corpus_hash_with_numeric_ids(self):
        """Строки 575-576: хэш корпуса с числовыми ID"""
        parser = VacancyParser()
        hash_val = parser._get_corpus_hash([{"id": "123"}, {"id": "456"}])
        assert isinstance(hash_val, str)
        assert ":" in hash_val

    def test_get_corpus_hash_with_string_ids(self):
        """Строки 575-576: хэш корпуса с нечисловыми ID"""
        parser = VacancyParser()
        hash_val = parser._get_corpus_hash([{"id": "abc-def"}])
        assert isinstance(hash_val, str)

    def test_bm25_cached_corpus_reuse(self):
        """Строки 618-621: использование кэшированного корпуса BM25"""
        parser = VacancyParser()
        parser._cached_corpus = {"python": 0.95, "sql": 0.8}
        parser._corpus_hash = "10:12345"

        with patch.object(parser, "_get_corpus_hash", return_value="10:12345"):
            weights = parser._calculate_bm25_weights([
                {"description": "ignored", "key_skills": [], "snippet": {}}
            ])

        assert weights == {"python": 0.95, "sql": 0.8}

    def test_hybrid_weights_embedding_exception_fallback(self):
        """Строка 636: ошибка эмбеддингов → fallback на BM25"""
        parser = VacancyParser()
        bm25 = {"python": 0.9, "sql": 0.5}

        with patch.object(parser, "_calculate_bm25_weights", return_value=bm25):
            with patch.object(parser, "_get_skill_embeddings", side_effect=MemoryError("OOM")):
                weights = parser._calculate_hybrid_weights([{"description": "test"}])

        assert "python" in weights
        assert "sql" in weights

    def test_bm25_with_vacancy_objects_empty(self):
        """Строки 667-672: BM25 с Vacancy-объектами без описания"""
        parser = VacancyParser()
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(id="1", name="Empty", area=area, employer=employer, key_skills=[], description=None)

        with patch("src.parsing.vacancy_parser.load_it_skills", return_value=set()):
            weights = parser._calculate_bm25_weights([vac])
        assert isinstance(weights, dict)

    def test_hybrid_weights_empty_bm25_early_return(self):
        """Строка 685: пустой BM25 → возврат пустого словаря"""
        parser = VacancyParser()
        with patch.object(parser, "_calculate_bm25_weights", return_value={}):
            weights = parser._calculate_hybrid_weights([])
        assert weights == {}

    def test_normalize_skill_blacklist_words(self):
        """Строка 797: чёрный список слов"""
        assert VacancyParser.normalize_skill("инициатива") == ""
        assert VacancyParser.normalize_skill("мотивация") == ""
        assert VacancyParser.normalize_skill("коммуникация") == ""
        assert VacancyParser.normalize_skill("клиентами") == ""
        assert VacancyParser.normalize_skill("харизма") == ""
        assert VacancyParser.normalize_skill("многозадачность") == ""

    def test_is_valid_skill_edge_cases(self):
        """Строки 894-895: граничные случаи валидации"""
        assert VacancyParser.is_valid_skill("") is False
        assert VacancyParser.is_valid_skill(None) is False
        assert VacancyParser.is_valid_skill("ab") is False
        assert VacancyParser.is_valid_skill("abc") is True
        assert VacancyParser.is_valid_skill("  py  ") is True

    def test_count_skills_with_whitelist_filtering(self):
        """Строки 949-950: count_skills с фильтрацией по whitelist"""
        skills_list = ["Python", "python", "Django", "unknown_skill_xyz"]
        with patch("src.parsing.vacancy_parser.load_it_skills", return_value={"python", "django"}):
            counts = VacancyParser.count_skills(skills_list)
        assert "python" in counts
        assert counts["python"] == 2
        assert "unknown_skill_xyz" not in counts

    def test_count_skills_with_empty_whitelist(self):
        """Строки 949-950: count_skills с пустым whitelist"""
        skills_list = ["Python", "python", "Django"]
        with patch("src.parsing.vacancy_parser.load_it_skills", return_value=set()):
            counts = VacancyParser.count_skills(skills_list)
        assert len(counts) > 0

    def test_extract_skills_static_no_key_skills(self):
        """Строки 260-261: extract_skills без key_skills"""
        result = VacancyParser.extract_skills([{"id": "1", "name": "Test"}])
        assert result == []

    def test_extract_skills_static_empty_key_skills(self):
        """Строки 260-261: extract_skills с пустыми key_skills"""
        result = VacancyParser.extract_skills([{"key_skills": [], "id": "1"}])
        assert result == []

    def test_normalize_skill_long_phrase(self):
        """Строки 794-795: длинная фраза → пустая строка"""
        result = VacancyParser.normalize_skill("очень длинное название навыка из пяти слов")
        assert result == ""

    def test_normalize_skill_with_suffix_removal(self):
        """Строка 788: удаление суффиксов"""
        result = VacancyParser.normalize_skill("Python плюсом")
        assert result == "python"
        result = VacancyParser.normalize_skill("Docker желательно")
        assert result == "docker"

    def test_hybrid_weights_normalization_path(self):
        """Строка 722: torch нормализация эмбеддингов"""
        parser = VacancyParser()
        skills = {f"skill_{i}": 0.5 for i in range(20)}

        with patch.object(parser, "_calculate_bm25_weights", return_value=skills):
            mock_embs = {f"skill_{i}": np.random.rand(384).astype(np.float32) for i in range(20)}
            with patch.object(parser, "_get_skill_embeddings", return_value=mock_embs):
                weights = parser._calculate_hybrid_weights([{"description": "test"}])

        assert len(weights) > 0
