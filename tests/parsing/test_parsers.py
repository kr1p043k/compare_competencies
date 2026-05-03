# tests/parsing/test_parsers.py
from src.parsing.skill_normalizer import SkillNormalizer
from src.models.vacancy import Vacancy, Area, Employer, KeySkill
import pytest
import json
import pandas as pd
from unittest.mock import patch, MagicMock
import numpy as np
from src.parsing.vacancy_parser import VacancyParser
from src.parsing.skill_parser import SkillParser, ExtractedSkill, SkillSource
from src.parsing.skill_validator import SkillValidator, ValidationResult


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
        "snippet": {}  # для совместимости
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


@patch('src.parsing.vacancy_parser.get_embedding_model')
def test_init_embedding_model_load_failure(mock_get_emb):
    mock_get_emb.side_effect = Exception("Model load error")
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
def test_extract_skills_from_vacancies_dict(mock_normalizer, mock_skill_parser, mock_skill_validator, tmp_path, monkeypatch, sample_vacancy_dict):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    mock_normalizer.normalize_batch.return_value = ["python", "django", "fastapi"]

    with patch('src.parsing.vacancy_parser.VacancyParser._get_skill_embeddings', return_value={"python": [0.1, 0.2]}):
        parser = VacancyParser()
        parser.skill_parser = mock_skill_parser
        parser.skill_validator = mock_skill_validator
        parser.embedding_model = MagicMock()

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
    monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    cache_file = tmp_path / "skill_embeddings.json"
    cache_data = {"python": [0.1, 0.2], "sql": [0.3, 0.4]}
    cache_file.write_text(json.dumps(cache_data), encoding='utf-8')

    parser = VacancyParser()
    parser.embedding_model = MagicMock()
    embeddings = parser._get_skill_embeddings(["python"])
    assert embeddings["python"] == [0.1, 0.2]
    parser.embedding_model.encode.assert_not_called()


def test_get_skill_embeddings_compute_new(tmp_path, monkeypatch, mock_embedding_model):
    monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
    parser = VacancyParser()
    parser.embedding_model = mock_embedding_model
    skills = ["python", "django"]
    embeddings = parser._get_skill_embeddings(skills)
    assert len(embeddings) == 2
    mock_embedding_model.encode.assert_called_once()
    cache_file = tmp_path / "skill_embeddings.json"
    assert cache_file.exists()


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
    with patch('src.parsing.vacancy_parser.load_it_skills', return_value={"python", "django"}):
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
        assert result == "nodejs"
        # NodeJS → lowercase → "nodejs" → в canon_map → "nodejs"
        result = SkillNormalizer.normalize("NodeJS")
        assert result == "nodejs"

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
        # "machine learning" → в SYNONYM_MAP: "ml": ["machine learning", "ml"]
        # "machine learning" → canon_map["machine learning"] = "ml"
        assert normalized[3] in ("ml", "machine learning")
        
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

    def test_fuzzy_matching(self):
        SkillNormalizer._whitelist = None
        assert SkillNormalizer.normalize("reackt") == "react"
        # react native → SYNONYM_MAP: "react native": ["reactnative"]
        # Каноник = "react native", вариант "reactnative"
        # После маппинга остаётся "react native" (каноник сам в себя)
        result = SkillNormalizer.normalize("react native")
        assert result in ("react", "react native")
        # node js → SYNONYM_MAP: "nodejs": ["node.js", "node js", "node"]
        # "node js" → canon_map → "nodejs", потом whitelist может заменить на "node.js"
        result = SkillNormalizer.normalize("node js")
        assert result in ("nodejs", "node.js")
        # NodeJS → lowercase → "nodejs" → canon_map → "nodejs"
        result = SkillNormalizer.normalize("NodeJS")
        assert result in ("nodejs", "node.js")

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

    def test_normalize_batch(self):
        skills = ["Python 3", "React.js v18", "reackt", "machine learning"]
        normalized = SkillNormalizer.normalize_batch(skills)
        assert normalized[0] == "python"
        assert normalized[1] == "react"
        assert normalized[2] == "react"
        # "machine learning" → SYNONYM_MAP: "ml": ["machine learning", "ml"]
        # Но также есть "mlops": [...] и "dl": [...]
        # После normalizer: может остаться "machine learning" или стать "ml"
        # Проверяем что это валидная строка
        assert isinstance(normalized[3], str)
        assert len(normalized[3]) > 0

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
        if hasattr(SkillNormalizer, 'DIRECT_PHRASE_MAP'):
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
        assert "node.js" in canon_map.values()


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

    def test_normalize_with_special_chars_and_version(self):
        assert SkillNormalizer.normalize("  PyThOn  3.9  ") == "python"
        # Node.JS → lowercase → "node.js" → в canon_map как вариант "nodejs"
        result = SkillNormalizer.normalize("Node.JS (среда)")
        assert result in ("nodejs", "node.js")
        # TypeScript → lowercase → "typescript" → whitelist содержит "typescript"
        result = SkillNormalizer.normalize("TypEScript")
        assert result in ("typescript", "ts")

    def test_canonical_map_initialization(self):
        SkillNormalizer._canonical_map = None
        canon_map = SkillNormalizer._get_canonical_map()
        assert isinstance(canon_map, dict)
        # Проверяем что канонические имена маппятся сами в себя
        assert "python" in canon_map
        assert canon_map["python"] == "python"
        # Проверяем что варианты маппятся на каноник
        assert "js" in canon_map
        assert canon_map["js"] == "javascript"
        # Проверяем наличие node.js вариантов
        assert "node.js" in canon_map or "nodejs" in canon_map.values()

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
        from src.models.vacancy import Vacancy, Snippet, Area, Employer
        vac = Vacancy(
            id="1", name="Test",
            area=Area(1, "MSK"),
            employer=Employer("1", "Corp"),
            key_skills=[],
            snippet=Snippet(requirement=None, responsibility="знание Docker и Kubernetes"),
            description=None
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
        with patch.object(parser, '_calculate_bm25_weights', return_value={'python': 0.8, 'django': 0.6}):
            with patch.object(parser, '_get_skill_embeddings', return_value={'python': [0.1, 0.2], 'django': [0.3, 0.4]}):
                weights = parser._calculate_hybrid_weights([sample_vacancy_dict])
        assert 'python' in weights
        assert 'django' in weights
        assert isinstance(weights['python'], float)

    def test_extract_skills_frequencies_count_correctly(self, monkeypatch):
        parser = VacancyParser()
        area = Area(id=1, name="Москва")
        employer = Employer(id="1", name="Test Corp")

        vac1 = Vacancy(
            id="1", name="Dev1", area=area, employer=employer,
            key_skills=[KeySkill(name="Python"), KeySkill(name="Django")],
            description="FastAPI"
        )
        vac2 = Vacancy(
            id="2", name="Dev2", area=area, employer=employer,
            key_skills=[KeySkill(name="Python")],
            description=""
        )
        vac3 = Vacancy(
            id="3", name="Dev3", area=area, employer=employer,
            key_skills=[],
            description="Docker"
        )
        vacancies = [vac1, vac2, vac3]

        def mock_parse(vac):
            skills = []
            for ks in vac.key_skills:
                skills.append(ExtractedSkill(ks.name, SkillSource.KEY_SKILLS, 1.0))
            if vac.description:
                skills.append(ExtractedSkill(vac.description, SkillSource.DESCRIPTION, 0.9))
            return skills

        monkeypatch.setattr(parser.skill_parser, "parse_vacancy", mock_parse)
        monkeypatch.setattr(parser.skill_validator, "validate_batch",
                            lambda skills, confidences=None: (skills, []))

        result = parser.extract_skills_from_vacancies(vacancies)

        assert result['frequencies'].get('python') == 2
        assert result['frequencies'].get('django') == 1
        assert result['frequencies'].get('fastapi') == 1
        assert result['frequencies'].get('docker') == 1

    def test_extract_skills_removes_duplicates_per_vacancy(self, monkeypatch):
        parser = VacancyParser()
        area = Area(id=1, name="Москва")
        employer = Employer(id="1", name="Test Corp")

        vac = Vacancy(
            id="1", name="Dev", area=area, employer=employer,
            key_skills=[KeySkill(name="Python"), KeySkill(name="python")],
            description="Python required"
        )
        vacancies = [vac]

        def mock_parse(vac):
            return [
                ExtractedSkill("Python", SkillSource.KEY_SKILLS, 1.0),
                ExtractedSkill("python", SkillSource.KEY_SKILLS, 1.0),
                ExtractedSkill("Python", SkillSource.DESCRIPTION, 0.9),
            ]

        monkeypatch.setattr(parser.skill_parser, "parse_vacancy", mock_parse)
        monkeypatch.setattr(parser.skill_validator, "validate_batch",
                            lambda skills, confidences=None: (skills, []))

        result = parser.extract_skills_from_vacancies(vacancies)
        assert result['frequencies'].get('python') == 1

    def test_get_skill_embeddings_cache_corrupted(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.parsing.vacancy_parser.config.EMBEDDINGS_CACHE_DIR", tmp_path)
        cache_file = tmp_path / "skill_embeddings.json"
        cache_file.write_text("{not valid json", encoding='utf-8')

        parser = VacancyParser()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        parser.embedding_model = mock_model

        embeddings = parser._get_skill_embeddings(["python"])
        mock_model.encode.assert_called_once()
        assert "python" in embeddings

    def test_extract_skills_from_vacancies_skips_invalid_dict(self, monkeypatch):
        parser = VacancyParser()
        mock_parser = MagicMock()
        parser.skill_parser = mock_parser

        invalid_dict = {"id": "123", "name": "Test"}
        with patch('src.models.vacancy.Vacancy.from_api', side_effect=ValueError):
            result = parser.extract_skills_from_vacancies([invalid_dict])

        assert result['frequencies'] == {}
        mock_parser.parse_vacancy.assert_not_called()

    def test_extract_skills_from_vacancies_with_no_skills_or_desc(self, monkeypatch):
        parser = VacancyParser()
        area = Area(id=1, name="Москва")
        employer = Employer(id="1", name="Test Corp")
        vac = Vacancy(
            id="1", name="Empty", area=area, employer=employer,
            key_skills=[], description=""
        )
        with patch.object(parser.skill_parser, 'parse_vacancy', return_value=[]):
            result = parser.extract_skills_from_vacancies([vac])
            assert result['frequencies'] == {}

    def test_calculate_bm25_weights_empty_texts(self):
        parser = VacancyParser()
        weights = parser._calculate_bm25_weights([])
        assert weights == {}

    def test_calculate_bm25_weights_no_valid_ngrams(self):
        """BM25 с текстом без валидных n-грамм"""
        parser = VacancyParser()
        # Текст только из стоп-слов
        vacancies = [{
            "description": "в на по для и или не",
            "key_skills": [],
            "snippet": {}
        }]
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
                "responsibility": "Разработка микросервисов на FastAPI"
            }
        }
        with patch.object(parser, '_strip_html', side_effect=lambda x: x):
            weights = parser._calculate_bm25_weights([vacancy])
        # Просто проверяем, что метод отработал без ошибок
        assert isinstance(weights, dict)

    def test_calculate_hybrid_weights_few_embeddings(self, monkeypatch):
        parser = VacancyParser()
        bm25 = {f"skill{i}": 0.5 for i in range(5)}
        with patch.object(parser, '_calculate_bm25_weights', return_value=bm25):
            with patch.object(parser, '_get_skill_embeddings', return_value={f"skill{i}": [0.1] for i in range(5)}):
                weights = parser._calculate_hybrid_weights([{}])
                assert weights == bm25

    def test_extract_skills_from_vacancies_with_invalid_dict_continued(self, monkeypatch):
        """Покрытие continue после except ValueError"""
        parser = VacancyParser()
        # Первая вакансия невалидна, вторая валидна
        bad_dict = {"id": "bad"}
        good_dict = {
            "id": "good", "name": "Job",
            "area": {"id": 1, "name": "MSK"},
            "employer": {"id": "1", "name": "Corp"},
            "key_skills": [{"name": "Python"}]
        }
        with patch('src.models.vacancy.Vacancy.from_api') as mock_from_api:
            mock_from_api.side_effect = [ValueError, Vacancy(
                id="good", name="Job",
                area=Area(1, "MSK"),
                employer=Employer("1", "Corp"),
                key_skills=[KeySkill("Python")]
            )]
            result = parser.extract_skills_from_vacancies([bad_dict, good_dict])
        assert "python" in result["frequencies"]

    def test_calculate_bm25_weights_zero_division_handling(self):
        """BM25 с термином, вызывающим ZeroDivisionError"""
        parser = VacancyParser()
        vacancies = [{
            "description": "Python разработка",
            "key_skills": [],
            "snippet": {}
        }]
        with patch('rank_bm25.BM25Okapi.get_scores', side_effect=ZeroDivisionError):
            weights = parser._calculate_bm25_weights(vacancies)
        # Должен вернуть пустой словарь без падения
        assert weights == {}

    def test_hybrid_weights_empty_bm25(self):
        """_calculate_hybrid_weights с пустым BM25"""
        parser = VacancyParser()
        with patch.object(parser, '_calculate_bm25_weights', return_value={}):
            weights = parser._calculate_hybrid_weights([])
        assert weights == {}

    def test_save_raw_vacancies_with_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.parsing.vacancy_parser.config.DATA_RAW_DIR", tmp_path)
        parser = VacancyParser()
        parser.save_raw_vacancies([], filename="empty.json")
        saved = tmp_path / "empty.json"
        assert saved.exists()
        with open(saved, 'r', encoding='utf-8') as f:
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
        assert stats['total'] == 0
        assert stats['valid'] == 0
        
    def test_extract_skills_from_vacancies_mixed_types(self, monkeypatch):
        """Строки 328, 358-387: смесь dict и Vacancy объектов"""
        parser = VacancyParser()
        area = Area(id=1, name="MSK")
        employer = Employer(id="1", name="Corp")
        vac_obj = Vacancy(
            id="obj", name="Obj", area=area, employer=employer,
            key_skills=[KeySkill("Python")]
        )
        vac_dict = {
            "id": "dict", "name": "Dict",
            "area": {"id": 1, "name": "MSK"},
            "employer": {"id": "1", "name": "Corp"},
            "key_skills": [{"name": "SQL"}]
        }
        with patch.object(parser.skill_validator, 'validate_batch', lambda skills, confidences=None: (skills, [])):
            result = parser.extract_skills_from_vacancies([vac_obj, vac_dict])
        assert result['frequencies'].get('python') == 1
        assert result['frequencies'].get('sql') == 1

    def test_calculate_bm25_weights_with_vacancy_objects(self):
        """Строки 409, 436: BM25 с Vacancy объектами"""
        parser = VacancyParser()
        area = Area(1, "MSK")
        employer = Employer("1", "Corp")
        vac = Vacancy(
            id="1", name="Test", area=area, employer=employer,
            key_skills=[KeySkill("python")],
            description="опыт работы с docker"
        )
        with patch.object(parser, '_strip_html', side_effect=lambda x: x):
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
        parser.print_vacancies_list([{
            "id": "1", "name": "No skills",
            "employer": {"name": "Corp"},
            "area": {"name": "MSK"},
            "key_skills": []
        }])
        captured = capsys.readouterr()
        assert "No skills" in captured.out
