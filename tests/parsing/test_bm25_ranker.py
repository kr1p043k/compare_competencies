# tests/parsing/test_bm25_ranker.py

import pytest
from unittest.mock import patch, MagicMock
from src import Ok
from src.parsing.skills.bm25_ranker import BM25Ranker

def identity_normalize(x):
    return Ok(x.lower() if isinstance(x, str) else x)

@pytest.fixture(autouse=True)
def mock_global_whitelist():
    """Глобальный мок для load_it_skills – используется по умолчанию."""
    with patch("src.parsing.skills.bm25_ranker.load_it_skills",
               return_value={"python", "sql", "machine learning", "deep learning", "java"}):
        yield

# ================= существующие тесты (исправлены) =================

def test_bm25_ranker_empty_vacancies():
    ranker = BM25Ranker()
    weights = ranker.calculate_weights([])
    assert weights == Ok({})

def test_bm25_ranker_simple_vacancy():
    ranker = BM25Ranker()
    vacancies = [{"description": "Python и SQL", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok() and isinstance(weights.unwrap(), dict)
    assert weights.is_ok() and ("python" in weights.unwrap() or "sql" in weights.unwrap())

def test_bm25_only_stopwords():
    ranker = BM25Ranker()
    vacancies = [{"description": "в на по для", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize):
        weights = ranker.calculate_weights(vacancies)
    assert weights == Ok({})

def test_bm25_with_known_skills():
    ranker = BM25Ranker()
    vacancies = [{"description": "python sql", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok() and isinstance(weights.unwrap(), dict)
    assert weights.is_ok() and "python" in weights.unwrap()

def test_bm25_caching():
    ranker = BM25Ranker()
    vacancies = [{"description": "python", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights1 = ranker.calculate_weights(vacancies)
            weights2 = ranker.calculate_weights(vacancies)
            assert weights1 == weights2

def test_bm25_caching_and_no_valid_ngrams():
    ranker = BM25Ranker()
    with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize):
        weights1 = ranker.calculate_weights([{"description": "python", "key_skills": [], "snippet": {}}])
        weights2 = ranker.calculate_weights([{"description": "python", "key_skills": [], "snippet": {}}])
    assert weights1 == weights2

def test_bm25_no_valid_ngrams_stopwords():
    ranker = BM25Ranker()
    vacancies = [{"description": "в на по для", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize):
        weights = ranker.calculate_weights(vacancies)
    assert weights == Ok({})

def test_bm25_with_phrases():
    ranker = BM25Ranker()
    vacancies = [{"description": "machine learning and deep learning", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok() and ("machine learning" in weights.unwrap() or "deep learning" in weights.unwrap())

def test_bm25_filter_by_whitelist_phrases():
    ranker = BM25Ranker()
    vacancies = [{"description": "python sql", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok() and "python" in weights.unwrap()
    assert weights.is_ok() and "sql" in weights.unwrap()
    assert weights.is_ok() and "machine learning" not in weights.unwrap()

def test_bm25_remove_stopwords_from_ngrams():
    ranker = BM25Ranker()
    vacancies = [{"description": "python и sql", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok() and "python" in weights.unwrap()
    assert weights.is_ok() and "sql" in weights.unwrap()
    assert weights.is_ok() and "и" not in weights.unwrap()

def test_bm25_corpus_hash_different():
    ranker = BM25Ranker()
    v1 = [{"id": "1", "description": "python"}]
    v2 = [{"id": "2", "description": "java"}]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            w1 = ranker.calculate_weights(v1)
            w2 = ranker.calculate_weights(v2)
            assert w1 != w2

# ================= новые тесты для покрытия пропущенных строк =================

def test_bm25_empty_text(monkeypatch):
    """Строка 329: описание отсутствует или пустое, текст не формируется."""
    ranker = BM25Ranker()
    # Вакансия без description, snippet и key_skills
    vacancies = [{"description": "", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize):
        weights = ranker.calculate_weights(vacancies)
    assert weights == Ok({})

def test_bm25_no_words():
    """Строка 333: текст состоит только из знаков препинания, words = []."""
    ranker = BM25Ranker()
    vacancies = [{"description": "!!! ???", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize):
        weights = ranker.calculate_weights(vacancies)
    assert weights == Ok({})

def test_bm25_no_valid_ngrams(monkeypatch):
    """Строка 354: tokenized_corpus остаётся пустым (нет навыков из whitelist)."""
    ranker = BM25Ranker()
    # Убираем стандартный whitelist и ставим такой, чтобы ни одно слово не подошло
    with patch("src.parsing.skills.bm25_ranker.load_it_skills", return_value={"c++"}):
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize):
            vacancies = [{"description": "some unknown skill", "key_skills": [], "snippet": {}}]
            weights = ranker.calculate_weights(vacancies)
    assert weights == Ok({})

def test_bm25_get_scores_empty():
    """Строка 384: bm25.get_scores возвращает пустой список."""
    ranker = BM25Ranker()
    vacancies = [{"description": "python", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        # Пустой результат для любого навыка
        instance.get_scores.return_value = []
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights = ranker.calculate_weights(vacancies)
    assert weights == Ok({})

def test_bm25_score_below_threshold():
    """Строки 388-389: avg <= порога → навык не добавляется."""
    ranker = BM25Ranker()
    vacancies = [{"description": "python", "key_skills": [], "snippet": {}}]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [0.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.1):
            weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok() and "python" not in weights.unwrap()
    assert weights == Ok({})

def test_bm25_with_vacancy_objects():
    """Покрытие ветки обработки объектов Vacancy."""
    ranker = BM25Ranker()
    mock_vacancy = MagicMock()
    mock_vacancy.description = "Python developer"
    skill1 = MagicMock()
    skill1.name = "python"
    skill2 = MagicMock()
    skill2.name = "sql"
    mock_vacancy.key_skills = [skill1, skill2]

    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights = ranker.calculate_weights([mock_vacancy])
    assert weights.is_ok() and "python" in weights.unwrap()

def test_bm25_with_snippet_and_skills():
    """Покрытие извлечения requirement, responsibility и key_skills."""
    ranker = BM25Ranker()
    vacancies = [{
        "description": "some text",
        "snippet": {"requirement": "python experience", "responsibility": "deploy sql"},
        "key_skills": [{"name": "machine learning"}]
    }]
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]
        with patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok() and any(skill in weights.unwrap() for skill in ["python", "sql", "machine learning"])

def test_bm25_doc_limit():
    """Покрытие ветки ограничения количества документов (unique_docs > max_docs)."""
    ranker = BM25Ranker()
    # 3000 вакансий с уникальными навыками, все в whitelist
    skills = [f"skill_{i}" for i in range(3000)]
    whitelist = set(skills)
    vacancies = [{"description": f"skill_{i}", "key_skills": [], "snippet": {}} for i in range(3000)]

    # Патчим BM25Okapi, чтобы перехватить аргумент конструктора
    with patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
        instance = MockBM25.return_value
        instance.get_scores.return_value = [1.0]

        with patch("src.parsing.skills.bm25_ranker.load_it_skills", return_value=whitelist), \
             patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0):
            weights = ranker.calculate_weights(vacancies)

    # Проверяем, что BM25Okapi был вызван с ограниченным числом документов
    MockBM25.assert_called_once()
    args, _ = MockBM25.call_args
    corpus_passed = args[0]  # это unique_docs после ограничения
    assert len(corpus_passed) == 300  # 3000 total // 10 = 300
    # Веса могут содержать все 3000 навыков, т.к. они оцениваются уже после обрезки
    assert weights.is_ok() and len(weights.unwrap()) == 3000

def test_bm25_morph_exception():
    """Покрытие except в лемматизации при ошибке pymorphy3."""
    ranker = BM25Ranker()
    with patch.object(ranker._morph, 'parse', side_effect=Exception("morph error")):
        vacancies = [{"description": "питон sql", "key_skills": [], "snippet": {}}]
        with patch("src.parsing.skills.bm25_ranker.load_it_skills", return_value={"питон", "sql"}), \
             patch("src.parsing.skills.bm25_ranker.SkillNormalizer.normalize", side_effect=identity_normalize), \
             patch("src.parsing.skills.bm25_ranker.config.BM25_MIN_SCORE", 0.0), \
             patch("src.parsing.skills.bm25_ranker.BM25Okapi") as MockBM25:
            instance = MockBM25.return_value
            instance.get_scores.return_value = [1.0]
            weights = ranker.calculate_weights(vacancies)
    assert weights.is_ok() and "питон" in weights.unwrap()  # лемма упала → осталось исходное слово
