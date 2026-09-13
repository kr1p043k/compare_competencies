"""Unit (v34): analysis scope predicate."""
from src.pipeline.teacher_analysis_runner import SCOPE_EXCLUDED, discipline_in_scope


def test_exact_exclusions():
    assert len(SCOPE_EXCLUDED) == 7
    for name in ["Дисциплины по ФКиС", "Физическая культура и спорт",
                 "Эмоциональный интеллект и критическое мышление инженера",
                 "Экономико-правовое обеспечение инженерной деятельности_очная",
                 "Стрессоустойчивость и личная эффективность",
                 "История России", "Философия"]:
        assert discipline_in_scope(name) is False


def test_english_single_level_plus_business():
    assert discipline_in_scope("Иностранный язык (англ. яз., уровень С1)") is True
    assert discipline_in_scope("Иностранный язык для деловой коммуникации") is True
    for lvl in ["А1", "А2", "В1", "В2"]:
        assert discipline_in_scope(f"Иностранный язык (англ. яз., уровень {lvl})") is False


def test_russian_and_core_kept():
    assert discipline_in_scope("Иностранный язык (русский язык)") is True
    assert discipline_in_scope("Базы данных и СУБД") is True
    assert discipline_in_scope("Машинное обучение в задачах компьютерного зрения") is True
    assert discipline_in_scope("") is False
