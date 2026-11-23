"""Unit (v34): analysis scope predicate."""
from src.pipeline.teacher_analysis_runner import (
    SCOPE_EXCLUDED,
    discipline_in_scope,
    scope_key,
)


def test_exact_exclusions():
    assert len(SCOPE_EXCLUDED) == 10
    for name in ["Дисциплины по ФКиС", "Физическая культура и спорт",
                 "Эмоциональный интеллект и критическое мышление инженера",
                 "Экономико-правовое обеспечение инженерной деятельности_очная",
                 "Стрессоустойчивость и личная эффективность",
                 "История России", "Философия",
                 "Воображение, изображение, реальность",
                 "Безопасность жизнедеятельности",
                 "Психология профессионального саморазвития"]:
        assert discipline_in_scope(name) is False


def test_form_suffix_is_ignored():
    # Регрессия: исключён был только вариант «…_очная», а тот же предмет
    # без суффикса формы обучения попадал в анализ (34% покрытия в сводке).
    assert discipline_in_scope("Экономико-правовое обеспечение инженерной деятельности") is False
    assert discipline_in_scope("Экономико-правовое обеспечение инженерной деятельности (заочная)") is False
    assert scope_key("Философия_очная") == scope_key("Философия")


def test_languages_are_out_of_scope():
    # Решение владельца: английский и прочие не-IT дисциплины не учитываются.
    # C1 раньше оставляли, но у него 0% покрытия - только занижал среднее.
    assert discipline_in_scope("Иностранный язык (англ. яз., уровень С1)") is False
    assert discipline_in_scope("Иностранный язык (англ. яз., уровень C1)") is False
    assert discipline_in_scope("Иностранный язык для деловой коммуникации") is False
    assert discipline_in_scope("Иностранный язык (русский язык)") is False
    for lvl in ["А1", "А2", "В1", "В2"]:
        assert discipline_in_scope(f"Иностранный язык (англ. яз., уровень {lvl})") is False


def test_core_kept():
    assert discipline_in_scope("Базы данных и СУБД") is True
    assert discipline_in_scope("Машинное обучение в задачах компьютерного зрения") is True
    assert discipline_in_scope("Высшая математика") is True
    assert discipline_in_scope("") is False
