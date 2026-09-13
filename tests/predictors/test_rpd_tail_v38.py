"""Unit (v38): RPD metadata tails stripped, content kept."""
from src.pipeline.teacher_analysis_runner import _strip_rpd_tail


def test_direction_codes_stripped():
    a = _strip_rpd_tail("администрирование СУБД PostgreSQL 09.03.02")
    assert a == "администрирование СУБД PostgreSQL"
    b = _strip_rpd_tail("командной работы, распределения ролей 09.05.01 10.05.02")
    assert b == "командной работы, распределения ролей"


def test_template_tail_cut():
    c = _strip_rpd_tail("разработка процедур Transact-SQL Код направления подготовки, специальности 10.03.01")
    assert c == "разработка процедур Transact-SQL"


def test_clean_untouched():
    assert _strip_rpd_tail("Базы данных и СУБД") == "Базы данных и СУБД"
    assert _strip_rpd_tail("python") == "python"
