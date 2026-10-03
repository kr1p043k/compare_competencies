"""Регрессия гэпов ОПК-4: матчер обязан находить C++/тестирование и не врать.

Эталон разобран 2026-09-22: длинные РПД-формулировки vs observed market.
"""
import pytest

from src.analyzers.skill_matcher import SkillMatcher, fold_script, sig_lemmas

MARKET = {
    "c++": 425,
    "c/c++": 160,
    "c++ gpu": 3,  # fringe
    "вычисления на gpu": 9,
    "программирование плк": 7,
    "тестирование": 73,
    "тестирование по": 14,
    "функциональное тестирование": 111,
    "модульное тестирование": 20,
    "анализ данных на python": 11,
}

PHRASES = [
    "использовать принципы структурного подхода для создания программ",
    "отлаживать программы на языке С/С++",
    "представлять числовые данные в кодах, выполнять над ними операции",
    "создавать программный код, реализовывать отдельные фрагменты сложных программ профессиональных задач",
    "создание программной реализации алгоритма, синтаксис и семантику языков программирования С/С++",
    "способ отладки и тестирования программ",
]


@pytest.fixture(scope="module")
def matcher():
    return SkillMatcher(dict(MARKET))


def test_cyrillic_cpp_matched(matcher):
    got = matcher.match(PHRASES[1]).ok()[0]
    assert got in ("c++", "c/c++")


def test_cyrillic_cpp_algo_matched(matcher):
    got = matcher.match(PHRASES[4]).ok()[0]
    assert got in ("c++", "c/c++")


def test_testing_family_matched(matcher):
    got = matcher.match(PHRASES[5]).ok()[0]
    assert got in ("тестирование", "тестирование по", "функциональное тестирование",
                   "модульное тестирование")


def test_genuine_absences_stay_gaps(matcher):
    assert matcher.match(PHRASES[0]).ok()[0] is None
    assert matcher.match(PHRASES[2]).ok()[0] is None
    # обобщённое «программирование» без объекта — ловушка ПЛК должна молчать
    assert matcher.match(PHRASES[3]).ok()[0] is None


def test_no_absurd_hits(matcher):
    absurd = {"вычисления на gpu", "c++ gpu", "анализ данных на python",
              "программирование плк"}
    for p in PHRASES:
        assert matcher.match(p).ok()[0] not in absurd


def test_fold_script_consistent():
    assert fold_script("с/с++") == "c/c++"
    q = sig_lemmas("способ отладки и тестирования программ")
    s = sig_lemmas("тестирование по")
    assert s and s <= q
    assert sig_lemmas("c/c++") <= sig_lemmas("отлаживать программы на языке С/С++")


def test_anchor_respects_fringe():
    m = SkillMatcher({"документация": 1})
    assert m.match("отчетная документация").ok()[0] is None
