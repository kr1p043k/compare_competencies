# -*- coding: utf-8 -*-
"""Парсер компетенций/заголовков аннотаций: реальные варианты написания
(замер 2026-10-14 на аннотациях ЮФУ с Яндекс.Диска)."""
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "sfu_annotations", Path(__file__).resolve().parent.parent.parent / "scripts" / "sfu_annotations.py")
sfu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sfu)


class TestParseCompetencies:
    def test_classic(self):
        assert sfu.parse_competencies("ОПК-1. Способен...") == ["ОПК-1"]

    def test_space_before_dash(self):
        # «Физика»: 'ОПК -1: способность...'
        assert sfu.parse_competencies("компетенций: ОПК -1: способность") == ["ОПК-1"]

    def test_en_dash(self):
        # «Компьютерные сети»: 'ОПК–2 Способен...'
        assert sfu.parse_competencies("компетенций: ОПК–2 Способен") == ["ОПК-2"]

    def test_minus_sign(self):
        assert sfu.parse_competencies("УК−8 в скобках") == ["УК-8"]

    def test_english_prefix(self):
        assert sfu.parse_competencies("OPK-3, UK-1 listed") == ["ОПК-3", "УК-1"]

    def test_in_parens(self):
        # «БЖД»: '(УК-8). Индикаторы...'
        assert "УК-8" in sfu.parse_competencies("ситуаций (УК-8). Индикаторы")

    def test_dedup_and_dot_numbers(self):
        got = sfu.parse_competencies("УК-1 ... УК-1 ... ОПК-2.1 и ОПК-2.1")
        assert got == ["УК-1", "ОПК-2.1"]

    def test_fname_fallback(self):
        # «Аннотация ВПК-8 (бакалавриат).pdf»: в тексте кода нет.
        assert sfu.parse_competencies("текст без кодов", "Аннотация ВПК-8 (бакалавриат).pdf") == ["ВПК-8"]

    def test_fname_no_false_positive_on_curriculum_code(self):
        # 'Б1.О.2.15' и '09.03.01УКД' — не компетенции.
        assert sfu.parse_competencies(
            "Дисциплина Б1.О.2.15 «X»", "АНН 09.03.01УКД Архитектура.pdf") == []

    def test_no_codes(self):
        assert sfu.parse_competencies("никаких кодов тут", "Course description.pdf") == []


class TestSafeCacheName:
    def test_trailing_dots_stripped(self):
        # Windows: open('x. ') -> FileNotFoundError, ронял весь сбор 01.03.01.
        name = sfu.safe_cache_name("Аннотация.")
        assert not name.endswith((".", " "))

    def test_long_name_capped(self):
        name = sfu.safe_cache_name("А" * 200 + ".pdf")
        assert len(name) <= 140
        assert name.endswith(".pdf")

    def test_deterministic(self):
        assert sfu.safe_cache_name("a b.pdf") == sfu.safe_cache_name("a b.pdf")


class TestTitleFromText:
    def test_ru_standard(self):
        t = "АННОТАЦИЯ\nрабочей программы учебной дисциплины\n«ИНФОРМАЦИОННОЕ МОДЕЛИРОВАНИЕ АРХИТЕКТУРНЫХ ОБЪЕКТОВ»\n1. Трудоемкость"
        assert sfu.title_from_text(t, "x.pdf") == "ИНФОРМАЦИОННОЕ МОДЕЛИРОВАНИЕ АРХИТЕКТУРНЫХ ОБЪЕКТОВ"

    def test_ru_gia_without_discipline_word(self):
        t = "АННОТАЦИЯ\nрабочей программы\n«Государственная итоговая аттестация»\n"
        assert sfu.title_from_text(t, "x.pdf") == "Государственная итоговая аттестация"

    def test_en_name_of_course_next_line(self):
        t = "Course description (annotation)\n1.     Name of the course\nDatabases and DBMS\n2. Department..."
        assert sfu.title_from_text(t, "x.pdf") == "Databases and DBMS"

    def test_en_of_the_course_junk_skipped(self):
        t = "Course description (annotation)\nof the course\nAlgorithmization and programming\n1. Total labor intensity"
        assert sfu.title_from_text(t, "x.pdf") == "Algorithmization and programming"

    def test_en_work_program(self):
        t = "ANNOTATION\nwork program of the academic discipline\nInformation technology security\n1. The name..."
        assert sfu.title_from_text(t, "x.pdf") == "Information technology security"

    def test_first_line_fallback(self):
        t = "Operating systems\nDepartment Institute of Computer Technologies\nLecturer Samoylov..."
        assert sfu.title_from_text(t, "x.pdf") == "Operating systems"

    def test_junk_numbered_line_not_title(self):
        # '1. Total labor intensity 8 ZET' не должно стать названием.
        t = "Course description (annotation)\n1. Total labor intensity 8 ZET\nMathematics"
        assert sfu.title_from_text(t, "x.pdf") == "Mathematics"

    def test_fallback_to_filename(self):
        # Все строки-заглушки отфильтрованы — остаётся имя файла.
        t = "Course description (annotation)\n1. Total labor intensity\n2. Department"
        assert sfu.title_from_text(t, "Web-программирование.pdf") == "Web-программирование"
