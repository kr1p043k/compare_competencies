"""Нормализация текста из PDF и устойчивость извлечения.

Проверяем артефакты pypdf, из-за которых теряются или склеиваются навыки:
лигатуры, мягкие переносы, неразрывные пробелы, дефисный перенос слов,
а также устойчивость к битым страницам и layout-фолбэк для таблиц РПД.
"""
from __future__ import annotations

import src.loaders.rpd_loader as rl
from src.loaders.rpd_loader import (
    RPDLoader,
    count_ksa_markers,
    cyrillic_ratio,
    normalize_pdf_text,
)


class TestNormalizePdfText:
    def test_expands_ligatures(self):
        assert normalize_pdf_text("con\ufb01guration") == "configuration"
        assert normalize_pdf_text("\ufb02ow \ufb00ect") == "flow ffect"

    def test_removes_soft_hyphens(self):
        assert normalize_pdf_text("про\u00adграммирование") == "программирование"

    def test_collapses_unicode_spaces(self):
        # NBSP, узкий неразрывный и обычный пробел схлопываются в один.
        assert normalize_pdf_text("a\u00a0b\u202fc") == "a b c"

    def test_strips_zero_width(self):
        assert normalize_pdf_text("ОПК\u200b-2") == "ОПК-2"

    def test_joins_hyphenated_word_break(self):
        # «про-\\nграммирование» -> одно слово.
        assert normalize_pdf_text("про-\nграммирование") == "программирование"

    def test_does_not_join_competency_code_break(self):
        # Код разорван после дефиса, но следующая строка начинается с цифры.
        assert normalize_pdf_text("ОПК-\n2") == "ОПК-\n2"

    def test_does_not_join_when_hyphen_has_leading_space(self):
        # «см. - \\nдалее» - дефис с пробелом перед ним, это не перенос слова.
        assert normalize_pdf_text("см. -\nдалее") == "см. -\nдалее"

    def test_collapses_inner_spaces_but_keeps_newlines(self):
        assert normalize_pdf_text("Знания:    основы\n\nУмения:  код") == "Знания: основы\n\nУмения: код"

    def test_empty_is_empty(self):
        assert normalize_pdf_text("") == ""


class TestCyrillicRatio:
    def test_counts_yo_as_cyrillic(self):
        # «ё»/«Ё» вне диапазона «а»..«я», раньше выпадали из числителя.
        assert cyrillic_ratio("ёЁёЁ") == 1.0

    def test_ratio_over_letters_only(self):
        # Цифры и пунктуация не разбавляют знаменатель: 4 русские буквы из 4.
        assert cyrillic_ratio("ОПК-2: 10.5%") == 1.0

    def test_latin_is_zero(self):
        assert cyrillic_ratio("Hello world 123") == 0.0

    def test_empty_is_zero(self):
        assert cyrillic_ratio("") == 0.0


class _FakePage:
    def __init__(self, text=None, error=None):
        self._text = text
        self._error = error

    def extract_text(self, *args, **kwargs):
        if self._error:
            raise self._error
        return self._text


class _FakeReader:
    def __init__(self, pages):
        self.pages = pages


def test_extract_text_pypdf_skips_broken_pages(monkeypatch):
    """Сбойная страница не обнуляет всю дисциплину - уцелевшие идут в текст."""
    pages = [
        _FakePage("Первая страница"),
        _FakePage(error=ValueError("broken stream")),
        _FakePage("Третья страница"),
    ]
    monkeypatch.setattr(rl, "PdfReader", lambda _p: _FakeReader(pages))
    text = rl.extract_text_pypdf("x.pdf")
    assert text is not None
    assert "Первая страница" in text
    assert "Третья страница" in text


def test_extract_text_pypdf_returns_none_when_open_fails(monkeypatch):
    def boom(_p):
        raise OSError("cannot open")

    monkeypatch.setattr(rl, "PdfReader", boom)
    assert rl.extract_text_pypdf("x.pdf") is None


def test_extract_text_pypdf_normalizes_output(monkeypatch):
    monkeypatch.setattr(rl, "PdfReader", lambda _p: _FakeReader([_FakePage("con\ufb01g")]))
    assert rl.extract_text_pypdf("x.pdf") == "config"


def test_layout_fallback_wins_when_more_markers(tmp_path, monkeypatch):
    """plain-режим склеил колонки (0 маркеров), layout вернул таблицу - берём layout."""
    plain = "ОПК-2 Знания Умения Навыки без двоеточий"
    layout = "ОПК-2\nЗнания:\n- основы\nУмения:\n- применять\nНавыки:\n- владеть\n"

    def fake_extract(_path, extraction_mode="plain"):
        return layout if extraction_mode == "layout" else plain

    monkeypatch.setattr(rl, "extract_text_pypdf", fake_extract)
    monkeypatch.setattr(rl, "cyrillic_ratio", lambda _t: 0.9)

    loader = RPDLoader(pdf_dir=str(tmp_path))
    text = loader.extract_text(str(tmp_path / "x.pdf"))
    assert text == layout
    assert count_ksa_markers(text) >= 3


def test_layout_fallback_ignored_when_not_better(tmp_path, monkeypatch):
    plain = "Знания:\n- a\nУмения:\n- b\nНавыки:\n- c\n"

    def fake_extract(_path, extraction_mode="plain"):
        # layout хуже или равен - оставляем plain.
        return "мусор" if extraction_mode == "layout" else plain

    monkeypatch.setattr(rl, "extract_text_pypdf", fake_extract)
    monkeypatch.setattr(rl, "cyrillic_ratio", lambda _t: 0.9)

    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader.extract_text(str(tmp_path / "x.pdf")) == plain


def test_layout_fallback_swallows_patch_signature_error(tmp_path, monkeypatch):
    """Патч extract_text_pypdf без kwarg не должен ронять extract_text."""
    monkeypatch.setattr(rl, "extract_text_pypdf", lambda _p: "Компетенции дисциплины")
    monkeypatch.setattr(rl, "cyrillic_ratio", lambda _t: 0.9)
    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader.extract_text(str(tmp_path / "x.pdf")) == "Компетенции дисциплины"
