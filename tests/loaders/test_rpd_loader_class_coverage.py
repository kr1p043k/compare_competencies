"""Покрытие методов класса RPDLoader: приватные хелперы, загрузка, сохранение.

Реальный pypdf/OCR не запускаем: подменяем извлечение текста, чтобы тесты были
детерминированными и быстрыми. Проверяем логику выбора файла, канонизации имён,
слияния дублей дисциплин и структуру результата.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.loaders.rpd_loader as rl
from src.loaders.rpd_loader import RPDLoader


# ---------- приватные хелперы ----------


def test_is_rpd_filters_service_files():
    loader = RPDLoader(pdf_dir=".")
    assert loader._is_rpd("Основы программирования.pdf") is True
    assert loader._is_rpd("Rating_2024.pdf") is False
    assert loader._is_rpd("ECTS.pdf") is False
    assert loader._is_rpd("BPM.pdf") is False
    assert loader._is_rpd("something.sig") is False


def test_discipline_name_strips_rpd_prefix():
    loader = RPDLoader(pdf_dir=".")
    assert loader._discipline_name("РПД_Базы данных.pdf") == "Базы данных"
    assert loader._discipline_name("Без префикса.pdf") == "Без префикса"


def test_find_pdf_prefers_exact_and_cyrillic(tmp_path: Path):
    for name in (
        "Базы данных.pdf",
        "Базы данных и СУБД.pdf",
        "Databases (Bazy dannykh).pdf",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")

    loader = RPDLoader(pdf_dir=str(tmp_path))

    exact = loader._find_pdf("Базы данных")
    assert exact is not None and Path(exact).name == "Базы данных.pdf"

    fuzzy = loader._find_pdf("Базы данных и СУБД")
    assert fuzzy is not None and Path(fuzzy).name == "Базы данных и СУБД.pdf"


def test_find_pdf_returns_none_when_no_match(tmp_path: Path):
    (tmp_path / "Математика.pdf").write_text("x", encoding="utf-8")
    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader._find_pdf("Совсем другое") is None


def test_resolve_canonical_uses_alias(tmp_path: Path):
    loader = RPDLoader(pdf_dir=str(tmp_path))
    if not rl.DISC_NORM:
        pytest.skip("нет алиасов дисциплин")
    alias, canonical = next(iter(rl.DISC_NORM.items()))
    assert loader._resolve_canonical(alias) == canonical


def test_resolve_canonical_autodetect_russian_variant(tmp_path: Path):
    (tmp_path / "РПД_Базы данных.pdf").write_text("x", encoding="utf-8")
    loader = RPDLoader(pdf_dir=str(tmp_path))
    # Смешанное имя -> должен найтись русскоязычный вариант по общим словам.
    resolved = loader._resolve_canonical("Базы данных (Databases)")
    assert resolved == "Базы данных"


def test_resolve_canonical_none_for_unknown(tmp_path: Path):
    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader._resolve_canonical("Совершенно незнакомый предмет") is None


# ---------- извлечение текста ----------


def test_extract_text_returns_none_when_pypdf_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "extract_text_pypdf", lambda _p: None)
    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader.extract_text(str(tmp_path / "x.pdf")) is None


def test_extract_text_returns_directly_when_cyrillic_enough(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "extract_text_pypdf", lambda _p: "Компетенции дисциплины")
    monkeypatch.setattr(rl, "cyrillic_ratio", lambda _t: 0.9)
    called = {"ocr": False}
    monkeypatch.setattr(rl, "extract_text_ocr", lambda _p: called.__setitem__("ocr", True) or "ocr")

    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader.extract_text(str(tmp_path / "x.pdf")) == "Компетенции дисциплины"
    assert called["ocr"] is False


def test_extract_text_falls_back_to_ocr_on_low_cyrillic(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "extract_text_pypdf", lambda _p: "ABC abc ABC")
    monkeypatch.setattr(rl, "cyrillic_ratio", lambda _t: 0.05)
    monkeypatch.setattr(rl, "extract_text_ocr", lambda _p: "Распознанный текст")
    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader.extract_text(str(tmp_path / "x.pdf")) == "Распознанный текст"


def test_extract_text_keeps_original_when_ocr_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "extract_text_pypdf", lambda _p: "ABC abc")
    monkeypatch.setattr(rl, "cyrillic_ratio", lambda _t: 0.05)
    monkeypatch.setattr(rl, "extract_text_ocr", lambda _p: None)
    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader.extract_text(str(tmp_path / "x.pdf")) == "ABC abc"


def test_extract_text_skips_ocr_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "extract_text_pypdf", lambda _p: "ABC abc")
    monkeypatch.setattr(rl, "cyrillic_ratio", lambda _t: 0.05)
    monkeypatch.setattr(rl, "extract_text_ocr", lambda _p: "не должно вызываться")
    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader.extract_text(str(tmp_path / "x.pdf"), use_ocr=False) == "ABC abc"


# ---------- parse_text ----------


def test_parse_text_uses_section_when_found(monkeypatch):
    loader = RPDLoader(pdf_dir=".")
    monkeypatch.setattr(rl, "find_section_text", lambda _t, _m: "секция")
    monkeypatch.setattr(rl, "extract_ksa_skills", lambda t: {"УК-1": {"flat": [t]}})
    out = loader.parse_text("текст")
    assert out == {"УК-1": {"flat": ["секция"]}}


def test_parse_text_falls_back_to_full_text(monkeypatch):
    loader = RPDLoader(pdf_dir=".")
    monkeypatch.setattr(rl, "find_section_text", lambda _t, _m: None)
    calls = []

    def fake_extract(t):
        calls.append(t)
        return {}

    monkeypatch.setattr(rl, "extract_ksa_skills", fake_extract)
    out = loader.parse_text("полный текст")
    assert out == {}
    assert calls == ["полный текст"]


# ---------- load_discipline / load_all ----------


def test_load_discipline_returns_none_when_pdf_missing(tmp_path):
    loader = RPDLoader(pdf_dir=str(tmp_path))
    assert loader.load_discipline("Нет такого") is None


def test_load_discipline_returns_none_when_text_empty(tmp_path, monkeypatch):
    (tmp_path / "РПД_Тест.pdf").write_text("x", encoding="utf-8")
    loader = RPDLoader(pdf_dir=str(tmp_path))
    monkeypatch.setattr(loader, "extract_text", lambda *_a, **_k: None)
    assert loader.load_discipline("Тест") is None


def test_load_discipline_returns_none_when_nothing_parsed(tmp_path, monkeypatch):
    (tmp_path / "РПД_Тест.pdf").write_text("x", encoding="utf-8")
    loader = RPDLoader(pdf_dir=str(tmp_path))
    monkeypatch.setattr(loader, "extract_text", lambda *_a, **_k: "текст")
    monkeypatch.setattr(loader, "parse_text", lambda _t: {})
    assert loader.load_discipline("Тест") is None


def test_load_discipline_returns_structured_result(tmp_path, monkeypatch):
    (tmp_path / "РПД_Тест.pdf").write_text("x", encoding="utf-8")
    loader = RPDLoader(pdf_dir=str(tmp_path))
    monkeypatch.setattr(loader, "extract_text", lambda *_a, **_k: "текст")
    monkeypatch.setattr(
        loader,
        "parse_text",
        lambda _t: {"УК-1": {"knowledge": [], "abilities": [], "skills": [], "flat": ["python"]}},
    )
    out = loader.load_discipline("Тест")
    assert out is not None
    assert out["competencies"] == ["УК-1"]
    assert out["skills"]["УК-1"] == ["python"]


def test_load_all_merges_duplicate_disciplines(tmp_path, monkeypatch):
    (tmp_path / "РПД_Базы данных.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "РПД_Базы данных (Databases).pdf").write_text("x", encoding="utf-8")

    loader = RPDLoader(pdf_dir=str(tmp_path), )
    monkeypatch.setattr(loader, "extract_text", lambda *_a, **_k: "текст")
    monkeypatch.setattr(loader, "_resolve_canonical", lambda _n: "Базы данных")

    def fake_parse(text):
        # Разные наборы навыков для двух файлов -> проверяем слияние.
        if "DB2" in text:
            return {"УК-2": {"knowledge": [], "abilities": [], "skills": [], "flat": ["sql"]}}
        return {"УК-1": {"knowledge": [], "abilities": [], "skills": [], "flat": ["python"]}}

    monkeypatch.setattr(loader, "parse_text", fake_parse)
    out = loader.load_all(dir_code="09.03.02", direction_name="Базы", profile="Профиль")

    disciplines = out["09.03.02"]["disciplines"]
    assert list(disciplines) == ["Базы данных"]
    comps = disciplines["Базы данных"]["competencies"]
    assert "УК-1" in comps
    assert out["09.03.02"]["direction_name"] == "Базы"
    assert out["09.03.02"]["profile"] == "Профиль"


def test_load_all_merges_children_into_parent(tmp_path, monkeypatch):
    (tmp_path / "РПД_Тест.pdf").write_text("x", encoding="utf-8")
    loader = RPDLoader(pdf_dir=str(tmp_path))
    monkeypatch.setattr(loader, "extract_text", lambda *_a, **_k: "текст")
    monkeypatch.setattr(
        loader,
        "parse_text",
        lambda _t: {
            "УК-1": {"knowledge": [], "abilities": [], "skills": [], "flat": ["a"]},
            "УК-1.1": {"knowledge": [], "abilities": [], "skills": [], "flat": ["b"]},
        },
    )
    out = loader.load_all()
    skills = out["09.03.02"]["disciplines"]["Тест"]["skills"]
    assert "b" in skills["УК-1"], "навык дочерней компетенции должен подняться к родителю"


def test_load_all_skips_files_without_text(tmp_path, monkeypatch):
    (tmp_path / "РПД_Пусто.pdf").write_text("x", encoding="utf-8")
    loader = RPDLoader(pdf_dir=str(tmp_path))
    monkeypatch.setattr(loader, "extract_text", lambda *_a, **_k: None)
    out = loader.load_all(dir_code="01.01.01")
    assert out["01.01.01"]["disciplines"] == {}


# ---------- save / stats ----------


def test_save_writes_json(tmp_path: Path):
    loader = RPDLoader(pdf_dir=".")
    target = tmp_path / "sub" / "krm.json"
    loader.save({"a": 1}, str(target))
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


def test_stats_counts_disciplines_skills_and_zero(tmp_path: Path):
    loader = RPDLoader(pdf_dir=".")
    data = {
        "09.03.02": {
            "disciplines": {
                "Д1": {"skills": {"УК-1": ["python", "sql"]}},
                "Д2": {"skills": {"УК-2": []}},
            }
        }
    }
    st = loader.stats(data)
    assert st == {"disciplines": 2, "skills": 2, "zero": 1}


def test_stats_handles_empty(tmp_path: Path):
    loader = RPDLoader(pdf_dir=".")
    assert loader.stats({}) == {"disciplines": 0, "skills": 0, "zero": 0}
