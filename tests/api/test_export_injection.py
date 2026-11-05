"""Excel formula injection: внешние строки не должны становиться формулами.

Бэкенд пишет в .xlsx данные hh.ru (название, работодатель, город, навыки).
Ячейка, начинающаяся с = + - @ таба или \r, при открытии в Excel/LibreOffice
трактуется как формула. _excel_safe ставит кавычку впереди, и openpyxl хранит
значение как строку ("s"), а не формулу ("f").
"""
import pandas as pd
import pytest
from openpyxl import load_workbook

from src.api_pkg.routers.teacher import _excel_safe


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("=1+1", "'=1+1"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("-5+cmd", "'-5+cmd"),
        ("+CMD", "'+CMD"),
        ("\t=cmd", "'\t=cmd"),
        ("\r=cmd", "'\r=cmd"),
        ("normal text", "normal text"),
        ("", ""),
        ("  =1+1", "  =1+1"),  # ведущий пробел - не формула, не трогаем
        ("=", "'="),
    ],
)
def test_excel_safe_strings(raw, expected):
    assert _excel_safe(raw) == expected


def test_excel_safe_passthrough_non_strings():
    assert _excel_safe(123) == 123
    assert _excel_safe(1.5) == 1.5
    assert _excel_safe(None) is None


def test_excel_write_keeps_text_columns_as_strings_not_formulas(tmp_path):
    """Регресс: =1+1 не должен записаться как формула, число остаётся числом."""
    rows = [
        {
            "Название": _excel_safe("=1+1"),
            "Работодатель": _excel_safe("@SUM(A1)"),
            "Город": _excel_safe("-5+cmd"),
            "Навыки": _excel_safe("\t=cmd"),
            "Опыт": _excel_safe("normal"),
            "Зарплата от": 50000,
            "Зарплата до": None,
        }
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "vacancies_export.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")

    ws = load_workbook(path).active
    # Строка 1 - заголовки, строка 2 - данные.
    cols = [ws.cell(row=2, column=i) for i in range(1, 8)]
    названия, работодатель, город, навыки, опыт, зарплата_от, зарплата_до = cols

    for cell in (названия, работодатель, город, навыки, опыт):
        assert cell.data_type == "s", cell.coordinate
        assert not str(cell.value).startswith("="), cell.value
    assert названия.value != "=1+1"
    assert названия.value == "'=1+1"
    assert работодатель.value == "'@SUM(A1)"
    assert город.value == "'-5+cmd"
    assert навыки.value == "'\t=cmd"
    assert опыт.value == "normal"  # обычная строка не изменилась
    assert зарплата_от.value == 50000
    assert зарплата_от.data_type == "n"  # число осталось числом
    assert зарплата_до.value is None
