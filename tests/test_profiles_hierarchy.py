"""Регресс бага «Тепловая карта: топ-15 навыков x уровни» (AnalyticsCharts):
колонка «Data Scientist (middle)» пуста при заполненном Base (junior).

Причина: коммит a504afbe (2FA hardening) случайно удалил эталонные профили
data/students/dc_competency.json и data/students/top_dc_competency.json,
а загрузчики профилей строили middle молча с нулём навыков:
/api/profiles/dc?full=true отдавал 200 + skills: [], из-за чего колонка
middle во frontend/src/app/components/AnalyticsCharts.tsx рисовалась
пустой, хотя middle обязан включать все навыки базового уровня.

Проверки:
1) реальные данные: файлы встроенных профилей существуют и коды вложены
   по уровням (base <= dc <= top_dc);
2) pipeline build_profiles (моковый payload) собирает middle из своих
   кодов + навыков base - иерархическое слияние, как уже сделано для top_dc.
"""

import json
from pathlib import Path

from src.pipeline.runner import build_profiles

BUILTIN_LEVELS = ("base", "dc", "top_dc")


def _student_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "students"


def _codes(name: str) -> list[str]:
    data = json.loads(
        (_student_dir() / f"{name}_competency.json").read_text(encoding="utf-8")
    )
    skills = data.get("навыки") or data.get("компетенции") or data.get("codes") or []
    if not skills:
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], str):
                skills = value
                break
    return skills


def test_builtin_profile_files_present():
    """После a504afbe файлы dc/top_dc пропали из ветки - middle/senior
    профили не резолвятся, колонки тепловой карты пусты."""
    for name in BUILTIN_LEVELS:
        path = _student_dir() / f"{name}_competency.json"
        assert path.exists(), f"нет {path.name}"


def test_builtin_levels_are_nested_by_codes():
    """Модель уровней (PROFILES_DISCIPLINES): middle >= junior, senior >= middle."""
    base, dc, top = (_codes(n) for n in BUILTIN_LEVELS)
    assert base, "base пуст"
    assert set(base) <= set(dc), "dc должен включать все коды base"
    assert set(dc) <= set(top), "top_dc должен включать все коды dc"


def test_build_profiles_middle_inherits_base_skills():
    """middle (dc) собирается из своих кодов + всех навыков base: иначе при
    пропаже dc_competency.json профиль пуст и колонка middle пуста."""
    all_codes = {"top_dc": ["T1"], "dc": ["D1"], "base": ["B1"]}
    mapping = {"T1": {"scikit-learn"}, "D1": {"pandas"}, "B1": {"python"}}
    profiles = build_profiles(all_codes, mapping)

    base_skills = set(profiles["base"].skills)
    dc_skills = set(profiles["dc"].skills)
    top_skills = set(profiles["top_dc"].skills)

    assert base_skills == {"python"}
    assert "pandas" in dc_skills, "свои навыки dc потерялись"
    assert base_skills <= dc_skills, "middle не наследует навыки base"
    assert {"python", "pandas"} <= top_skills
