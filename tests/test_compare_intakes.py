"""Тесты чистых хелперов сравнения наборов (`scripts/compare_intakes.py`).

Не зависят от реальных данных проекта: классификация added/removed/kept, знак
дельты покрытия и порог значимости (0.05), нормализация имён с аббревиатурами и
грамотная обработка отсутствующего года (IntakeNotFoundError).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "compare_intakes", ROOT / "scripts" / "compare_intakes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_classify_added_removed_kept(mod):
    old = [
        {"name": "Математика", "coverage_ratio": 0.5, "gaps": 10, "coverage_level": "medium"},
        {"name": "Физика", "coverage_ratio": 0.3, "gaps": 20, "coverage_level": "low"},
    ]
    new = [
        {"name": "Математика", "coverage_ratio": 0.6, "gaps": 8, "coverage_level": "medium"},
        {"name": "Программирование", "coverage_ratio": 0.9, "gaps": 2, "coverage_level": "high"},
    ]
    report = mod.compare_intakes(old, new)

    assert [d["name"] for d in report["added"]] == ["Программирование"]
    assert [d["name"] for d in report["removed"]] == ["Физика"]
    assert [d["name"] for d in report["kept"]] == ["Математика"]
    assert report["summary"] == {
        "added": 1,
        "removed": 1,
        "kept": 1,
        "avg_coverage_old": pytest.approx(0.4),
        "avg_coverage_new": pytest.approx(0.75),
        "avg_coverage_delta": pytest.approx(0.35),
        "improved": 1,
        "worsened": 0,
        "unchanged": 0,
    }


def test_coverage_delta_sign_and_threshold(mod):
    # Рост на 0.06 >= порога 0.05 -> significant_change=True.
    up = mod.compare_intakes(
        [{"name": "A", "coverage_ratio": 0.10, "gaps": 1}],
        [{"name": "A", "coverage_ratio": 0.16, "gaps": 0}],
    )["kept"][0]
    assert up["coverage_delta"] == pytest.approx(0.06)
    assert up["significant_change"] is True

    # Падение на 0.02 < порога -> значимого изменения нет, знак отрицательный.
    down = mod.compare_intakes(
        [{"name": "A", "coverage_ratio": 0.50, "gaps": 1}],
        [{"name": "A", "coverage_ratio": 0.48, "gaps": 2}],
    )["kept"][0]
    assert down["coverage_delta"] == pytest.approx(-0.02)
    assert down["significant_change"] is False

    # Обобщение знаков на уровне сводки.
    report = mod.compare_intakes(
        [{"name": "A", "coverage_ratio": 0.5, "gaps": 0}, {"name": "B", "coverage_ratio": 0.5, "gaps": 0}],
        [{"name": "A", "coverage_ratio": 0.6, "gaps": 0}, {"name": "B", "coverage_ratio": 0.4, "gaps": 0}],
    )
    assert report["summary"]["improved"] == 1
    assert report["summary"]["worsened"] == 1
    assert report["summary"]["unchanged"] == 0


def test_name_normalization_abbrev_matches_official(mod):
    # Аббревиатура "ИИ и МО" должна совпасть с полным официальным названием.
    assert mod.norm("Генеративный ИИ и МО") == mod.norm(
        "Генеративный искусственный интеллект и машинное обучение"
    )
    report = mod.compare_intakes(
        [{"name": "Генеративный искусственный интеллект и машинное обучение", "coverage_ratio": 0.5, "gaps": 1}],
        [{"name": "Генеративный ИИ и МО", "coverage_ratio": 0.7, "gaps": 2}],
    )
    assert report["summary"]["kept"] == 1
    assert report["summary"]["added"] == 0
    assert report["summary"]["removed"] == 0
    kept = report["kept"][0]
    # Сопоставилось как одна дисциплина, но имя в новом наборе - аббревиатура,
    # поэтому сохраняется старое (полное) имя.
    assert kept["name"] == "Генеративный ИИ и МО"
    assert kept["old_name"] == "Генеративный искусственный интеллект и машинное обучение"


def test_missing_intake_graceful(mod, tmp_path):
    # Пустой каталог: года набора нет - должна быть понятная ошибка, а не крах.
    assert mod.summary_path("09.03.02", 1900, base=tmp_path) is None
    with pytest.raises(mod.IntakeNotFoundError) as exc_info:
        mod.build_intake_report("09.03.02", 1900, 2025, base=tmp_path)
    assert "1900" in str(exc_info.value)
    assert exc_info.value.label == "старый"
