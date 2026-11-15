"""Консистентность whitelist: it_skills.json (канон) vs skill_taxonomy.json.

Таксономия не должна содержать навыков вне канона, кроме заморожённого
списка известного дрейфа. Появление НОВЫХ расхождений ломает тест
(регресс-щит против рассогласования источников правды).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IT_SKILLS = ROOT / "data" / "reference" / "it_skills.json"
TAXONOMY = ROOT / "data" / "reference" / "skill_taxonomy.json"

# Замороженный дрейф: навыки только в таксономии, отсутствующие в каноне.
# Уменьшать при синхронизации; любые НОВЫЕ записи = провал теста.
KNOWN_TAXONOMY_ONLY = {
    "1c предприятие 8", "3d modelling", "apache hadoop hdfs", "apache kafka streams",
    "archimate", "arduino", "cryengine", "cto", "digitalocean", "dynamodb",
    "embedded linux", "express", "game design", "game physics", "geoserver", "godot",
    "heroku", "ips", "knime", "kubeflow", "nat", "product management",
    "project management", "pwa", "raspberry pi", "remote sensing", "salesforce",
    "stm32", "team lead", "tech lead", "usergate", "vercel", "verilog", "vhdl",
    "xamarin", "большие языковые модели исследования", "критическое мышление",
    "лидерство", "линейная алгебра", "математическое моделирование",
    "презентационные навыки", "реактивное программирование", "решение проблем",
    "тайм-менеджмент", "тестирование по", "численные методы",
}


def _canon() -> set[str]:
    data = json.loads(IT_SKILLS.read_text(encoding="utf-8"))
    return {s.strip().lower() for s in data}


def _taxonomy_skills() -> set[str]:
    data = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    out: set[str] = set()
    for info in data.get("categories", {}).values():
        for skill in info.get("skills", []):
            out.add(str(skill).strip().lower())
    return out


def test_taxonomy_extra_matches_frozen_drift() -> None:
    extra = _taxonomy_skills() - _canon()
    new = extra - KNOWN_TAXONOMY_ONLY
    resolved = KNOWN_TAXONOMY_ONLY - extra
    assert not new, f"новые навыки вне канона: {sorted(new)}"
    assert not resolved, f"дрейф устранён (убрать из KNOWN_TAXONOMY_ONLY): {sorted(resolved)}"


def test_it_skills_has_no_case_insensitive_duplicates() -> None:
    raw = json.loads(IT_SKILLS.read_text(encoding="utf-8"))
    lowered = [s.strip().lower() for s in raw]
    dups = sorted({x for x in lowered if lowered.count(x) > 1})
    assert dups == [], f"дубли канона: {dups}"
