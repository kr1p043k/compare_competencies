"""Покрытие src/reports/pdf_report.build_profile_pdf на синтетических данных.

Существующий tests/api/test_report_pdf.py скипается без реального
full_recommendations_*.json. Здесь собираем PDF из словаря в памяти, поэтому
билдер реально исполняется: KPI, алерты, таблицы ролей и рекомендаций,
встраивание PNG-графиков и подвалы страниц.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fpdf")
pytest.importorskip("matplotlib")

from src.reports import pdf_report  # noqa: E402


def _full_rec(*, low: bool = False) -> dict:
    if low:
        summary = {
            "readiness_score": 10.0,
            "market_coverage_score": 5.0,
            "coverage_strict": 4.0,
            "avg_gap": 55.0,
        }
    else:
        summary = {
            "readiness_score": 72.0,
            "market_coverage_score": 64.0,
            "coverage_strict": 60.0,
            "avg_gap": 12.0,
        }
    return {
        "target_profession": "Backend-разработчик",
        "summary": summary,
        "closest_roles": [
            {"role": "Backend Developer", "semantic_similarity": 91.2,
             "skills_covered": 12, "cluster_level": "middle", "rank_score": 0.91},
            {"role": "Data Engineer", "semantic_similarity": 74.0,
             "skills_covered": 8, "cluster_level": "middle", "rank_score": 0.74},
        ],
        "recommendations": [
            {"rank": 1, "skill": "python", "importance_score": 0.9,
             "importance_base": 0.8, "importance_bonus": 0.1, "priority": "Высокий"},
            {"rank": 2, "skill": "docker", "importance_score": 0.7,
             "importance_base": 0.7, "importance_bonus": 0.0, "priority": "Средний"},
        ],
    }


def test_build_pdf_high_summary(tmp_path: Path):
    data = pdf_report.build_profile_pdf("base", _full_rec(), tmp_path)
    assert isinstance(data, bytes)
    assert data.startswith(b"%PDF")
    assert len(data) > 1000


def test_build_pdf_low_summary_triggers_alerts(tmp_path: Path):
    data = pdf_report.build_profile_pdf("base", _full_rec(low=True), tmp_path)
    assert data.startswith(b"%PDF")
    # Блок алертов добавляет строки - файл не пустой и валиден.
    assert len(data) > 1000


def test_build_pdf_handles_missing_sections(tmp_path: Path):
    # Пустой словарь: ни summary, ни ролей, ни рекомендаций - не должно падать.
    data = pdf_report.build_profile_pdf("base", {}, tmp_path)
    assert data.startswith(b"%PDF")


def test_build_pdf_embeds_existing_charts(tmp_path: Path):
    # Кладём простой PNG туда, где билдер его ищет, - картинка встраивается.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof_dir = tmp_path / "base"
    prof_dir.mkdir(parents=True, exist_ok=True)
    png = prof_dir / "cluster_insights_base.png"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(png, dpi=80)
    plt.close(fig)

    data = pdf_report.build_profile_pdf("base", _full_rec(), tmp_path)
    assert data.startswith(b"%PDF")
    assert data.count(b"/Subtype /Image") >= 1


def test_build_pdf_falls_back_to_profile_name(tmp_path: Path):
    rec = _full_rec()
    rec.pop("target_profession")
    data = pdf_report.build_profile_pdf("myprofile", rec, tmp_path)
    assert data.startswith(b"%PDF")


def _pdf_text(blob: bytes) -> str:
    import io

    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(blob))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def test_pdf_contains_key_sections(tmp_path: Path):
    # Синтетика, повторяющая структуру full_recommendations_*.json:
    # и разделы, и ключевой текст (включая emoji в why_important) рендерятся.
    rec = _full_rec()
    rec["gaps"] = {
        "sql": {"skill": "sql", "importance": 0.4463, "category": "missing"},
        "python": {"skill": "python", "importance": 0.4136, "category": "missing"},
        "llm": {"skill": "llm", "importance": 0.1076, "category": "weak"},
    }
    rec["domain_coverage"] = {
        "Backend": {"domain": "Backend", "user_has": 17, "total_required": 138,
                    "coverage": 0.123, "importance": 0.0769},
    }
    rec["recommendations"] = [
        {"rank": 1, "skill": "python", "importance_score": 0.99,
         "importance_base": 0.65, "importance_bonus": 0.34, "priority": "HIGH",
         "why_important": "🔴 Высокий спрос на Python",
         "how_to_learn": "Практика: 10+ задач",
         "expected_timeframe": "1-2 месяца", "expected_outcome": "рост"},
    ]

    data = pdf_report.build_profile_pdf("base", rec, tmp_path)
    text = _pdf_text(data)

    assert "Топ-пробелы" in text
    assert "Покрытие по доменам" in text
    assert "Рекомендации" in text
    # emoji убран, а текст после него сохранён и извлекается.
    assert "Высокий спрос" in text
    assert "отсутствует" in text  # статус пробела "missing" локализован
