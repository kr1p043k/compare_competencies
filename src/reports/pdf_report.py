"""PDF-отчёт по профилю: метрики, роли, рекомендации + PNG-графики с диска."""
import os
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def _dejavu() -> tuple[str, str]:
    import matplotlib

    base = Path(matplotlib.__file__).resolve().parent / "mpl-data" / "fonts" / "ttf"
    return str(base / "DejaVuSans.ttf"), str(base / "DejaVuSans-Bold.ttf")


def build_profile_pdf(profile: str, full_rec: dict, reports_dir: Path | str) -> bytes:
    """Собирает PDF из full_recommendations_*.json + готовых PNG. Возвращает байты."""
    from fpdf import FPDF

    regular, bold = _dejavu()
    if not (os.path.exists(regular) and os.path.exists(bold)):
        raise RuntimeError("DejaVu TTF not found for PDF cyrillic support")

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_font("DejaVu", "", regular)
    pdf.add_font("DejaVu", "B", bold)

    summary = full_rec.get("summary", {}) or {}
    date = datetime.now(timezone.utc).strftime("%d.%m.%Y")

    pdf.add_page()
    pdf.set_font("DejaVu", "B", 18)
    pdf.cell(0, 10, f"Отчёт по профилю: {profile}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 10)
    pdf.cell(0, 6, f"Сформирован {date} · целевой профиль: {full_rec.get('target_profession', '')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    def section(title: str) -> None:
        pdf.set_font("DejaVu", "B", 13)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def kv(label: str, value: str) -> None:
        pdf.set_font("DejaVu", "", 10)
        pdf.cell(60, 6, label)
        pdf.set_font("DejaVu", "B", 10)
        pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    section("Ключевые метрики")
    kv("Готовность (readiness)", f"{summary.get('readiness_score', 0):.1f}")
    kv("Покрытие рынка", f"{summary.get('market_coverage_score', 0):.1f}")
    kv("Покрытие строгое", f"{summary.get('coverage_strict', summary.get('market_skill_coverage', 0)):.1f}")
    kv("Покрытие взвешенное", f"{summary.get('coverage_weighted', summary.get('skill_coverage', 0)):.1f}")
    kv("Средний разрыв", f"{summary.get('avg_gap', 0):.1f}")
    kv("Соответствие рынку", f"{summary.get('match_score', 0):.1f}")
    pdf.ln(3)

    section("Ближайшие роли")
    pdf.set_font("DejaVu", "B", 9)
    pdf.cell(85, 6, "Роль")
    pdf.cell(25, 6, "Сходство")
    pdf.cell(25, 6, "Покрытие")
    pdf.cell(25, 6, "Уровень")
    pdf.cell(0, 6, "Ранг", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 9)
    for r in (full_rec.get("closest_roles", []) or [])[:5]:
        pdf.cell(85, 6, str(r.get("role", ""))[:60])
        pdf.cell(25, 6, f"{r.get('semantic_similarity', 0):.1f}%")
        pdf.cell(25, 6, str(r.get("skills_covered", "")))
        pdf.cell(25, 6, str(r.get("cluster_level", "")))
        pdf.cell(0, 6, f"{r.get('rank_score', 0):.3f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    section("Топ рекомендаций")
    pdf.set_font("DejaVu", "B", 9)
    pdf.cell(10, 6, "#")
    pdf.cell(55, 6, "Навык")
    pdf.cell(30, 6, "Важность")
    pdf.cell(45, 6, "База + бонус")
    pdf.cell(0, 6, "Приоритет", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 9)
    for rec in (full_rec.get("recommendations", []) or [])[:15]:
        base = rec.get("importance_base", rec.get("importance_score", 0))
        bonus = rec.get("importance_bonus", 0)
        pdf.cell(10, 6, str(rec.get("rank", "")))
        pdf.cell(55, 6, str(rec.get("skill", ""))[:38])
        pdf.cell(30, 6, f"{rec.get('importance_score', 0) * 100:.1f}")
        pdf.cell(45, 6, f"{base * 100:.0f}+{bonus * 100:.0f}")
        pdf.cell(0, 6, str(rec.get("priority", "")), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("DejaVu", "", 8)
    pdf.multi_cell(0, 5, "Важность — композитный балл (blend gap-анализа и ML + бонусы тренда, домена, роли), а не доля вакансий.")

    reports = Path(reports_dir)
    prof_pngs = [
        reports / profile / f"radar_{profile}.png",
        reports / profile / f"cluster_insights_{profile}.png",
        reports / profile / f"ml_importance_{profile}.png",
    ]
    global_pngs = [
        reports / "skills_heatmap.png",
        reports / "skill_correlation_heatmap.png",
        reports / "coverage_comparison.png",
    ]
    for png in prof_pngs + global_pngs:
        if png.exists():
            pdf.add_page()
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 8, png.stem.replace("_", " "), new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(png), x=10, w=190)
            logger.info("pdf_chart_embedded", chart=png.name)

    out = pdf.output()
    return bytes(out)
