"""PDF-отчёт по профилю: editorial-вёрстка (impeccable restrained + theme-factory discipline).

Тема: светлый папирусный фон запрещён (AI-слоп); берём чистый белый,
чернила off-black, один акцент — фирменный синий приложения (#1D4ED8),
воздух и иерархия вместо рамок везде. Кириллица — DejaVu из matplotlib.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

INK = (15, 23, 42)
MUTED = (100, 116, 139)
ACCENT = (29, 78, 216)
BAND = (239, 246, 255)
ZEBRA = (248, 250, 252)
RULE = (226, 232, 240)


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

    class Report(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("DejaVu", "", 8)
            self.set_text_color(*MUTED)
            self.cell(0, 10, f"Competency Gap Analyzer · стр. {self.page_no()}/{{nb}}",
                      align="C")

    pdf = Report(format="A4")
    pdf.alias_nb_pages("{nb}")
    pdf.set_auto_page_break(True, margin=20)
    pdf.set_margins(18, 15, 18)
    pdf.add_font("DejaVu", "", regular)
    pdf.add_font("DejaVu", "B", bold)

    summary = full_rec.get("summary", {}) or {}
    date = datetime.now(timezone.utc).strftime("%d.%m.%Y")
    target = full_rec.get("target_profession", "") or profile

    # Обложечный блок: акцентная полоса + титул.
    pdf.add_page()
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, 0, 210, 52, style="F")
    pdf.set_xy(18, 12)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(219, 234, 254)
    pdf.cell(0, 6, "COMPETENCY GAP ANALYZER", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(18)
    pdf.set_font("DejaVu", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, 10, f"Отчёт по профилю\n{target}")
    pdf.set_x(18)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(219, 234, 254)
    pdf.cell(0, 6, f"Сформирован {date}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(60)

    def section(title: str) -> None:
        pdf.set_font("DejaVu", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*RULE)
        pdf.set_line_width(0.6)
        pdf.line(18, pdf.get_y(), 192, pdf.get_y())
        pdf.ln(3)

    def kpi_row(cells: list[tuple[str, str]]) -> None:
        pdf.set_font("DejaVu", "B", 16)
        w = 174 / max(len(cells), 1)
        y0 = pdf.get_y()
        pdf.set_draw_color(*RULE)
        for i, (label, value) in enumerate(cells):
            x = 18 + i * w
            pdf.rect(x, y0, w, 24)
            pdf.set_xy(x, y0 + 3)
            pdf.set_text_color(*ACCENT)
            pdf.cell(w, 9, value, align="C")
            pdf.set_xy(x, y0 + 13)
            pdf.set_font("DejaVu", "", 8)
            pdf.set_text_color(*MUTED)
            pdf.cell(w, 6, label, align="C")
            pdf.set_font("DejaVu", "B", 16)
        pdf.set_y(y0 + 28)

    def table(head: list[tuple[str, float]], rows: list[list[str]]) -> None:
        widths = [w for _, w in head]
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_fill_color(*ACCENT)
        pdf.set_text_color(255, 255, 255)
        for (title, w) in head:
            pdf.cell(w, 7, title, border=0, fill=True)
        pdf.ln()
        pdf.set_font("DejaVu", "", 9)
        pdf.set_text_color(*INK)
        for i, row in enumerate(rows):
            if i % 2:
                pdf.set_fill_color(*ZEBRA)
                fill = True
            else:
                fill = False
            max_lines = 1
            for (cell, w) in zip(row, widths):
                lines = pdf.multi_cell(w, 6, cell, dry_run=True, output="LINES")
                max_lines = max(max_lines, len(lines))
            h = max(6, max_lines * 6)
            y0 = pdf.get_y()
            if y0 + h > 277:
                pdf.add_page()
                y0 = pdf.get_y()
            x0 = 18
            for (cell, w) in zip(row, widths):
                pdf.set_xy(x0, y0)
                pdf.multi_cell(w, 6, cell, border=0, fill=fill)
                x0 += w
            pdf.set_y(y0 + h)

    section("Ключевые метрики")
    kpi_row([
        ("Готовность", f"{summary.get('readiness_score', 0):.1f}"),
        ("Покрытие рынка", f"{summary.get('market_coverage_score', 0):.1f}"),
        ("Строгое", f"{summary.get('coverage_strict', summary.get('market_skill_coverage', 0)):.1f}"),
        ("Разрыв", f"{summary.get('avg_gap', 0):.1f}"),
    ])
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5, "Readiness = 0.45 × рынок + 0.30 × сильные% − 0.25 × слабые%. "
                         "Строгое — бинарное пересечение со спросом; взвешенное — с учётом спроса.")
    pdf.ln(2)

    alerts: list[tuple[str, str]] = []
    if summary.get("readiness_score", 100) < 30:
        alerts.append(("Критично", "готовность ниже 30 — программе не хватает базового покрытия"))
    if summary.get("coverage_strict", summary.get("market_skill_coverage", 100)) < 20:
        alerts.append(("Внимание", "строгое покрытие ниже 20 — язык программы расходится с рынком"))
    if summary.get("avg_gap", 0) > 40:
        alerts.append(("Внимание", "средний разрыв выше 40 — много слабых навыков"))
    if alerts:
        pdf.set_font("DejaVu", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(0, 8, "Сигналы", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*RULE)
        pdf.set_line_width(0.6)
        pdf.line(18, pdf.get_y(), 192, pdf.get_y())
        pdf.ln(2)
        for level, text in alerts:
            pdf.set_font("DejaVu", "B", 9)
            pdf.set_text_color(*(176, 0, 32) if level == "Критично" else (176, 106, 0))
            pdf.cell(22, 6, level + ":")
            pdf.set_font("DejaVu", "", 9)
            pdf.set_text_color(*INK)
            pdf.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    section("Ближайшие роли")
    table(
        [("Роль", 80), ("Сходство", 25), ("Покрытие", 25), ("Уровень", 22), ("Ранг", 22)],
        [[str(r.get("role", ""))[:52],
          f"{r.get('semantic_similarity', 0):.1f}%",
          str(r.get("skills_covered", "")),
          str(r.get("cluster_level", "")),
          f"{r.get('rank_score', 0):.3f}"]
         for r in (full_rec.get("closest_roles", []) or [])[:5]],
    )
    pdf.ln(3)

    section("Топ рекомендаций")
    table(
        [("#", 10), ("Навык", 62), ("Важность", 28), ("База + бонус", 36), ("Приоритет", 38)],
        [[str(rec.get("rank", "")),
          str(rec.get("skill", ""))[:34],
          f"{rec.get('importance_score', 0) * 100:.1f}",
          f"{rec.get('importance_base', rec.get('importance_score', 0)) * 100:.0f}"
          f"+{rec.get('importance_bonus', 0) * 100:.0f}",
          str(rec.get("priority", ""))]
         for rec in (full_rec.get("recommendations", []) or [])[:15]],
    )
    pdf.ln(2)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5, "Важность — композитный балл (blend gap-анализа и ML + бонусы тренда, домена, роли), а не доля вакансий.")

    reports = Path(reports_dir)
    charts: list[tuple[str, Path]] = [
        ("Радар: профиль vs рынок", reports / profile / f"radar_{profile}.png"),
        ("Ближайшие кластеры", reports / profile / f"cluster_insights_{profile}.png"),
        ("Важность навыков (ML)", reports / profile / f"ml_importance_{profile}.png"),
        ("Покрытие навыков", reports / "skills_heatmap.png"),
        ("Совместная встречаемость", reports / "skill_correlation_heatmap.png"),
        ("Сравнение покрытия", reports / "coverage_comparison.png"),
    ]
    for title, png in charts:
        if not png.exists():
            continue
        pdf.add_page()
        pdf.set_font("DejaVu", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*RULE)
        pdf.set_line_width(0.6)
        pdf.line(18, pdf.get_y(), 192, pdf.get_y())
        pdf.ln(3)
        pdf.image(str(png), x=18, w=174)
        logger.info("pdf_chart_embedded", chart=png.name)

    out = pdf.output()
    return bytes(out)
