"""PDF-отчёт по профилю: editorial-вёрстка (impeccable restrained + theme-factory discipline).

Тема: чистый белый фон, чернила off-black, один акцент - фирменный синий
приложения (#1D4ED8), воздух и иерархия вместо рамок везде. Кириллица -
DejaVu из matplotlib.

Отчёт собирает из full_recommendations_*.json связный документ:
сводку, таблицу метрик, сигналы, покрытие по доменам, топ-пробелы,
ближайшие роли, подробные рекомендации и PNG-графики.
"""
from __future__ import annotations

import os
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

INK = (15, 23, 42)
MUTED = (100, 116, 139)
ACCENT = (29, 78, 216)
BAND = (239, 246, 255)
ZEBRA = (248, 250, 252)
RULE = (226, 232, 240)
CRIT = (176, 0, 32)
WARN = (176, 106, 0)

_PRIORITY_RU = {"HIGH": "Высокий", "MEDIUM": "Средний", "LOW": "Низкий"}
_GAP_RU = {"missing": "отсутствует", "weak": "слабый", "strong": "сильный"}


def _dejavu() -> tuple[str, str]:
    import matplotlib

    base = Path(matplotlib.__file__).resolve().parent / "mpl-data" / "fonts" / "ttf"
    return str(base / "DejaVuSans.ttf"), str(base / "DejaVuSans-Bold.ttf")


def _num(value: Any, default: float = 0.0) -> float:
    """Безопасное приведение к float: кривые/None значения не ломают вёрстку."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean(text: Any) -> str:
    """Строка для DejaVu: удаляет emoji (Unicode category So), которые шрифт не рендерит,
    и схлопывает горизонтальные пробелы."""
    s = "" if text is None else str(text)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "So")
    return re.sub(r"[^\S\n]+", " ", s).strip()


def build_profile_pdf(profile: str, full_rec: dict[str, Any], reports_dir: Path | str) -> bytes:
    """Собирает PDF из full_recommendations_*.json + готовых PNG. Возвращает байты."""
    from fpdf import FPDF

    regular, bold = _dejavu()
    if not (os.path.exists(regular) and os.path.exists(bold)):
        raise RuntimeError("DejaVu TTF not found for PDF cyrillic support")

    class Report(FPDF):
        def footer(self) -> None:
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
    cov_details = summary.get("coverage_details", {}) or {}
    date = datetime.now(UTC).strftime("%d.%m.%Y")
    target = _clean(full_rec.get("target_profession")) or profile

    readiness = _num(summary.get("readiness_score"))
    match_cov = _num(summary.get("match_score"))
    strict = _num(summary.get("coverage_strict", summary.get("market_skill_coverage")))
    weighted = _num(summary.get("coverage_weighted", summary.get("skill_coverage")))
    domain_cov = _num(summary.get("domain_coverage_score"))
    prof_cov = _num(summary.get("profession_coverage"))
    covered = _num(cov_details.get("covered_skills_count"))
    total_market = _num(cov_details.get("total_market_skills"))

    # Нормализация коллекций (в данных это dict, в тестах могут быть списки).
    gaps_src = full_rec.get("gaps", {}) or {}
    gaps: list[dict[str, Any]] = (
        [g for g in gaps_src.values() if isinstance(g, dict)]
        if isinstance(gaps_src, dict) else list(gaps_src)
    )
    dc_src = full_rec.get("domain_coverage", {}) or {}
    domains: list[dict[str, Any]] = (
        [d for d in dc_src.values() if isinstance(d, dict)]
        if isinstance(dc_src, dict) else list(dc_src)
    )
    roles: list[dict[str, Any]] = list(full_rec.get("closest_roles", []) or [])
    recs: list[dict[str, Any]] = list(full_rec.get("recommendations", []) or [])

    # === Вспомогательные функции - вложенные, замыкаются на pdf. ===

    def section(title: str) -> None:
        pdf.set_font("DejaVu", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*RULE)
        pdf.set_line_width(0.6)
        pdf.line(18, pdf.get_y(), 192, pdf.get_y())
        pdf.ln(3)

    def note(text: str) -> None:
        pdf.set_font("DejaVu", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 5, text)
        pdf.ln(2)

    def lines_of(cell: str, w: float) -> int:
        res = pdf.multi_cell(w, 6, cell, dry_run=True, output="LINES")
        return len(res) if isinstance(res, list) else 1

    def table(head: list[tuple[str, float]], rows: list[list[str]]) -> None:
        widths = [w for _, w in head]
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_fill_color(*ACCENT)
        pdf.set_text_color(255, 255, 255)
        for title, w in head:
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
            max_lines = max(lines_of(cell, w) for cell, w in zip(row, widths, strict=False))
            h = max(6, max_lines * 6)
            y0 = pdf.get_y()
            if y0 + h > 277:
                pdf.add_page()
                y0 = pdf.get_y()
            x0: float = 18.0
            for cell, w in zip(row, widths, strict=False):
                pdf.set_xy(x0, y0)
                pdf.multi_cell(w, 6, cell, border=0, fill=fill)
                x0 += w
            pdf.set_y(y0 + h)

    def labeled(label: str, body: Any) -> None:
        text = _clean(body)
        if not text:
            return
        pdf.set_font("DejaVu", "B", 9)
        pdf.set_text_color(*INK)
        pdf.cell(0, 6, label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("DejaVu", "", 9)
        pdf.multi_cell(0, 5, text)

    # === Обложка. ===
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

    # === Сводка (narrative). ===
    section("Сводка")
    covered_part = f"{covered:.0f} из {total_market:.0f} рыночных навыков" if total_market else "н/д"
    missing_skills = sorted(
        (g for g in gaps if (_clean(g.get("category")) == "missing")),
        key=lambda g: _num(g.get("importance")),
        reverse=True,
    )
    narrative = (
        f"Профиль «{target}» готов к целевому рынку на {readiness:.1f} из 100, "
        f"соответствует целевой профессии на {match_cov:.1f}. "
        f"Строгое покрытие рынка - {strict:.1f} ({covered_part}), взвешенное - {weighted:.1f}. "
        f"Покрытие доминирующего домена - {domain_cov:.1f}, целевой профессии - {prof_cov:.1f}."
    )
    if missing_skills:
        top = ", ".join(_clean(g.get("skill")) for g in missing_skills[:3])
        narrative += f" Ключевые пробелы: {top}."
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 6, narrative)
    pdf.ln(2)

    # === Сигналы (алерты). ===
    alerts: list[tuple[str, str]] = []
    if readiness < 30:
        alerts.append(("Критично", "готовность ниже 30 - программе не хватает базового покрытия"))
    if strict < 20:
        alerts.append(("Внимание", "строгое покрытие ниже 20 - язык программы расходится с рынком"))
    if prof_cov < 20 and prof_cov > 0:
        alerts.append(("Внимание", "покрытие целевой профессии ниже 20 - мало пересечения с целевой ролью"))
    if _num(summary.get("avg_gap")) > 40:
        alerts.append(("Внимание", "средний разрыв выше 40 - много слабых навыков"))
    important_missing = [
        g for g in missing_skills if _num(g.get("importance")) >= 0.30
    ]
    if important_missing:
        names = ", ".join(_clean(g.get("skill")) for g in important_missing[:3])
        alerts.append(("Внимание", f"в программе отсутствуют востребованные навыки: {names}"))
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
            pdf.set_text_color(*(CRIT if level == "Критично" else WARN))
            pdf.cell(24, 6, level + ":", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("DejaVu", "", 9)
            pdf.set_text_color(*INK)
            pdf.multi_cell(0, 6, text)
        pdf.ln(2)

    # === Ключевые метрики. ===
    section("Ключевые метрики")
    covered_str = f"{covered:.0f} / {total_market:.0f}" if total_market else "н/д"
    table(
        [("Метрика", 130), ("Значение", 44)],
        [
            ["Готовность (readiness)", f"{readiness:.1f}"],
            ["Соответствие целевой профессии", f"{match_cov:.1f}"],
            ["Строгое покрытие рынка", f"{strict:.1f}"],
            ["Взвешенное покрытие рынка", f"{weighted:.1f}"],
            ["Покрытие домена", f"{domain_cov:.1f}"],
            ["Покрытие целевой профессии", f"{prof_cov:.1f}"],
            ["Покрыто рыночных навыков", covered_str],
        ],
    )
    note("Readiness = 0.45 × рынок + 0.30 × сильные% − 0.25 × слабые%. "
         "Строгое покрытие - бинарное пересечение программы со спросом; взвешенное - с учётом веса спроса.")

    # === Покрытие по доменам. ===
    section("Покрытие по доменам")
    domain_rows: list[list[str]] = []
    for d in sorted(domains, key=lambda d: (-_num(d.get("importance")), -_num(d.get("coverage")))):
        name = _clean(d.get("domain"))
        has = _num(d.get("user_has"))
        total = _num(d.get("total_required"))
        domain_rows.append([
            name[:40],
            f"{_num(d.get('coverage')) * 100:.1f}%",
            f"{has:.0f} / {total:.0f}",
            f"{_num(d.get('importance')) * 100:.1f}",
        ])
    if domain_rows:
        table(
            [("Домен", 64), ("Покрытие", 28), ("Навыки", 44), ("Важность", 38)],
            domain_rows,
        )
        note("Покрытие - доля навыков домена, уже закрытых программой. "
             "Важность - вес домена в целевой нише (сумма весов доменов не нормирована к 100).")
    else:
        note("Данные о покрытии по доменам отсутствуют.")

    # === Топ-пробелы. ===
    section("Топ-пробелы")
    gap_rows: list[list[str]] = []
    for g in sorted(
        (g for g in gaps if (_clean(g.get("category")) in ("missing", "weak"))),
        key=lambda g: _num(g.get("importance")),
        reverse=True,
    )[:12]:
        gap_rows.append([
            _clean(g.get("skill"))[:44],
            _GAP_RU.get(_clean(g.get("category")), _clean(g.get("category"))),
            f"{_num(g.get('importance')) * 100:.1f}",
        ])
    if gap_rows:
        table(
            [("Навык", 90), ("Статус", 44), ("Важность", 40)],
            gap_rows,
        )
        note("Показаны навыки со статусом «отсутствует» и «слабый», отсортированные по важности. "
             "Важность - относительный вес навыка в спросе (0-100).")
    else:
        note("Явных пробелов не выявлено.")

    # === Ближайшие роли. ===
    section("Ближайшие роли")
    role_rows = [
        [_clean(r.get("role"))[:52],
         f"{_num(r.get('semantic_similarity')):.1f}%",
         _clean(r.get("skills_covered")),
         _clean(r.get("cluster_level")),
         f"{_num(r.get('rank_score')):.3f}"]
        for r in roles[:5]
    ]
    if role_rows:
        table(
            [("Роль", 80), ("Сходство", 25), ("Покрытие", 25), ("Уровень", 22), ("Ранг", 22)],
            role_rows,
        )
        pdf.ln(3)
    else:
        note("Данные о ближайших ролях отсутствуют.")

    # === Рекомендации. ===
    section("Рекомендации")
    detailed = recs[:6]
    for rec in detailed:
        rank = rec.get("rank", "")
        skill = _clean(rec.get("skill"))
        priority = _PRIORITY_RU.get(_clean(rec.get("priority")), _clean(rec.get("priority")))
        base = _num(rec.get("importance_base"), _num(rec.get("importance_score")))
        bonus = _num(rec.get("importance_bonus"))
        pdf.set_font("DejaVu", "B", 11)
        pdf.set_text_color(*ACCENT)
        pdf.cell(0, 7, f"{rank}. {skill}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("DejaVu", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 6, f"Приоритет: {priority} · Состав: база {base * 100:.0f} + бонус {bonus * 100:.0f}",
                 new_x="LMARGIN", new_y="NEXT")
        labeled("Зачем этот навык:", rec.get("why_important"))
        labeled("Как освоить:", rec.get("how_to_learn"))
        timeframe = _clean(rec.get("expected_timeframe"))
        outcome = _clean(rec.get("expected_outcome"))
        if timeframe and outcome:
            labeled("Срок и результат:", f"{timeframe} · {outcome}")
        elif timeframe:
            labeled("Срок:", timeframe)
        elif outcome:
            labeled("Ожидаемый результат:", outcome)
        pdf.ln(3)
    rest = recs[6:15]
    if rest:
        pdf.set_font("DejaVu", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 6, "Остальные рекомендации:", new_x="LMARGIN", new_y="NEXT")
        table(
            [("#", 10), ("Навык", 54), ("Балл", 30), ("База+бонус", 36), ("Приоритет", 44)],
            [[str(rec.get("rank", "")),
              _clean(rec.get("skill"))[:34],
              f"{_num(rec.get('importance_score')) * 100:.1f}",
              f"{_num(rec.get('importance_base'), _num(rec.get('importance_score'))) * 100:.0f}"
              f"+{_num(rec.get('importance_bonus')) * 100:.0f}",
              _PRIORITY_RU.get(_clean(rec.get("priority")), _clean(rec.get("priority")))]
             for rec in rest],
        )
    note("Важность - композитный балл (blend gap-анализа и ML + бонусы тренда, домена, роли), а не доля вакансий.")

    # === Графики. ===
    reports = Path(reports_dir)
    charts: list[tuple[str, Path]] = [
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

    data = pdf.output()
    if data is None:
        raise RuntimeError("empty PDF output")
    return bytes(data)
