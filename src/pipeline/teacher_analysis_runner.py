"""Main runner: loads DB data → analyzers → predictors → data/result/teacher/."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import structlog

from src.result import Ok, Err, Result
from src.errors import AnalysisDataError, AnalysisRunnerError, RecommendationError
from src.db import create_pool, close_pool, get_pool
from src.models.teacher_analysis import DirectionSummary, GapAnalysisResult
from src.analyzers.skill_matcher import SkillMatcher, normalize as normalize_skill
from src.analyzers.coverage_analyzer import CoverageAnalyzer
from src.analyzers.trend_analyzer import TrendAnalyzer
from src.predictors.curriculum_recommender import CurriculumRecommender
from src.predictors.curriculum_optimizer import CurriculumOptimizer

logger = structlog.get_logger(__name__)
OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "result" / "teacher"


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()[:80]


async def run_teacher_analysis(
    direction_code: str | None = None,
    discipline_filter: str | None = None,
) -> Result[dict, AnalysisRunnerError]:
    logger.info("analysis_started", direction=direction_code, discipline=discipline_filter)

    # — DB connection —
    await create_pool()
    pool = get_pool()
    if not pool:
        logger.error("no_db_pool")
        return Err(AnalysisRunnerError(stage="db", message="Failed to create database pool"))

    # — load market skills (HH data + it_skills taxonomy) —
    try:
        mrows = await pool.fetch(
            """SELECT DISTINCT ON (LOWER(market_skill_name)) market_skill_name, frequency
               FROM market_skill_mappings
               ORDER BY LOWER(market_skill_name), frequency DESC"""
        )
    except Exception as exc:
        logger.error("market_skills_query_failed", error=str(exc))
        await close_pool()
        return Err(AnalysisRunnerError(stage="market_skills", message=str(exc)))

    market_skills: dict[str, int] = {}
    for r in mrows:
        k = r["market_skill_name"].lower().strip()
        if k:
            market_skills[k] = r["frequency"]

    # Merge it_skills taxonomy into market skills (use freq=1 as base for taxonomy-only skills)
    it_skills_path = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "it_skills.json"
    if it_skills_path.exists():
        import json
        with open(it_skills_path, "r", encoding="utf-8") as f:
            it_data = json.load(f)
        for name in it_data:
            k = name.strip().lower()
            if k and k not in market_skills:
                market_skills[k] = 1

    if not market_skills:
        logger.warning("no_market_skills_found")
    logger.info("market_skills_loaded", count=len(market_skills), from_hh=len(mrows))

    # — load disciplines —
    dir_filter = ""
    params: list = []
    if direction_code:
        dir_filter = " AND d.code = $" + str(len(params) + 1)
        params.append(direction_code)
    disc_filter = ""
    if discipline_filter:
        disc_filter = " AND d2.name ILIKE $" + str(len(params) + 1)
        params.append(f"%{discipline_filter}%")

    try:
        drows = await pool.fetch(
            f"""SELECT d2.id AS disc_id, d2.name AS disc_name,
                       c.id AS comp_id, c.code AS comp_code,
                       s.name AS skill_name,
                       k.original_text AS ksa_text, k.ksa_type
                FROM directions d
                JOIN disciplines d2 ON d2.direction_id = d.id
                JOIN competencies c ON c.discipline_id = d2.id
                LEFT JOIN competency_skills cs ON cs.competency_id = c.id
                LEFT JOIN skills s ON s.id = cs.skill_id
                LEFT JOIN ksa_entries k ON k.competency_id = c.id
                WHERE 1=1{dir_filter}{disc_filter}
                ORDER BY d2.name, c.code, k.sort_order""",
            *params,
        )
    except Exception as exc:
        logger.error("discipline_query_failed", error=str(exc))
        await close_pool()
        return Err(AnalysisRunnerError(stage="disciplines", message=str(exc)))

    disciplines: dict[str, dict] = {}
    for r in drows:
        dn = r["disc_name"]
        if dn not in disciplines:
            disciplines[dn] = {"id": str(r["disc_id"]), "competencies": {}}
        cc = r["comp_code"]
        if cc not in disciplines[dn]["competencies"]:
            disciplines[dn]["competencies"][cc] = set()
        if r["skill_name"]:
            disciplines[dn]["competencies"][cc].add(r["skill_name"])
        if r["ksa_text"]:
            disciplines[dn]["competencies"][cc].add(r["ksa_text"])
    # Convert sets to lists for downstream
    for dn in disciplines:
        for cc in disciplines[dn]["competencies"]:
            disciplines[dn]["competencies"][cc] = list(disciplines[dn]["competencies"][cc])

    if not disciplines:
        logger.error("no_disciplines_loaded", direction=direction_code)
        await close_pool()
        return Err(AnalysisRunnerError(
            stage="disciplines",
            message=f"No disciplines found for direction={direction_code}",
        ))
    logger.info("disciplines_loaded", count=len(disciplines))

    # — load direction meta —
    try:
        if direction_code:
            direction = await pool.fetchrow(
                "SELECT id, code, name, profile FROM directions WHERE code=$1", direction_code
            )
        else:
            direction = await pool.fetchrow(
                "SELECT id, code, name, profile FROM directions LIMIT 1"
            )
    except Exception as exc:
        logger.error("direction_query_failed", error=str(exc))
        await close_pool()
        return Err(AnalysisRunnerError(stage="direction", message=str(exc)))

    if not direction:
        logger.error("no_direction_found")
        await close_pool()
        return Err(AnalysisRunnerError(
            stage="direction",
            message=f"Direction '{direction_code}' not found",
        ))

    # — load trend snapshots —
    try:
        srows = await pool.fetch(
            "SELECT snapshot_date, skill_freq FROM trend_snapshots ORDER BY snapshot_date ASC"
        )
    except Exception as exc:
        logger.error("snapshots_query_failed", error=str(exc))
        await close_pool()
        return Err(AnalysisRunnerError(stage="snapshots", message=str(exc)))

    snapshots = []
    for r in srows:
        d = dict(r)
        if isinstance(d.get("skill_freq"), str):
            d["skill_freq"] = json.loads(d["skill_freq"])
        snapshots.append(d)

    await close_pool()
    logger.info("snapshots_loaded", count=len(snapshots))

    # — init services —
    matcher = SkillMatcher(market_skills)
    coverage_analyzer = CoverageAnalyzer(matcher)
    trend_analyzer = TrendAnalyzer(snapshots)
    rec_engine = CurriculumRecommender()
    optimizer = CurriculumOptimizer()

    dir_code = direction["code"]
    out_dir = OUTPUT / dir_code
    os.makedirs(out_dir, exist_ok=True)

    all_gaps: Counter = Counter()
    discipline_reports: list[tuple[str, GapAnalysisResult]] = []

    # Build direction-level RPD sets for cross-discipline awareness
    direction_rpd_norm: set[str] = set()
    discipline_skill_map: dict[str, set[str]] = {}
    for dname, disc_data in disciplines.items():
        dskills: set[str] = set()
        for skills_list in disc_data["competencies"].values():
            for s in skills_list:
                n = normalize_skill(s)
                if n and len(n) >= 3:
                    dskills.add(n)
                    direction_rpd_norm.add(n)
        discipline_skill_map[dname] = dskills

    for dname, disc_data in sorted(disciplines.items()):
        logger.info("analyzing_discipline", discipline=dname)
        disc_out = out_dir / _safe_filename(dname)
        os.makedirs(disc_out, exist_ok=True)

        cov_result = coverage_analyzer.analyze_discipline(
            disc_data["id"], dname, disc_data["competencies"],
            direction_rpd_norm=direction_rpd_norm,
            discipline_skill_map=discipline_skill_map,
        )
        if cov_result.is_err():
            logger.error("discipline_analysis_failed", discipline=dname, error=str(cov_result.err()))
            continue
        coverage = cov_result.unwrap()

        recs_result = rec_engine.generate(coverage)
        recs = recs_result.unwrap_or([])

        for g in coverage.gaps_list:
            all_gaps[g] += 1

        result = GapAnalysisResult(discipline=coverage, recommendations=recs)
        discipline_reports.append((dname, result))

        disc_data_out = {
            "direction": dir_code,
            "direction_name": direction["name"],
            "discipline": dname,
            "metrics": {
                "total_rpd_skills": coverage.total_skills,
                "market_matched": coverage.market_matched,
                "gaps": coverage.gaps,
                "coverage_ratio": coverage.coverage_ratio,
                "coverage_level": coverage.coverage_level,
                "top_market_matched_skills": [
                    {"skill": m.skill_name, "frequency": m.frequency, "match_type": m.match_type}
                    for m in coverage.top_matched
                ],
                "gaps_in_curriculum": coverage.gaps_list,
                "emerging_market_skills_not_in_rpd": [
                    {"skill": e.skill_name, "frequency": e.frequency}
                    for e in coverage.emerging
                ],
                "truly_missing_market_skills": [
                    {"skill": e.skill_name, "frequency": e.frequency}
                    for e in coverage.truly_missing
                ],
                "covered_in_other_disciplines": [
                    {"skill": cr.skill_name, "frequency": cr.frequency, "discipline": cr.discipline}
                    for cr in coverage.cross_references
                ],
            },
            "competencies": [
                {"code": cc.code, "total_skills": cc.total_skills,
                 "matched_skills": cc.matched_skills, "coverage": cc.coverage}
                for cc in coverage.competencies
            ],
            "recommendations": [
                {"type": r.type, "priority": r.priority, "message": r.message}
                for r in recs
            ],
        }
        fname = _safe_filename(dname) + ".json"
        try:
            (disc_out / fname).write_text(
                json.dumps(disc_data_out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("discipline_file_write_failed", discipline=dname, error=str(exc))
            continue
        logger.info("discipline_done", discipline=dname,
                     coverage=f"{coverage.coverage_ratio * 100:.1f}%",
                     gaps=coverage.gaps, emerging=len(coverage.emerging))

    if not discipline_reports:
        logger.error("no_disciplines_analyzed")
        return Err(AnalysisRunnerError(
            stage="analysis",
            message="No disciplines were successfully analyzed",
        ))

    avg_cov = round(
        sum(r.discipline.coverage_ratio for _, r in discipline_reports) / len(discipline_reports), 4
    ) if discipline_reports else 0

    # Direction-level emerging: skills not found in ANY discipline
    direction_emerging_result = matcher.get_emerging(direction_rpd_norm, top_n=15)
    direction_emerging: list[dict] = []
    if direction_emerging_result.is_ok():
        direction_emerging = [
            {"skill": s, "frequency": f}
            for s, f, _ in direction_emerging_result.unwrap()
        ]

    summary = DirectionSummary(
        direction_code=dir_code,
        direction_name=direction["name"],
        profile=direction["profile"] or "",
        total_disciplines=len(disciplines),
        average_coverage=avg_cov,
        total_gaps=len(all_gaps),
        top_cross_discipline_gaps=[
            {"skill": s, "disciplines": c} for s, c in all_gaps.most_common(15)
        ],
        top_emerging=direction_emerging,
        disciplines=[
            {"name": dn, "coverage_ratio": r.discipline.coverage_ratio,
             "coverage_level": r.discipline.coverage_level,
             "gaps": r.discipline.gaps, "emerging": len(r.discipline.emerging)}
            for dn, r in discipline_reports
        ],
    )

    summary_recs_result = rec_engine.generate_summary_recommendations(
        [r.discipline for _, r in discipline_reports], avg_cov, len(all_gaps), direction_emerging,
    )
    summary_recs = summary_recs_result.unwrap_or([])

    optimizer_result = optimizer.optimize(summary)
    optimizer_recs = optimizer_result.unwrap_or([])

    rising_result = trend_analyzer.get_rising(10)
    declining_result = trend_analyzer.get_declining(10)

    if rising_result.is_err():
        logger.warning("rising_trends_unavailable", reason=str(rising_result.err()))
    if declining_result.is_err():
        logger.warning("declining_trends_unavailable", reason=str(declining_result.err()))

    dir_summary = {
        "direction": dir_code,
        "direction_name": direction["name"],
        "profile": direction["profile"],
        "total_disciplines": summary.total_disciplines,
        "average_coverage": summary.average_coverage,
        "coverage_level": "high" if avg_cov >= 0.5 else "medium" if avg_cov >= 0.2 else "low",
        "total_gaps_across_all": summary.total_gaps,
        "top_cross_discipline_gaps": summary.top_cross_discipline_gaps,
        "top_emerging_across_all": summary.top_emerging,
        "recommendations": [
            {"type": r.type, "priority": r.priority, "message": r.message}
            for r in summary_recs + optimizer_recs
        ],
        "trends": {
            "rising": rising_result.unwrap_or([]),
            "declining": declining_result.unwrap_or([]),
        },
        "disciplines": summary.disciplines,
        "generated_at": datetime.now().isoformat(),
    }

    try:
        (out_dir / "_summary.json").write_text(
            json.dumps(dir_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.error("summary_write_failed", error=str(exc))
        return Err(AnalysisRunnerError(stage="write_summary", message=str(exc)))

    # — charts —
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        chart_dir = out_dir / "_charts"
        os.makedirs(chart_dir, exist_ok=True)

        # 1) Discipline coverage bar chart
        fig, ax = plt.subplots(figsize=(max(8, len(summary.disciplines) * 0.6), 5))
        dnames = [d["name"][:30] for d in summary.disciplines]
        dcov = [d["coverage_ratio"] * 100 for d in summary.disciplines]
        colors = ["#2ecc71" if v >= 50 else "#f39c12" if v >= 20 else "#e74c3c" for v in dcov]
        bars = ax.barh(range(len(dnames)), dcov, color=colors, height=0.6)
        ax.set_yticks(range(len(dnames)))
        ax.set_yticklabels(dnames, fontsize=8)
        ax.set_xlabel("Покрытие рынка %")
        ax.set_title(f"Покрытие дисциплин — {direction['name']}", fontsize=13)
        ax.invert_yaxis()
        for bar, v in zip(bars, dcov):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, max(dcov) + 15 if dcov else 100)
        fig.tight_layout()
        fig.savefig(chart_dir / "discipline_coverage.png", dpi=150)
        plt.close(fig)
        logger.info("chart_discipline_coverage_saved")

        # 2) Top cross-discipline gaps
        if summary.top_cross_discipline_gaps:
            fig, ax = plt.subplots(figsize=(10, max(4, len(summary.top_cross_discipline_gaps) * 0.35)))
            gap_skills = [g["skill"][:40] for g in summary.top_cross_discipline_gaps]
            gap_counts = [g["disciplines"] for g in summary.top_cross_discipline_gaps]
            bars = ax.barh(range(len(gap_skills)), gap_counts, color="#e74c3c", height=0.6)
            ax.set_yticks(range(len(gap_skills)))
            ax.set_yticklabels(gap_skills, fontsize=8)
            ax.set_xlabel("Дисциплин с пробелом")
            ax.set_title("Сквозные пробелы (топ)", fontsize=13)
            ax.invert_yaxis()
            for bar, v in zip(bars, gap_counts):
                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                        str(v), va="center", fontsize=8)
            fig.tight_layout()
            fig.savefig(chart_dir / "top_gaps.png", dpi=150)
            plt.close(fig)
            logger.info("chart_top_gaps_saved")

        # 3) Top emerging skills
        if summary.top_emerging:
            fig, ax = plt.subplots(figsize=(10, max(4, len(summary.top_emerging) * 0.35)))
            em_skills = [e["skill"][:40] for e in summary.top_emerging]
            em_freq = [e["frequency"] for e in summary.top_emerging]
            bars = ax.barh(range(len(em_skills)), em_freq, color="#3498db", height=0.6)
            ax.set_yticks(range(len(em_skills)))
            ax.set_yticklabels(em_skills, fontsize=8)
            ax.set_xlabel("Частота на рынке")
            ax.set_title("Восходящие навыки (топ)", fontsize=13)
            ax.invert_yaxis()
            for bar, v in zip(bars, em_freq):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        str(v), va="center", fontsize=8)
            fig.tight_layout()
            fig.savefig(chart_dir / "top_emerging.png", dpi=150)
            plt.close(fig)
            logger.info("chart_top_emerging_saved")

        # 4) Per-discipline competency coverage
        for dn, r in discipline_reports:
            comps = r.discipline.competencies
            if not comps:
                continue
            fname = _safe_filename(dn) + "_competencies.png"
            cnames = [c.code for c in comps]
            ccov = [c.coverage * 100 for c in comps]
            fig, ax = plt.subplots(figsize=(max(6, len(comps) * 0.5), 4))
            ccolors = ["#2ecc71" if v >= 50 else "#f39c12" if v >= 20 else "#e74c3c" for v in ccov]
            ax.bar(range(len(cnames)), ccov, color=ccolors, width=0.6)
            ax.set_ylabel("Покрытие %")
            ax.set_title(f"Компетенции — {dn[:40]}", fontsize=11)
            ax.set_xticks(range(len(cnames)))
            ax.set_xticklabels(cnames, rotation=45, ha="right", fontsize=8)
            for bar, v in zip(bars, ccov):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", fontsize=7)
            fig.tight_layout()
            fig.savefig(chart_dir / fname, dpi=150)
            plt.close(fig)

        logger.info("teacher_charts_generated", chart_dir=str(chart_dir))
    except Exception as exc:
        logger.warning("teacher_chart_failed", error=str(exc))

    logger.info("analysis_completed",
                 direction=dir_code,
                 disciplines=len(discipline_reports),
                 avg_coverage=avg_cov)
    return Ok(dir_summary)
