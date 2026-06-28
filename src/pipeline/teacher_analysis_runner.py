"""Main runner: loads DB data → analyzers → predictors → data/result/teacher/.

Enhanced with embedding-based gap analysis (semantic coverage + SHAP).
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import numpy as np
import structlog

from src import config
from src.result import Ok, Err, Result
from src.errors import AnalysisDataError, AnalysisRunnerError, RecommendationError
from src.db import create_pool, close_pool, get_pool
from src.models.teacher_analysis import DirectionSummary, GapAnalysisResult
from src.analyzers.skill_matcher import SkillMatcher, normalize as normalize_skill
from src.analyzers.coverage_analyzer import CoverageAnalyzer
from src.analyzers.trend_analyzer import SnapshotTrendAnalyzer
from src.predictors.curriculum_recommender import CurriculumRecommender
from src.predictors.curriculum_optimizer import CurriculumOptimizer

logger = structlog.get_logger(__name__)
OUTPUT = Path(__file__).resolve().parent.parent.parent / "data" / "result" / "teacher"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()[:80]


def _enhance_disciplines_with_gap_analysis(
    disciplines: dict[str, dict],
    market_skill_names: list[str],
    skill_weights_dict: dict[str, float],
    output_dir: Path,
    dir_summary: dict,
) -> dict:
    """Run embedding-based coverage + LTR/SHAP for each discipline.

    Falls back silently if LTR model / embeddings unavailable.
    Returns dir_summary with enhanced fields merged in.
    """
    logger.info("gap_enhance_start", disciplines=len(disciplines), market_skills=len(market_skill_names))
    if not market_skill_names:
        logger.warning("gap_enhance_skipped_no_market_skills")
        return dir_summary

    # — 1. Build EmbeddingComparator with market skills —
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from src.analyzers.comparison.embedding_comparator import EmbeddingComparator
        from src.analyzers.comparison.embedding_provider import EmbeddingProviderFactory

        comp = EmbeddingComparator(similarity_threshold=0.5)
        comp.build_market_index(market_skill_names, level="middle")
        logger.info("market_embedding_index_built", skills=len(market_skill_names))
    except Exception as exc:
        logger.warning("gap_enhance_skip_embedding", error=str(exc))
        return dir_summary

    # — 2. Load LTR model for SHAP —
    ltr_engine = None
    ltr_model_path = MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
    if ltr_model_path.exists():
        try:
            from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine

            ltr_engine = LTRRecommendationEngine()
            match ltr_engine.load_model(ltr_model_path):
                case Ok(_):
                    logger.info("ltr_model_loaded_for_shap")
                case Err(e):
                    logger.warning("ltr_model_load_skipped", error=str(e))
                    ltr_engine = None
        except Exception as exc:
            logger.warning("ltr_model_load_failed", error=str(exc))
            ltr_engine = None

    # — 3. Per-discipline enhancement —
    enhanced_total = 0
    for dname, disc_data in disciplines.items():
        disc_skills: list[str] = []
        for skills_list in disc_data["competencies"].values():
            disc_skills.extend(skills_list)
        disc_skills = list(set(s.lower().strip() for s in disc_skills if s and len(s.strip()) >= 2))
        if not disc_skills:
            continue

        try:
            # — embedding coverage —
            match comp.compare_student_to_market(disc_skills):
                case Ok(embed_result):
                    embed_coverage = {
                        "semantic_coverage": embed_result.get("weighted_coverage", 0),
                        "avg_semantic_similarity": embed_result.get("avg_similarity", 0),
                        "top_market_matches": embed_result.get("matches", [])[:10],
                    }
                case _:
                    embed_coverage = {}
        except Exception as exc:
            logger.warning("embed_coverage_failed", discipline=dname, error=str(exc))
            embed_coverage = {}

        # — LTR scores + SHAP —
        ltr_shap = {}
        if ltr_engine is not None and ltr_engine.is_fitted:
            try:
                market_all = list(ltr_engine.skill_metadata.keys())
                known_lower = {s.lower().strip() for s in disc_skills}
                missing = [s for s in market_all if s.lower().strip() not in known_lower]

                match ltr_engine.predict_skill_impact_with_shap(
                    disc_skills, missing, compute_shap=True
                ):
                    case Ok((impacts, shap_vals, X)):
                        shap_per_skill = {}
                        if shap_vals is not None and X is not None and len(impacts) > 0:
                            for i, (skill, score, _) in enumerate(impacts):
                                if i < len(shap_vals):
                                    row_shap = {}
                                    for j, feat in enumerate(X.columns):
                                        if j < shap_vals.shape[1]:
                                            row_shap[feat] = round(float(shap_vals[i, j]), 4)
                                    shap_per_skill[skill] = row_shap

                        ltr_shap = {
                            "ltr_scores": [{"skill": s, "score": sc} for s, sc, _ in impacts[:10]],
                            "shap_values": shap_per_skill,
                        }
                    case Err(e):
                        logger.warning("ltr_shap_skipped", discipline=dname, error=str(e))
            except Exception as exc:
                logger.warning("ltr_shap_error", discipline=dname, error=str(exc))

        # — 4. Merge enhanced data into existing JSON —
        disc_dir = output_dir / _safe_filename(dname)
        json_file = disc_dir / (_safe_filename(dname) + ".json")
        if json_file.exists():
            try:
                existing = json.loads(json_file.read_text(encoding="utf-8"))
                existing["enhanced"] = {
                    "semantic_coverage": embed_coverage,
                    "ltr_and_shap": ltr_shap,
                }
                json_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
                enhanced_total += 1
            except Exception as exc:
                logger.warning("enhanced_merge_failed", discipline=dname, error=str(exc))

    logger.info("gap_enhance_done", enhanced=enhanced_total, total=len(disciplines))

    # — 5. Update direction summary with enhanced data —
    if dir_summary and "disciplines" in dir_summary:
        for d in dir_summary["disciplines"]:
            disc_dir = output_dir / _safe_filename(d["name"])
            json_file = disc_dir / (_safe_filename(d["name"]) + ".json")
            if json_file.exists():
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    enh = data.get("enhanced", {})
                    sc = enh.get("semantic_coverage", {})
                    d["semantic_coverage"] = sc.get("semantic_coverage", d.get("coverage_ratio", 0))
                    d["avg_semantic_similarity"] = sc.get("avg_semantic_similarity", 0)
                    ls = enh.get("ltr_and_shap", {})
                    d["shap_count"] = len(ls.get("shap_values", {}))
                except Exception:
                    pass
        dir_summary["has_enhanced_gap"] = True

    return dir_summary


async def run_teacher_analysis(
    direction_code: str | None = None,
    discipline_filter: str | None = None,
    user_id: str | None = None,
) -> Result[dict, AnalysisRunnerError]:
    logger.info("analysis_started", direction=direction_code, discipline=discipline_filter)

    # — DB connection —
    await create_pool()
    pool = get_pool()
    if not pool:
        logger.error("no_db_pool")
        return Err(AnalysisRunnerError(stage="db", message="Failed to create database pool"))

    # — create pipeline run —
    from src.pipeline.db_writer import create_pipeline_run, complete_pipeline_run, save_to_analysis_results

    run_id = await create_pipeline_run("teacher-analysis")
    await pool.execute(
        "UPDATE pipeline_runs SET user_id=$1 WHERE id=$2",
        user_id, run_id,
    )

    # — load market skills from vacancies (real frequencies) + it_skills taxonomy —
    market_skills: dict[str, int] = {}
    vac_count = 0

    # Кэш: market_skills пересчитываются только при изменении вакансий
    from src.cache_manager import CacheManager
    _cache_dir = config.DATA_CACHE_DIR
    _cache_mgr = CacheManager(_cache_dir)
    _vac_hash = await pool.fetchval(
        "SELECT MD5(COALESCE(MAX(created_at)::text, '0')) FROM vacancies "
        "WHERE key_skills IS NOT NULL AND jsonb_array_length(key_skills) > 0"
    )
    _cache_key = "teacher_market_skills"
    match _cache_mgr.load(_cache_key):
        case Ok(cached):
            if isinstance(cached, dict) and cached.get("hash") == _vac_hash:
                market_skills = cached["skills"]
                vac_count = cached.get("count", 0)
                logger.info("market_skills_loaded_from_cache", count=len(market_skills))

    if not market_skills:
        try:
            vrows = await pool.fetch(
                """SELECT LOWER(TRIM(value)) AS skill, COUNT(*) AS frequency
                   FROM vacancies, jsonb_array_elements_text(key_skills) AS value
                   WHERE key_skills IS NOT NULL AND jsonb_array_length(key_skills) > 0
                   GROUP BY LOWER(TRIM(value))
                   ORDER BY frequency DESC"""
            )
            for r in vrows:
                market_skills[r["skill"]] = r["frequency"]
            vac_count = len(market_skills)
            logger.info("market_skills_from_vacancies", count=vac_count)
        except Exception as exc:
            logger.warning("market_skills_vacancies_failed", error=str(exc))

        it_skills_path = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "it_skills.json"
        if it_skills_path.exists():
            import json
            with open(it_skills_path, "r", encoding="utf-8") as f:
                it_data = json.load(f)
            for name in it_data:
                k = name.strip().lower()
                if k and k not in market_skills:
                    market_skills[k] = 1

        _cache_mgr.save(_cache_key, {"hash": _vac_hash, "skills": market_skills, "count": vac_count})

    if not market_skills:
        logger.warning("no_market_skills_found")
    logger.info("market_skills_loaded", count=len(market_skills), from_vacancies=vac_count)

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
                 LEFT JOIN skills s ON s.id = cs.skill_id AND s.source != 'market'
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
        sf = d.get("skill_freq")
        if isinstance(sf, dict):
            pass
        elif isinstance(sf, str):
            d["skill_freq"] = json.loads(sf)
        else:
            d["skill_freq"] = {}
        snapshots.append(d)

    logger.info("snapshots_loaded", count=len(snapshots))

    # — init services —
    from src.analyzers.comparison.embedding_provider import EmbeddingProviderFactory
    from src.analyzers.discipline_relevance import DisciplineAwareScorer
    _discipline_scorer = DisciplineAwareScorer()
    _discipline_scorer.load()
    matcher = SkillMatcher(market_skills, embedding_provider=EmbeddingProviderFactory.get())
    coverage_analyzer = CoverageAnalyzer(matcher, discipline_scorer=_discipline_scorer)
    trend_analyzer = SnapshotTrendAnalyzer(snapshots)
    rec_engine = CurriculumRecommender()
    optimizer = CurriculumOptimizer()

    profession_taxonomy: dict = {}
    try:
        prof_path = config.REFERENCE_DIR / "profession_taxonomy.json"
        if prof_path.exists():
            profession_taxonomy = json.loads(prof_path.read_text(encoding="utf-8")).get("professions", {})
            logger.info("profession_taxonomy_loaded", professions=len(profession_taxonomy))
    except Exception:
        pass

    dir_code = direction["code"]
    out_dir = OUTPUT / dir_code
    os.makedirs(out_dir, exist_ok=True)

    # Skip если данные не изменились с последнего полного pipeline
    summary_path = out_dir / "_summary.json"
    if summary_path.exists():
        try:
            last_run = await pool.fetchval(
                "SELECT MAX(completed_at) FROM pipeline_runs "
                "WHERE status = 'completed' AND action IN ('full-cycle', 'hh-import')"
            )
            summary_mtime = summary_path.stat().st_mtime
            if last_run and summary_mtime > last_run.timestamp():
                logger.info("skipping_analysis_data_unchanged", direction=dir_code)
                import json
                with open(summary_path, "r", encoding="utf-8") as _sf:
                    return Ok(json.load(_sf))
        except Exception:
            pass

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

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    _discipline_lock = threading.Lock()

    def _analyze_one(dname, disc_data):
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
            return None
        coverage = cov_result.unwrap()

        recs_result = rec_engine.generate(coverage)
        recs = recs_result.unwrap_or([])

        # Filter recommendations by discipline relevance
        filtered: list = []
        for r in recs:
            skill = r.skill_name or r.type
            relevance = _discipline_scorer.compute_relevance(skill, dname)
            if relevance.level == "UNRELATED":
                r.priority = "low"
                r.message += " (низкая релевантность дисциплине)"
            filtered.append(r)
        recs = filtered

        with _discipline_lock:
            for g in coverage.gaps_list:
                all_gaps[g] += 1

        result = GapAnalysisResult(discipline=coverage, recommendations=recs)

        disc_data_out = {
            "direction": dir_code,
            "direction_name": direction["name"],
            "discipline": dname,
            "metrics": {
                "total_rpd_skills": coverage.total_skills,
                "market_matched": coverage.market_matched,
                "gaps": coverage.gaps,
                "coverage_ratio": coverage.coverage_ratio,
                "weighted_coverage": coverage.weighted_coverage,
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
            return None
        logger.info("discipline_done", discipline=dname,
                     coverage=f"{coverage.coverage_ratio * 100:.1f}%",
                     gaps=coverage.gaps, emerging=len(coverage.emerging))
        return (dname, result)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_analyze_one, dname, disc_data): dname
            for dname, disc_data in sorted(disciplines.items())
        }
        for future in as_completed(futures):
            dname = futures[future]
            try:
                res = future.result()
                if res is not None:
                    discipline_reports.append(res)
            except Exception as e:
                logger.error("discipline_parallel_failed", discipline=dname, error=str(e))

    # — enhanced gap analysis (embedding + LTR/SHAP) —
    try:
        market_skill_names_for_enhance = list(market_skills.keys())
        skill_weights_path = DATA_DIR / "processed" / "skill_weights.json"
        sw: dict[str, float] = {}
        if skill_weights_path.exists():
            sw = json.loads(skill_weights_path.read_text(encoding="utf-8"))
        _enhance_disciplines_with_gap_analysis(
            disciplines=disciplines,
            market_skill_names=market_skill_names_for_enhance,
            skill_weights_dict=sw,
            output_dir=out_dir,
            dir_summary=None,
        )
    except Exception as exc:
        logger.warning("enhanced_gap_analysis_skipped", error=str(exc))

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
        "target_professions": list(profession_taxonomy.keys())[:10],
        "generated_at": datetime.now().isoformat(),
    }

    # — merge enhanced data from per-discipline JSONs into summary —
    for d in dir_summary.get("disciplines", []):
        d_file = out_dir / _safe_filename(d["name"]) / (_safe_filename(d["name"]) + ".json")
        if d_file.exists():
            try:
                disc_data = json.loads(d_file.read_text(encoding="utf-8"))
                enh = disc_data.get("enhanced", {})
                sc = enh.get("semantic_coverage", {})
                d["semantic_coverage"] = sc.get("semantic_coverage", d.get("coverage_ratio", 0))
                d["avg_semantic_similarity"] = sc.get("avg_semantic_similarity", 0)
                ls = enh.get("ltr_and_shap", {})
                d["shap_scores"] = ls.get("ltr_scores", [])
            except Exception:
                pass
    dir_summary["has_enhanced_gap"] = True

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
            cbars = ax.bar(range(len(cnames)), ccov, color=ccolors, width=0.6)
            ax.set_ylabel("Покрытие %")
            ax.set_title(f"Компетенции — {dn[:40]}", fontsize=11)
            ax.set_xticks(range(len(cnames)))
            ax.set_xticklabels(cnames, rotation=45, ha="right", fontsize=8)
            for cb, v in zip(cbars, ccov):
                ax.text(cb.get_x() + cb.get_width() / 2, cb.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", fontsize=7)
            fig.tight_layout()
            fig.savefig(chart_dir / fname, dpi=150)
            plt.close(fig)

        logger.info("teacher_charts_generated", chart_dir=str(chart_dir))
    except Exception as exc:
        logger.warning("teacher_chart_failed", error=str(exc))

    # — save to coverage_analyses —
    try:
        for _, result in discipline_reports:
            dc = result.discipline
            if dc and dc.discipline_id:
                dir_id = direction.get("id") if direction else None
                cov_ratio = round(dc.coverage_ratio, 4)
                await pool.execute(
                    """INSERT INTO coverage_analyses
                       (discipline_id, direction_id, total_skills,
                        market_matched_skills, coverage_ratio, analysis_date)
                       VALUES ($1, $2, $3, $4, $5, NOW())""",
                    dc.discipline_id, dir_id,
                    dc.total_skills, dc.market_matched, cov_ratio,
                )
        logger.info("coverage_analyses_saved", count=len(discipline_reports))
    except Exception as exc:
        logger.warning("coverage_analyses_save_failed", error=str(exc))

    # — save to pipeline_runs + analysis_results —
    try:
        params = {"direction": dir_code, "discipline_filter": discipline_filter, "disciplines": len(disciplines)}
        stats = {**params, "avg_coverage": avg_cov, "total_disciplines": len(discipline_reports)}
        await complete_pipeline_run(run_id, status="completed", stats=stats)

        from src.pipeline.db_writer import save_to_analysis_results
        await save_to_analysis_results(
            run_id=run_id,
            analysis_type="teacher",
            data=dir_summary,
        )
    except Exception as db_err:
        logger.warning("db_write_failed", error=str(db_err))

    logger.info("analysis_completed",
                 direction=dir_code,
                 disciplines=len(discipline_reports),
                 avg_coverage=avg_cov)
    return Ok(dir_summary)
