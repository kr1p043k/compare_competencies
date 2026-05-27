"""Оркестрация построения всех графиков с учётом таксономии профессий."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import structlog

from src import config

from ._utils import load_ml_recommendations, load_skill_weights
from .clusters import plot_cluster_insights
from .correlation import plot_skill_correlation_heatmap
from .coverage import (
    plot_coverage_comparison,
    plot_domain_skill_gaps,
    plot_profession_coverage,
    plot_skills_heatmap,
)
from .importance import plot_ml_importance, plot_weight_distribution
from .radar import plot_skill_comparison_radar

logger = structlog.get_logger(__name__)


def save_all_charts(
    results: dict[str, Any],
    output_dir: Path,
    use_ml: bool = True,
    vacancies_skills_list: list[list[str]] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("generating_all_charts", output_dir=str(output_dir), profiles=len(results))

    logger.info("chart_coverage_comparison")
    plot_coverage_comparison(results, output_dir / "coverage_comparison.png")
    logger.info("chart_profession_coverage")
    plot_profession_coverage(results, output_dir / "profession_coverage.png")
    logger.info("chart_domain_skill_gaps")
    plot_domain_skill_gaps(results, output_dir / "domain_skill_gaps.png")

    skill_weights = load_skill_weights()
    market_top = list(skill_weights.keys())[:15] if skill_weights else []

    logger.info("chart_per_profile_start", profiles=list(results.keys()))
    for profile_name, eval_dict in results.items():
        prof_dir = output_dir / profile_name
        prof_dir.mkdir(exist_ok=True)
        student_skills = eval_dict.get("student_skills", [])
        if market_top:
            plot_skill_comparison_radar(
                student_skills,
                market_top,
                profile_name.capitalize(),
                prof_dir / f"radar_{profile_name}.png",
            )
        if use_ml:
            plot_ml_importance(profile_name, save_path=prof_dir / f"ml_importance_{profile_name}.png")
        plot_weight_distribution(skill_weights, save_path=prof_dir / f"weights_{profile_name}.png")

        recs = load_ml_recommendations(profile_name)
        if recs:
            deficits = [(r[0], r[1]) for r in recs if r[1] > 50][:10]
            if deficits:
                skills = [d[0] for d in deficits]
                freqs = [d[1] for d in deficits]
                fig, ax = plt.subplots()
                ax.barh(skills, freqs, color="salmon")
                ax.set_title(f"Топ-10 дефицитов высокого спроса — {profile_name}")
                ax.set_xlabel("Важность (%)")
                ax.invert_yaxis()
                fig.savefig(prof_dir / f"deficits_{profile_name}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info("deficits_saved", profile=profile_name, count=len(deficits))

    logger.info("chart_per_profile_done")
    if vacancies_skills_list:
        try:
            logger.info("chart_correlation_heatmap")
            from src.analyzers.skills.skill_correlation import SkillCorrelationAnalyzer

            corr_analyzer = SkillCorrelationAnalyzer()
            corr_analyzer.fit(vacancies_skills_list)
            plot_skill_correlation_heatmap(
                corr_analyzer, top_n=25, save_path=output_dir / "skill_correlation_heatmap.png"
            )
        except Exception as e:
            logger.warning("correlation_heatmap_failed", error=str(e))

    logger.info("chart_skills_heatmap")
    plot_skills_heatmap(results, top_n=20, save_path=output_dir / "skills_heatmap.png")
    logger.info("chart_cluster_insights")
    plot_cluster_insights(results, output_dir)
    logger.info("all_charts_generated")


def run_notebook(notebook_name: str, output_dir: Path | None = None) -> bool:
    base_dir = Path(__file__).parent.parent.parent
    notebook_path = base_dir / "notebooks" / notebook_name
    if not notebook_path.exists():
        logger.error("notebook_not_found", path=str(notebook_path))
        return False

    if output_dir is None:
        output_path = notebook_path
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / notebook_name

    try:
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(notebook_path),
            "--output",
            str(output_path),
            "--ExecutePreprocessor.timeout=600",
        ]
        logger.info("running_notebook", name=notebook_name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("notebook_completed", name=notebook_name, output=str(output_path))
            return True
        else:
            logger.error("notebook_failed", name=notebook_name, error=result.stderr[:200])
            return False
    except Exception as e:
        logger.exception("notebook_exception", error=str(e))
        return False


def show_context_info() -> None:
    print("\n" + "=" * 80)
    print("КОНТЕКСТНАЯ ИНФОРМАЦИЯ О ПРОЕКТЕ")
    print("=" * 80)

    market_file = config.DATA_PROCESSED_DIR / "competency_frequency.json"
    if market_file.exists():
        try:
            with open(market_file, encoding="utf-8") as f:
                market_skills = json.load(f)
            logger.info("market_skills_summary", unique_skills=len(market_skills))
            print(f"Рыночные навыки (частота): {len(market_skills)} уникальных")
            top_skills = sorted(market_skills.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Топ-5 востребованных: {', '.join([s for s, _ in top_skills])}")
        except Exception as e:
            logger.warning("market_skills_load_failed", error=str(e))
            print(f"Не удалось загрузить рыночные навыки: {e}")
    else:
        logger.warning("market_skills_file_not_found")
        print("Файл с рыночными навыками не найден. Сначала соберите данные.")

    mapping_file = config.COMPETENCY_MAPPING_FILE
    if mapping_file.exists():
        try:
            with open(mapping_file, encoding="utf-8") as f:
                mapping = json.load(f)
            logger.info("competency_mapping_summary", competencies=len(mapping))
            print(f"Компетенций в маппинге: {len(mapping)}")
        except Exception as e:
            logger.warning("competency_mapping_load_failed", error=str(e))
            print(f"Не удалось загрузить маппинг компетенций: {e}")
    else:
        logger.warning("competency_mapping_file_not_found", path=str(mapping_file))
        print(f"Файл маппинга не найден: {mapping_file}")

    taxonomy_file = config.PROFESSION_TAXONOMY_PATH
    if taxonomy_file.exists():
        try:
            with open(taxonomy_file, encoding="utf-8") as f:
                tax = json.load(f)
            profs = tax.get("professions", {})
            targets = tax.get("profile_targets", {})
            print(f"Таксономия профессий: {len(profs)} профессий")
            for pname, cfg in targets.items():
                print(f"  - {pname} → {cfg.get('target_profession', '?')} ({cfg.get('target_domains', [])})")
        except Exception as e:
            print(f"Не удалось загрузить таксономию: {e}")
    else:
        print(f"Файл таксономии не найден: {taxonomy_file}")

    students_dir = config.STUDENTS_DIR
    students = list(students_dir.glob("*_competency.json"))
    logger.info("student_profiles_found", count=len(students))
    print(f"Профили студентов (JSON): {len(students)}")
    for student_file in students:
        try:
            with open(student_file, encoding="utf-8") as f:
                data = json.load(f)
                skills = data.get("навыки", [])
                print(f"  - {student_file.stem.replace('_competency', '')}: {len(skills)} компетенций")
        except Exception as e:
            print(f"  - Ошибка чтения {student_file.name}: {e}")

    results_dir = config.DATA_DIR / "result"
    reports_found = 0
    if results_dir.exists():
        for student_dir in results_dir.glob("*"):
            if student_dir.is_dir():
                report_file = student_dir / f"full_recommendations_{student_dir.name}.json"
                if report_file.exists():
                    reports_found += 1
    logger.info("reports_summary", ready_reports=reports_found)
    print(f"Готовые отчёты gap-анализа: {reports_found}")

    print("\nРЕКОМЕНДАЦИИ ПО ЗАПУСКУ:")
    if not market_file.exists():
        print("  - Соберите рыночные данные: python main.py --it-sector --excel")
    elif not mapping_file.exists():
        print("  - Создайте файл competency_mapping.json в data/processed/")
    elif not students:
        print("  - Загрузите профили студентов через student_loader.py или вручную в data/students/")
    elif reports_found == 0:
        print("  - Запустите gap-анализ: python main.py --run-gap-analysis")
    else:
        print("  - Все данные в порядке. Графики уже можно сгенерировать:")
        print('    python -c "from src.visualization import save_all_charts; ..."')
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    show_context_info()

    skill_weights = load_skill_weights()
    if not skill_weights:
        logger.error("no_skill_weights_for_demo")
        sys.exit(1)

    fig1 = plot_weight_distribution(skill_weights)
    plt.show()

    results_for_charts = {}
    summary_path = config.DATA_RESULT_DIR / "profiles_comparison_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        evaluations = data.get("evaluations", {})
        for profile_name, eval_dict in evaluations.items():
            results_for_charts[profile_name] = eval_dict

    if results_for_charts:
        fig2 = plot_coverage_comparison(results_for_charts)
        plt.show()

    recs = load_ml_recommendations("base")
    if recs:
        fig3 = plot_ml_importance("base", top_n=10)
        plt.show()
    else:
        print("\nML-рекомендации для 'base' не найдены.")
        print("   Обучите модель командой:")
        print("   python -m src.predictors.ltr_recommendation_engine --load-raw --train")

    run_nb = input("\nЗапустить ноутбуки анализа? (y/n): ").strip().lower()
    if run_nb == "y":
        run_notebook("01_hh_analysis.ipynb")
        run_notebook("02_competency_matching.ipynb")
