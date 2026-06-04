"""Vacancy clustering module: train and analyze clusters."""

import json
import sys
from datetime import datetime
from pathlib import Path

import structlog

from src import config
from src.result import Err, Ok, Result
from src.analyzers.clustering.vacancy_clustering import VacancyClusterer
from src.parsing.skills.vacancy_parser import VacancyParser
from src.parsing.utils import read_json

logger = structlog.get_logger(__name__)


def prepare_vacancies_for_clustering(raw_vacancies: list) -> list:
    parser = VacancyParser()
    prepared = []
    for vac in raw_vacancies:
        skills = parser.parse_skills(vac)
        if not skills:
            continue
        prepared.append({
            "name": vac.get("name", ""),
            "skills": skills,
            "experience": vac.get("experience", {}).get("id", "noExp"),
            "salary_from": vac.get("salary", {}).get("from") if vac.get("salary") else None,
            "salary_to": vac.get("salary", {}).get("to") if vac.get("salary") else None,
            "currency": vac.get("salary", {}).get("currency") if vac.get("salary") else None,
            "area": vac.get("area", {}).get("name") if vac.get("area") else None,
            "employer": vac.get("employer", {}).get("name") if vac.get("employer") else None,
            "published_at": vac.get("published_at", ""),
            "alternate_url": vac.get("alternate_url", ""),
            "snippet_requirement": vac.get("snippet", {}).get("requirement", "") if vac.get("snippet") else "",
            "snippet_responsibility": vac.get("snippet", {}).get("responsibility", "") if vac.get("snippet") else "",
        })
    return prepared


def train_clusters(level: str = "all", save_report: bool = True, interpret: bool = True) -> bool:
    print("\n" + "=" * 80)
    print("   ЗАПУСК ОБУЧЕНИЯ КЛАСТЕРОВ ВАКАНСИЙ")
    print("=" * 80 + "\n")

    detailed_file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
    vacancies_path = detailed_file if detailed_file.exists() else basic_file

    if not vacancies_path.exists():
        logger.error("vacancies_file_not_found", path=str(vacancies_path))
        print("   Файл с вакансиями не найден.")
        return False

    logger.info("loading_vacancies", path=str(vacancies_path))
    raw_vacancies = read_json(vacancies_path)
    if not raw_vacancies:
        logger.error("no_vacancies_loaded")
        return False

    if isinstance(raw_vacancies, dict) and "items" in raw_vacancies:
        raw_vacancies = raw_vacancies["items"]
    logger.info("vacancies_loaded", count=len(raw_vacancies))

    prepared = prepare_vacancies_for_clustering(raw_vacancies)
    logger.info("vacancies_prepared", count=len(prepared))

    clusterer = VacancyClusterer()
    result = clusterer.cluster_vacancies(prepared, level=level)
    if result.is_err():
        logger.error("clustering_failed", error=str(result))
        return False

    clustered = result.unwrap()
    logger.info("clustering_completed", clusters=len(clustered.get("clusters", [])))

    if save_report:
        report_dir = config.REPORTS_DIR
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"cluster_report_{level}_{datetime.now():%Y%m%d_%H%M%S}.json"
        report = {
            "level": level,
            "total_vacancies": len(prepared),
            "num_clusters": len(clustered.get("clusters", [])),
            "clusters": clustered.get("clusters", []),
            "generated_at": str(datetime.now()),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("report_saved", path=str(report_path))
        print(f"\n   Отчёт сохранён: {report_path}")

    if interpret and "clusters" in clustered:
        print("\n" + "-" * 60)
        print("   ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ")
        print("-" * 60)
        for i, cl in enumerate(clustered["clusters"], 1):
            name = cl.get("name", cl.get("label", f"Cluster {i}"))
            size = cl.get("size", 0)
            top = ", ".join(cl.get("top_skills", cl.get("keywords", []))[:5])
            print(f"  {i:2d}. {name:30s}  [{size:3d} vac]  {top}")

    print("\n" + "=" * 80)
    print("   ОБУЧЕНИЕ КЛАСТЕРОВ ЗАВЕРШЕНО")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train vacancy clusters")
    parser.add_argument("--level", type=str, default="all", choices=["all", "junior", "middle", "senior"])
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-interpret", action="store_true")
    args = parser.parse_args()
    success = train_clusters(level=args.level, save_report=not args.no_report, interpret=not args.no_interpret)
    sys.exit(0 if success else 1)
