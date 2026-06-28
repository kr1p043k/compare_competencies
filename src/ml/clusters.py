"""Vacancy clustering module: train and analyze clusters."""

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

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
        snippet = vac.get("snippet") or {}
        desc = " ".join(filter(None, [
            snippet.get("requirement"),
            snippet.get("responsibility"),
            vac.get("description"),
        ]))
        skills = list(dict.fromkeys(
            s["name"] for s in (vac.get("key_skills") or []) if isinstance(s, dict) and s.get("name")
        ))
        if desc:
            parsed = parser.extract_skills_from_description(desc).unwrap_or([])
            skills = list(dict.fromkeys(skills + parsed))
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
    result = read_json(vacancies_path)
    if result.is_err():
        logger.error("no_vacancies_loaded", error=str(result))
        return False
    raw_vacancies = result.unwrap()
    if not raw_vacancies:
        logger.error("empty_vacancies")
        return False

    if isinstance(raw_vacancies, dict) and "items" in raw_vacancies:
        raw_vacancies = raw_vacancies["items"]
    logger.info("vacancies_loaded", count=len(raw_vacancies))

    prepared = prepare_vacancies_for_clustering(raw_vacancies)
    logger.info("vacancies_prepared", count=len(prepared))

    clusterer = VacancyClusterer()
    clusterer.fit(prepared, level=level)
    if not clusterer.is_fitted:
        logger.error("clustering_failed")
        return False

    labels_list = list(clusterer.labels_) if hasattr(clusterer, "labels_") else []
    unique_labels = sorted(set(l for l in labels_list if l >= 0))
    logger.info("clustering_completed", clusters=len(unique_labels))

    _CLUSTER_STOP = frozenset({
        "для", "от", "по", "на", "c", "о", "об", "из", "без", "в", "и",
        "не", "а", "но", "за", "до", "при", "про", "как", "еще", "уже",
        "na", "ot", "po", "c", "пo", "oт", "na",
    })

    # load known skill taxonomy for clean cluster naming
    _known_skills: set[str] = set()
    try:
        it_path = config.REFERENCE_DIR / "it_skills.json"
        if it_path.exists():
            raw = json.loads(it_path.read_text(encoding="utf-8"))
            _known_skills = {s.strip().lower() for s in raw if s.strip()}
            logger.info("known_skills_loaded", count=len(_known_skills))
    except Exception:
        logger.warning("known_skills_not_loaded")

    # build cluster summaries with TF-IDF-like scoring
    global_skill_freq: Counter = Counter()
    cluster_skill_freqs: list[Counter] = []
    cluster_sizes: list[int] = []
    for cl_id in unique_labels:
        indices = [i for i, l in enumerate(labels_list) if l == cl_id]
        cluster_sizes.append(len(indices))
        cluster_vacs = [prepared[i] for i in indices]
        cf: Counter = Counter()
        for v in cluster_vacs:
            for sk in v.get("skills", []):
                cf[sk] += 1
                global_skill_freq[sk] += 1
        cluster_skill_freqs.append(cf)

    n_clusters = len(unique_labels)
    clusters_out = []
    for cl_id, cf, size in zip(unique_labels, cluster_skill_freqs, cluster_sizes):
        scored = []
        for sk, freq in cf.most_common(50):
            if len(sk) <= 2 or sk.lower() in _CLUSTER_STOP:
                continue
            if re.search(r'^[a-zа-я]{1,2}$', sk.strip()):
                continue
            # boost if skill is in known taxonomy
            taxonomy_boost = 2.0 if sk.lower() in _known_skills else 0.5
            n_with_skill = sum(1 for other_cf in cluster_skill_freqs if other_cf[sk] > 0)
            idf = math.log(max(n_clusters / max(n_with_skill, 1), 1.0))
            scored.append((sk, freq * idf * taxonomy_boost))
        scored.sort(key=lambda x: -x[1])
        top_skills = [s for s, _ in scored[:10]]
        clusters_out.append({
            "label": f"Cluster {cl_id}",
            "size": size,
            "top_skills": top_skills,
        })

    if save_report:
        report_dir = config.REPORTS_DIR
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"cluster_report_{level}_{datetime.now():%Y%m%d_%H%M%S}.json"
        report = {
            "level": level,
            "total_vacancies": len(prepared),
            "num_clusters": len(clusters_out),
            "clusters": clusters_out,
            "generated_at": str(datetime.now()),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("report_saved", path=str(report_path))
        print(f"\n   Отчёт сохранён: {report_path}")

    if interpret and clusters_out:
        print("\n" + "-" * 60)
        print("   ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ")
        print("-" * 60)
        for i, cl in enumerate(clusters_out, 1):
            name = cl.get("label", f"Cluster {i}")
            size = cl.get("size", 0)
            top = ", ".join(cl.get("top_skills", [])[:5])
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
