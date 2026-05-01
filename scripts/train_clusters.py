"""
train_clusters.py — Улучшенный скрипт обучения кластеров вакансий
Поддержка аргументов, красивый вывод, сохранение отчёта.
"""

import argparse
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.analyzers.vacancy_clustering import VacancyClusterer
from src.parsing.vacancy_parser import VacancyParser
from src.parsing.utils import read_json
from src import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def prepare_vacancies_for_clustering(raw_vacancies: list) -> list:
    parser = VacancyParser()
    prepared = []

    for vac in raw_vacancies:
        skills = []
        key_skills = vac.get('key_skills', [])
        if key_skills:
            skills = [s['name'] for s in key_skills if isinstance(s, dict) and 'name' in s]

        desc = vac.get('description', '') or ''
        snippet = vac.get('snippet', {}) or {}
        req = snippet.get('requirement', '') or ''
        resp = snippet.get('responsibility', '') or ''
        text_skills = parser.extract_skills_from_description(f"{desc} {req} {resp}")

        from src.parsing.skill_normalizer import SkillNormalizer
        all_skills = SkillNormalizer.deduplicate(skills + text_skills)

        exp_obj = vac.get('experience', {})
        if isinstance(exp_obj, dict):
            exp_id = exp_obj.get('id', '').lower()
            if 'less1' in exp_id or 'junior' in exp_id or 'no_experience' in exp_id:
                experience = 'junior'
            elif 'between1and3' in exp_id or 'between3and6' in exp_id:
                experience = 'middle'
            elif 'between6and10' in exp_id or 'morethan10' in exp_id:
                experience = 'senior'
            else:
                experience = 'middle'
        elif isinstance(exp_obj, str):
            experience = exp_obj.lower()
        else:
            experience = 'middle'

        name = vac.get('name', '').lower()
        if 'junior' in name or 'младший' in name or 'стажер' in name:
            experience = 'junior'
        elif 'senior' in name or 'старший' in name or 'ведущий' in name:
            experience = 'senior'

        prepared.append({
            'skills': all_skills,
            'experience': experience,
            'id': vac.get('id', ''),
            'name': vac.get('name', ''),          # <-- добавили
            'key_skills': key_skills              # <-- можно убрать в проде, но для отладки полезно
        })

    return prepared


def train_clusters(level: str = "all", save_report: bool = True, interpret: bool = True):
    print("\n" + "="*80)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ КЛАСТЕРОВ ВАКАНСИЙ")
    print("="*80 + "\n")

    # Приоритет: детальный файл, затем базовый
    detailed_file = config.DATA_RESULT_DIR / "hh_vacancies_detailed.json"
    basic_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"

    if detailed_file.exists():
        vacancies_path = detailed_file
        logger.info(f"Используем детальные вакансии из {vacancies_path}")
    elif basic_file.exists():
        vacancies_path = basic_file
        logger.warning("Детальный файл не найден, используем базовый (навыков будет мало)")
    else:
        logger.error("Нет файлов вакансий")
        return False

    raw_vacancies = read_json(vacancies_path)
    if raw_vacancies is None:
        logger.error(f"Не удалось загрузить вакансии из {vacancies_path}")
        return False

    logger.info(f"Загружено сырых вакансий: {len(raw_vacancies):,}")

    print("🔍 Извлечение навыков из вакансий...")
    all_vacancies = prepare_vacancies_for_clustering(raw_vacancies)
    logger.info(f"Подготовлено вакансий: {len(all_vacancies)}")

    # Диагностика пустых навыков
    empty = [v for v in all_vacancies if not v['skills']]
    if empty:
        logger.warning(f"Обнаружено {len(empty)} вакансий без навыков")
        logger.info("Примеры пустых вакансий (первые 3):")
        for v in empty[:3]:
            logger.info(f"Вакансия ID={v['id']}, Название='{v['name']}', key_skills={v.get('key_skills', [])}")

    clusterer = VacancyClusterer(min_cluster_size=5)

    levels_to_train = ['junior', 'middle', 'senior'] if level == "all" else [level.lower()]

    report = {
        "trained_at": datetime.now().isoformat(),
        "total_vacancies": len(all_vacancies),
        "levels": {},
        "config": {
            "min_cluster_size": clusterer.min_cluster_size,
            "use_hdbscan": clusterer.use_hdbscan_fallback
        }
    }

    for lvl in levels_to_train:
        print(f"\n📍 Обучение уровня → {lvl.upper()}")
        print("-" * 60)

        level_vacancies = [v for v in all_vacancies if v.get('experience') == lvl]
        before_filter = len(level_vacancies)
        # Фильтруем пустые
        level_vacancies = [v for v in level_vacancies if v['skills']]
        after_filter = len(level_vacancies)
        if before_filter != after_filter:
            logger.info(f"Уровень {lvl}: отфильтровано {before_filter - after_filter} пустых вакансий")

        n = len(level_vacancies)
        print(f"   Вакансий после фильтрации: {n:,}")

        total_skills = sum(len(v['skills']) for v in level_vacancies)
        avg_skills = total_skills / n if n > 0 else 0
        print(f"   Среднее навыков на вакансию: {avg_skills:.1f}")

        if n < 30:
            print(f"   ⚠️  Слишком мало данных ({n}). Пропускаем уровень.")
            report["levels"][lvl] = {"status": "skipped", "reason": "too_few_samples", "count": n}
            continue

        # Адаптивные параметры
        if lvl == "junior":
            clusterer.n_clusters = 5
            clusterer.min_clusters = 2
            clusterer.max_clusters = 8
            clusterer.use_hdbscan_fallback = False
            clusterer.min_cluster_size = 3
            print("   → Используется KMeans (junior — мало данных)")
        elif lvl == "middle":
            clusterer.min_cluster_size = 8
            clusterer.use_hdbscan_fallback = True
            clusterer.max_clusters = 25
            print("   → Используется HDBSCAN (cosine)")
        else:
            clusterer.min_cluster_size = 6
            clusterer.use_hdbscan_fallback = True
            clusterer.max_clusters = 20
            print("   → Используется HDBSCAN (cosine)")

        try:
            clusterer.fit(level_vacancies, level=lvl)
            clusters_count = getattr(clusterer, 'n_clusters_', 0)
            print(f"   ✅ Успешно создано кластеров: {clusters_count}")

            if interpret and clusterer.is_fitted:
                print("   📊 Топ-навыки в каждом кластере:")
                for cid in range(clusters_count):
                    top_skills = clusterer.get_top_skills_in_cluster(cid, top_n=5)
                    print(f"      Кластер {cid:2d}: {', '.join(top_skills)}")

            report["levels"][lvl] = {
                "status": "success",
                "vacancies_count": n,
                "clusters_count": clusters_count,
                "clusterer_type": getattr(clusterer, 'clusterer_type', 'unknown'),
                "avg_skills_per_vacancy": round(avg_skills, 1)
            }
        except Exception as e:
            logger.error(f"Ошибка при обучении {lvl}: {e}")
            print(f"   ❌ Ошибка: {e}")
            report["levels"][lvl] = {"status": "failed", "error": str(e)}

    if save_report:
        report_path = config.DATA_PROCESSED_DIR / "cluster_training_report.json"
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Отчёт сохранён: {report_path}")

    print("\n" + "="*80)
    print("🎉 ОБУЧЕНИЕ КЛАСТЕРОВ ЗАВЕРШЕНО")
    print("="*80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение кластеров вакансий с адаптивными параметрами")
    parser.add_argument("--level", type=str, default="all", choices=["all", "junior", "middle", "senior"])
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-interpret", action="store_true")
    args = parser.parse_args()

    success = train_clusters(
        level=args.level,
        save_report=not args.no_report,
        interpret=not args.no_interpret
    )
    sys.exit(0 if success else 1)