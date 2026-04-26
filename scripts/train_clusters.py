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

# Добавляем корень проекта в PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.analyzers.vacancy_clustering import VacancyClusterer
from src.parsing.vacancy_parser import VacancyParser
from src.parsing.utils import read_json
from src import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def prepare_vacancies_for_clustering(raw_vacancies: list) -> list:
    """
    Извлекает навыки из сырых вакансий и приводит к формату,
    ожидаемому VacancyClusterer: [{'skills': [...], 'experience': str}, ...]
    """
    parser = VacancyParser()
    prepared = []

    for vac in raw_vacancies:
        # 1. Навыки из key_skills (список словарей -> список строк)
        skills = []
        key_skills = vac.get('key_skills', [])
        if key_skills:
            skills = [s['name'] for s in key_skills if isinstance(s, dict) and 'name' in s]

        # 2. Текстовые навыки из описания и сниппета
        desc = vac.get('description', '') or ''
        snippet = vac.get('snippet', {}) or {}
        req = snippet.get('requirement', '') or ''
        resp = snippet.get('responsibility', '') or ''
        text_skills = parser.extract_skills_from_description(f"{desc} {req} {resp}")

        # 3. Объединяем и нормализуем (убираем дубли, синонимы)
        from src.parsing.skill_normalizer import SkillNormalizer
        all_skills = SkillNormalizer.deduplicate(skills + text_skills)

        # 4. Определяем уровень опыта (можно будет заменить на Vacancy модель)
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

        # Эвристика по названию
        name = vac.get('name', '').lower()
        if 'junior' in name or 'младший' in name or 'стажер' in name:
            experience = 'junior'
        elif 'senior' in name or 'старший' in name or 'ведущий' in name:
            experience = 'senior'

        prepared.append({
            'skills': all_skills,
            'experience': experience,
            'id': vac.get('id', '')
        })

    return prepared


def train_clusters(level: str = "all", save_report: bool = True, interpret: bool = True):
    """Основная функция обучения кластеров"""
    
    print("\n" + "="*80)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ КЛАСТЕРОВ ВАКАНСИЙ")
    print("="*80 + "\n")

    # Загрузка сырых вакансий
    vacancies_path = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
    
    if not vacancies_path.exists():
        logger.error(f"❌ Файл вакансий не найден: {vacancies_path}")
        print(f"   Путь: {vacancies_path}")
        print("   Проверьте, что данные лежат в data/raw/hh_vacancies_basic.json")
        return False

    raw_vacancies = read_json(vacancies_path)
    if raw_vacancies is None:
        logger.error(f"❌ Не удалось загрузить вакансии из {vacancies_path}")
        return False

    logger.info(f"Загружено сырых вакансий: {len(raw_vacancies):,}")

    # Подготовка вакансий (извлечение навыков)
    print("🔍 Извлечение навыков из вакансий...")
    all_vacancies = prepare_vacancies_for_clustering(raw_vacancies)
    logger.info(f"Подготовлено вакансий с навыками: {len(all_vacancies)}")

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

        # Фильтрация по уровню
        level_vacancies = [v for v in all_vacancies if v.get('experience') == lvl]
        n = len(level_vacancies)
        print(f"   Вакансий найдено: {n:,}")

        # Статистика по навыкам
        vacs_with_skills = sum(1 for v in level_vacancies if len(v.get('skills', [])) > 0)
        total_skills = sum(len(v['skills']) for v in level_vacancies)
        avg_skills = total_skills / n if n > 0 else 0
        print(f"   Вакансий с навыками: {vacs_with_skills} из {n}")
        print(f"   Среднее навыков на вакансию: {avg_skills:.1f}")
        if avg_skills < 2:
            print(f"   ⚠️  Очень мало навыков, кластеризация будет затруднена.")

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
            clusterer.min_cluster_size = 3   # ещё меньше для junior
            print("   → Используется KMeans (junior — мало данных)")
        elif lvl == "middle":
            clusterer.min_cluster_size = 8
            clusterer.use_hdbscan_fallback = True
            clusterer.max_clusters = 25
            print("   → Используется HDBSCAN (cosine)")
        else:  # senior
            clusterer.min_cluster_size = 6
            clusterer.use_hdbscan_fallback = True
            clusterer.max_clusters = 20
            print("   → Используется HDBSCAN (cosine)")

        # Обучение
        try:
            clusterer.fit(level_vacancies, level=lvl)
            
            clusters_count = getattr(clusterer, 'n_clusters_', 0)
            print(f"   ✅ Успешно создано кластеров: {clusters_count}")

            # Интерпретация кластеров (показываем топ-навыки)
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

    # Сохранение отчёта
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
    parser = argparse.ArgumentParser(
        description="Обучение кластеров вакансий с адаптивными параметрами"
    )
    parser.add_argument(
        "--level", 
        type=str, 
        default="all",
        choices=["all", "junior", "middle", "senior"],
        help="Какой уровень обучать (по умолчанию all)"
    )
    parser.add_argument(
        "--no-report", 
        action="store_true",
        help="Не сохранять JSON-отчёт"
    )
    parser.add_argument(
        "--no-interpret", 
        action="store_true",
        help="Отключить интерпретацию кластеров (топ-навыки)"
    )

    args = parser.parse_args()

    success = train_clusters(
        level=args.level,
        save_report=not args.no_report,
        interpret=not args.no_interpret
    )

    sys.exit(0 if success else 1)