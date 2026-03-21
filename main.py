#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import time
import json
import re
from pathlib import Path
from collections import Counter

# Устанавливаем кодировку для вывода в консоль Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.parsing.hh_api import HeadHunterAPI
from src.parsing.vacancy_parser import VacancyParser
from src.parsing.utils import setup_logging
from src import config
from src.loaders_student.student_loader import generate_profiles_from_csv
from src.utils import load_competency_mapping

def parse_arguments():
    parser = argparse.ArgumentParser(description="Сбор и анализ вакансий с hh.ru по региону")
    parser.add_argument('--query', '-q', type=str, default="Data Scientist",
                        help="Поисковый запрос (например, 'Data Scientist')")
    parser.add_argument('--area-id', '-a', type=int, default=1,
                        help="ID региона (по умолчанию 1 - Москва)")
    parser.add_argument('--max-pages', '-p', type=int, default=20,
                        help="Максимальное количество страниц (по умолчанию 20)")
    parser.add_argument('--period', '-d', type=int, default=30,
                        help="Период поиска в днях (по умолчанию 30)")
    parser.add_argument('--show-vacancies', '-v', action='store_true',
                        help="Показать список найденных вакансий в консоли")
    parser.add_argument('--skip-details', '-s', action='store_true',
                        help="Пропустить загрузку деталей (только базовый поиск)")
    parser.add_argument('--excel', '-e', action='store_true',
                        help="Сохранить результаты в Excel")
    parser.add_argument('--no-filter', '-nf', action='store_true',
                        help="Отключить фильтрацию навыков по белому списку")
    return parser.parse_args()

def safe_print(text):
    """Безопасный вывод текста в консоль"""
    try:
        print(text)
    except UnicodeEncodeError:
        clean_text = re.sub(r'[^\x00-\x7F]+', '', text)
        print(clean_text)

def normalize_skill_for_matching(skill):
    """Нормализует навык для сопоставления с маппингом"""
    # Приводим к нижнему регистру и удаляем лишние пробелы
    normalized = skill.lower().strip()
    # Удаляем пунктуацию, кроме дефиса
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    # Заменяем множественные пробелы на один
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def main():
    setup_logging()
    logger = logging.getLogger("main")

    args = parse_arguments()
    logger.info("=" * 60)
    logger.info("СБОР ВАКАНСИЙ С HH.RU ПО РЕГИОНУ")
    logger.info("=" * 60)
    logger.info(f"Запрос: '{args.query}'")
    logger.info(f"Регион ID: {args.area_id}, период: {args.period} дней, макс. страниц: {args.max_pages}")
    logger.info(f"Загружать детали: {not args.skip_details}")
    logger.info(f"Фильтрация навыков: {'ВЫКЛЮЧЕНА' if args.no_filter else 'ВКЛЮЧЕНА'}")

    # ------------------------------------------------------------------
    # ЭТАП 0: ОБНОВЛЕНИЕ ПРОФИЛЕЙ УЧЕНИКОВ
    # ------------------------------------------------------------------
    try:
        logger.info("Проверка наличия CSV-матрицы компетенций...")
        csv_path = config.DATA_RAW_DIR / "competency_matrix.csv"
        if not csv_path.exists():
            csv_path = config.DATA_DIR / "last_uploaded" / "competency_matrix.csv"
        
        if csv_path.exists():
            profiles = generate_profiles_from_csv(csv_path)
            logger.info(f"✅ Профили учеников успешно обновлены из CSV: {len(profiles)} профилей")
        else:
            logger.warning("⚠️ CSV-файл не найден, используются существующие профили.")
    except Exception as e:
        logger.exception(f"❌ Ошибка при обновлении профилей: {e}")

    hh_api = HeadHunterAPI()
    parser = VacancyParser()

    # ------------------------------------------------------------------
    # ЭТАП 1: ПОИСК ВАКАНСИЙ
    # ------------------------------------------------------------------
    logger.info("ЭТАП 1: Поиск вакансий")

    search_fields = ['name', 'company_name', 'description']

    basic_vacancies = hh_api.search_vacancies(
        text=args.query,
        area=args.area_id,
        period_days=args.period,
        max_pages=args.max_pages,
        per_page=100,
        search_fields=search_fields
    )

    if not basic_vacancies:
        logger.error("Не получено ни одной вакансии. Завершение работы.")
        return

    logger.info(f"Найдено {len(basic_vacancies)} вакансий")

    # ------------------------------------------------------------------
    # ЭТАП 2: ПОЛУЧЕНИЕ ДЕТАЛЬНОЙ ИНФОРМАЦИИ
    # ------------------------------------------------------------------
    if args.skip_details:
        logger.info("Пропускаем загрузку деталей по запросу")
        vacancies_to_process = basic_vacancies
        parser.save_raw_vacancies(basic_vacancies, filename="hh_vacancies_basic.json")
    else:
        logger.info(f"ЭТАП 2: Загрузка детальной информации для {len(basic_vacancies)} вакансий")
        detailed_vacancies = []
        success_count = 0

        for idx, vac in enumerate(basic_vacancies, 1):
            vac_id = vac['id']
            if idx % 20 == 0:
                logger.info(f"Прогресс: {idx}/{len(basic_vacancies)} вакансий")

            details = hh_api.get_vacancy_details(vac_id)

            if details:
                detailed_vacancies.append(details)
                success_count += 1
                if details.get('key_skills'):
                    logger.debug(f"✅ Вакансия {vac_id} содержит {len(details['key_skills'])} ключевых навыков")
            else:
                logger.warning(f"⚠️ Не удалось загрузить детали для вакансии {vac_id}")

            time.sleep(config.REQUEST_DELAY)

        logger.info(f"Загружена детальная информация для {success_count}/{len(basic_vacancies)} вакансий")
        vacancies_to_process = detailed_vacancies
        parser.save_raw_vacancies(detailed_vacancies, filename="hh_vacancies_detailed.json")

    # ------------------------------------------------------------------
    # ЭТАП 3: ВЫВОД В КОНСОЛЬ (если запрошено)
    # ------------------------------------------------------------------
    if args.show_vacancies:
        parser.print_vacancies_list(vacancies_to_process)

    # ------------------------------------------------------------------
    # ЭТАП 4: ИЗВЛЕЧЕНИЕ И АНАЛИЗ НАВЫКОВ
    # ------------------------------------------------------------------
    logger.info("ЭТАП 4: Извлечение и анализ навыков")

    # Извлекаем навыки из key_skills
    all_skills = parser.extract_skills(vacancies_to_process)

    if not all_skills:
        logger.info("key_skills не найдены, пробуем извлечь из текста вакансий...")
        all_skills = parser.extract_skills_from_text(vacancies_to_process)

    if not all_skills:
        logger.warning("❌ Не удалось извлечь навыки ни из key_skills, ни из текста.")
        return

    logger.info(f"Извлечено {len(all_skills)} сырых навыков")
    
    # Подсчитываем частоты
    skill_frequencies = parser.count_skills(all_skills)
    logger.info(f"Уникальных навыков после нормализации: {len(skill_frequencies)}")
    
    # Сохраняем частоты навыков (с фильтрацией или без)
    parser.save_processed_frequencies(skill_frequencies, apply_filter=not args.no_filter)
    
    # Вывод топ-20 навыков
    top_skills = sorted(skill_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
    safe_print("\n" + "=" * 60)
    safe_print("ТОП-20 НАИБОЛЕЕ ВОСТРЕБОВАННЫХ НАВЫКОВ")
    safe_print("=" * 60)
    for i, (skill, count) in enumerate(top_skills, 1):
        safe_print(f"{i:2}. {skill:<50} {count:>4} упоминаний")

    # ------------------------------------------------------------------
    # ПРЕОБРАЗОВАНИЕ В УЧЕБНЫЕ КОМПЕТЕНЦИИ
    # ------------------------------------------------------------------
    logger.info("Преобразование рыночных навыков в учебные компетенции...")
    
    try:
        mapping = load_competency_mapping()
        if mapping:
            logger.info(f"Загружен маппинг для {len(mapping)} компетенций")
            
            # Создаем обратный индекс с нормализацией
            skill_to_comp = {}
            for comp, keywords in mapping.items():
                for keyword in keywords:
                    normalized_keyword = normalize_skill_for_matching(keyword)
                    skill_to_comp.setdefault(normalized_keyword, []).append(comp)
            
            logger.info(f"Создан обратный индекс для {len(skill_to_comp)} ключевых слов")

            comp_counter = Counter()
            matched_skills = 0
            unmatched_skills = []
            
            for skill, freq in skill_frequencies.items():
                normalized_skill = normalize_skill_for_matching(skill)
                
                if normalized_skill in skill_to_comp:
                    matched_skills += 1
                    for comp in skill_to_comp[normalized_skill]:
                        comp_counter[comp] += freq
                else:
                    # Пытаемся найти частичное совпадение
                    found = False
                    for keyword in skill_to_comp.keys():
                        if keyword in normalized_skill or normalized_skill in keyword:
                            matched_skills += 1
                            for comp in skill_to_comp[keyword]:
                                comp_counter[comp] += freq
                            found = True
                            break
                    if not found and len(unmatched_skills) < 10:
                        unmatched_skills.append(skill)

            logger.info(f"Сопоставлено навыков с компетенциями: {matched_skills} из {len(skill_frequencies)}")
            
            if unmatched_skills:
                logger.info(f"Примеры несопоставленных навыков: {unmatched_skills[:5]}")
            
            if comp_counter:
                # Сохраняем результат
                comp_freq_path = config.DATA_PROCESSED_DIR / "competency_frequency_mapped.json"
                comp_dict = dict(comp_counter.most_common())
                
                with open(comp_freq_path, 'w', encoding='utf-8') as f:
                    json.dump(comp_dict, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ Частоты компетенций сохранены в {comp_freq_path}")

                # Вывод топ-20 компетенций
                top_comps = comp_counter.most_common(20)
                safe_print("\n" + "=" * 60)
                safe_print("ТОП-20 УЧЕБНЫХ КОМПЕТЕНЦИЙ НА РЫНКЕ")
                safe_print("=" * 60)
                for i, (comp, freq) in enumerate(top_comps, 1):
                    safe_print(f"{i:2}. {comp:<25} {freq:>4} суммарных упоминаний")
                
                # Выводим статистику
                safe_print("\n" + "=" * 60)
                safe_print("СТАТИСТИКА СОПОСТАВЛЕНИЯ")
                safe_print("=" * 60)
                safe_print(f"Всего уникальных навыков: {len(skill_frequencies)}")
                safe_print(f"Сопоставлено с компетенциями: {matched_skills}")
                safe_print(f"Получено компетенций: {len(comp_counter)}")
                safe_print(f"Всего упоминаний компетенций: {sum(comp_counter.values())}")
            else:
                safe_print("\n⚠️ Нет совпадений между рыночными навыками и учебными компетенциями")
                safe_print("Проверьте файл competency_mapping.json")
        else:
            logger.warning("Маппинг не загружен, пропущено преобразование в компетенции.")
            safe_print("\n⚠️ Файл competency_mapping.json не найден или пуст")
    except Exception as e:
        logger.exception(f"Ошибка при преобразовании компетенций: {e}")
        safe_print(f"\n❌ Ошибка при преобразовании компетенций: {e}")

    # ------------------------------------------------------------------
    # ЭТАП 5: СОХРАНЕНИЕ В EXCEL
    # ------------------------------------------------------------------
    if args.excel:
        logger.info("ЭТАП 5: Сохранение результатов в Excel")
        df = parser.aggregate_to_dataframe(vacancies_to_process)
        if not df.empty:
            filename = f"vacancies_{args.query}_{args.area_id}.xlsx".replace(' ', '_')
            parser.save_to_excel(df, filename)
            logger.info(f"✅ Excel файл сохранен: {filename}")

    logger.info("=" * 60)
    logger.info("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()