import json
from collections import Counter
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from pathlib import Path
from src import config

logger = logging.getLogger(__name__)

class VacancyParser:
    """
    Класс для обработки вакансий и извлечения навыков.
    """

    # ===== СОХРАНЕНИЕ ДАННЫХ =====
    def save_raw_vacancies(self, vacancies: List[Dict[str, Any]], filename: str = "hh_vacancies.json"):
        """Сохраняет сырые данные о вакансиях в JSON-файл."""
        filepath = config.DATA_RAW_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=2)
        logger.info(f"Сырые данные сохранены в {filepath} (вакансий: {len(vacancies)})")

    def save_processed_frequencies(self, frequencies: Dict[str, int], filename: str = "competency_frequency.json"):
        """Сохраняет обработанные частоты навыков в JSON-файл."""
        filepath = config.DATA_PROCESSED_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(frequencies, f, ensure_ascii=False, indent=2)
        logger.info(f"Частоты навыков сохранены в {filepath} (навыков: {len(frequencies)})")

    def save_area_vacancies(self, data: Dict[str, Any], area_id: str, area_name: str, page: int):
        """Сохраняет JSON-ответ для конкретной страницы региона (для перебора по регионам)."""
        safe_area_name = "".join(c for c in area_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{area_id}_{safe_area_name}_{page}.json"
        filepath = config.DATA_AREAS_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Сохранены данные региона {area_name} (страница {page})")

    # ===== ИЗВЛЕЧЕНИЕ НАВЫКОВ =====
    @staticmethod
    def extract_skills(vacancies: List[Dict[str, Any]]) -> List[str]:
        """Извлекает все ключевые навыки из списка вакансий (поле key_skills)."""
        all_skills = []
        vacancies_with_skills = 0
        for vacancy in vacancies:
            skills_data = vacancy.get('key_skills', [])
            if skills_data:
                vacancies_with_skills += 1
                for skill in skills_data:
                    skill_name = skill.get('name', '')
                    if skill_name:
                        all_skills.append(skill_name)
        logger.info(f"Из {len(vacancies)} вакансий извлечены навыки из {vacancies_with_skills} (остальные без key_skills)")
        return all_skills

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """Приводит навык к единому виду (нижний регистр, обрезка пробелов)."""
        return skill.lower().strip()

    @staticmethod
    def count_skills(skills_list: List[str]) -> Dict[str, int]:
        """Подсчитывает частоту каждого навыка после нормализации."""
        normalized_skills = [VacancyParser.normalize_skill(s) for s in skills_list]
        skill_counts = Counter(normalized_skills)
        logger.info(f"Подсчитаны частоты навыков: {len(skill_counts)} уникальных")
        return dict(skill_counts)

    # ===== ВЫВОД В КОНСОЛЬ (CLI) =====
    @staticmethod
    def print_vacancies_list(vacancies: List[Dict[str, Any]], limit: int = 30):
        """
        Выводит список вакансий в консоль в виде таблицы.
        """
        if not vacancies:
            print("❌ Список вакансий пуст.")
            return

        print("\n📋 Найденные вакансии:")
        print("=" * 140)
        print(f"{'№':<4} {'ID':<10} {'Название':<50} {'Компания':<35} {'Зарплата':<25} {'Навыки':<8}")
        print("-" * 140)

        for idx, item in enumerate(vacancies[:limit], 1):
            vacancy_id = item.get('id', 'N/A')

            name = item.get('name', 'Без названия')
            if len(name) > 47:
                name = name[:44] + "..."

            employer = item.get('employer', {})
            employer_name = employer.get('name', 'N/A')
            if len(employer_name) > 32:
                employer_name = employer_name[:29] + "..."

            salary = item.get('salary')
            if salary and salary.get('from') and salary.get('to'):
                salary_str = f"{salary['from']}-{salary['to']} {salary.get('currency', 'RUR')}"
            elif salary and salary.get('from'):
                salary_str = f"от {salary['from']} {salary.get('currency', 'RUR')}"
            elif salary and salary.get('to'):
                salary_str = f"до {salary['to']} {salary.get('currency', 'RUR')}"
            else:
                salary_str = "Не указана"
            if len(salary_str) > 24:
                salary_str = salary_str[:21] + "..."

            skills_count = len(item.get('key_skills', []))

            print(f"{idx:<4} {vacancy_id:<10} {name:<50} {employer_name:<35} {salary_str:<25} {skills_count:<8}")

        if len(vacancies) > limit:
            print(f"... и ещё {len(vacancies) - limit} вакансий")
        print("=" * 140)

    # ===== АГРЕГАЦИЯ В EXCEL =====
    def aggregate_area_results(self) -> pd.DataFrame:
        """Собирает все JSON из data/areas/ в один DataFrame."""
        all_vacancies_data = []
        area_files = list(config.DATA_AREAS_DIR.glob("*.json"))

        if not area_files:
            logger.warning("Нет файлов для агрегации в папке areas")
            return pd.DataFrame()

        logger.info(f"Найдено {len(area_files)} файлов для агрегации")

        for filepath in area_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception as e:
                logger.error(f"Ошибка чтения файла {filepath}: {e}")
                continue

            if not json_data or json_data.get('found', 0) == 0:
                continue

            for item in json_data.get('items', []):
                salary = item.get('salary')
                salary_from = salary.get('from') if salary else None
                salary_to = salary.get('to') if salary else None

                address = item.get('address')
                address_raw = address.get('raw') if address else None

                snippet = item.get('snippet', {})
                employer = item.get('employer', {})

                all_vacancies_data.append([
                    item.get('id'),
                    item.get('name'),
                    item.get('area', {}).get('id'),
                    item.get('area', {}).get('name'),
                    salary_from,
                    salary_to,
                    item.get('published_at'),
                    employer.get('id'),
                    employer.get('name'),
                    snippet.get('requirement'),
                    snippet.get('responsibility'),
                    address_raw,
                    item.get('schedule', {}).get('name'),
                    item.get('employment', {}).get('name'),
                    item.get('experience', {}).get('name'),
                ])

        columns = [
            'id', 'name', 'area_id', 'area_name',
            'salary_from', 'salary_to', 'published_at',
            'employer_id', 'employer_name',
            'requirement', 'responsibility', 'address_raw',
            'schedule_name', 'employment_name', 'experience_name'
        ]
        df = pd.DataFrame(all_vacancies_data, columns=columns)
        logger.info(f"Агрегировано {len(df)} вакансий")
        return df

    def save_to_excel(self, df: pd.DataFrame, filename: str = "vacancies_result.xlsx"):
        """Сохраняет DataFrame в Excel-файл в data/processed/."""
        if df.empty:
            logger.warning("DataFrame пуст, Excel не создан")
            return
        filepath = config.DATA_PROCESSED_DIR / filename
        df.to_excel(filepath, index=False)
        logger.info(f"Результаты сохранены в Excel: {filepath}")