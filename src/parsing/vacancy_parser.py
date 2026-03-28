"""
Парсер вакансий с поддержкой как старых dict, так и новых типизированных моделей.
"""

import json
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Union
import logging
import pandas as pd
from pathlib import Path

from src import config
from src.parsing.utils import load_it_skills, filter_skills_by_whitelist
from src.models.vacancy import Vacancy
from src.parsing.skill_parser import SkillParser
from src.parsing.skill_validator import SkillValidator
from src.parsing.skill_normalizer import SkillNormalizer

logger = logging.getLogger(__name__)


class VacancyParser:
    """
    Парсер вакансий - совместим с обоими форматами (dict и Vacancy объекты).
    Теперь использует новые парсер, валидатор и нормализатор.
    """

    def __init__(self):
        self.skill_parser = SkillParser()
        self.skill_validator = SkillValidator(
            whitelist=load_it_skills()
        )

    # =========================================================================
    # СОХРАНЕНИЕ
    # =========================================================================

    def save_raw_vacancies(
        self,
        vacancies: Union[List[Dict], List[Vacancy]],
        filename: str = "hh_vacancies.json"
    ):
        """Сохраняет вакансии в JSON (работает с dict и Vacancy)"""
        filepath = config.DATA_RAW_DIR / filename
        
        data_to_save = []
        for vac in vacancies:
            if isinstance(vac, Vacancy):
                data_to_save.append(vac.raw_data)
            else:
                data_to_save.append(vac)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Сырые данные сохранены в {filepath} (вакансий: {len(vacancies)})")

    def save_processed_frequencies(
        self,
        frequencies: Dict[str, int],
        filename: str = "competency_frequency.json",
        apply_filter: bool = True
    ):
        """Сохраняет частоты навыков в JSON"""
        if apply_filter:
            whitelist = load_it_skills()
            if whitelist:
                frequencies = filter_skills_by_whitelist(frequencies, whitelist)
                logger.info(f"Фильтрация применена, осталось {len(frequencies)} навыков")

        filepath = config.DATA_PROCESSED_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(frequencies, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Частоты навыков сохранены в {filepath} (навыков: {len(frequencies)})")

    # =========================================================================
    # ИЗВЛЕЧЕНИЕ НАВЫКОВ
    # =========================================================================

    def extract_skills_from_vacancies(
        self,
        vacancies: Union[List[Dict], List[Vacancy]]
    ) -> Dict[str, int]:
        """
        Извлекает навыки с НОРМАЛИЗАЦИЕЙ для TF-IDF анализа.
        
        Порядок обработки:
        1. Парсинг (извлечение всех найденных навыков)
        2. Нормализация (python3 -> python)
        3. Валидация (удаление мусора)
        4. Дедупликация (удаление дубликатов)
        5. Подсчёт частоты
        """
        all_extracted_skills = []
        
        # Конвертируем в Vacancy если нужно
        vacancy_objects = []
        for vac in vacancies:
            if isinstance(vac, dict):
                try:
                    vacancy_objects.append(Vacancy.from_api(vac))
                except ValueError as e:
                    logger.warning(f"Невалидная вакансия: {e}")
                    continue
            else:
                vacancy_objects.append(vac)
        
        # === ШАГ 1: ПАРСИНГ ===
        for vacancy in vacancy_objects:
            extracted = self.skill_parser.parse_vacancy(vacancy)
            all_extracted_skills.extend(extracted)
        
        logger.info(f"Парсинг завершён: {self.skill_parser.get_stats()}")
        
        # === ШАГ 2: НОРМАЛИЗАЦИЯ ===
        skill_texts = [s.text for s in all_extracted_skills]
        normalized_skills = SkillNormalizer.normalize_batch(skill_texts)
        
        logger.info(f"После нормализации: {len(set(normalized_skills))} уникальных навыков")
        
        # === ШАГ 3: ВАЛИДАЦИЯ ===
        valid_skills, validation_results = self.skill_validator.validate_batch(normalized_skills)
        
        rejection_report = self.skill_validator.get_rejection_report(validation_results)
        logger.info(f"Валидация завершена:")
        logger.info(f"  - Всего проверено: {rejection_report['total_validated']}")
        logger.info(f"  - Валидных: {rejection_report['valid']}")
        logger.info(f"  - Отклонено: {rejection_report['rejected']}")
        if rejection_report['rejection_reasons']:
            logger.info(f"  - Причины отклонений: {rejection_report['rejection_reasons']}")
        
        # === ШАГ 4: ДЕДУПЛИКАЦИЯ ===
        unique_skills = SkillNormalizer.deduplicate(valid_skills)
        logger.info(f"После дедупликации: {len(unique_skills)} навыков")
        
        # === ШАГ 5: ПОДСЧЁТ ЧАСТОТЫ ===
        skill_freq = Counter(unique_skills)
        
        logger.info(f"Итого: {len(skill_freq)} уникальных валидных навыков")
        
        return dict(skill_freq)

    # =========================================================================
    # СТАРЫЕ МЕТОДЫ (для обратной совместимости)
    # =========================================================================

    @staticmethod
    def clean_highlighttext(text: str) -> str:
        """Удаляет теги <highlighttext> из hh.ru"""
        if not text:
            return ""
        text = re.sub(r'</?highlighttext[^>]*>', '', text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def extract_skills(vacancies: List[Dict[str, Any]]) -> List[str]:
        """Старый метод - извлекает ключевые навыки (для совместимости)"""
        all_skills = []
        vacancies_with_skills = 0

        for vacancy in vacancies:
            skills_data = vacancy.get('key_skills', [])
            if skills_data:
                vacancies_with_skills += 1
                for skill_item in skills_data:
                    skill_name = skill_item.get('name', '')
                    if skill_name:
                        clean_name = VacancyParser.clean_highlighttext(skill_name)
                        all_skills.append(clean_name)

        logger.info(f"Из {len(vacancies)} вакансий извлечены навыки из {vacancies_with_skills} (key_skills)")
        return all_skills

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """Старый метод - нормализация (для совместимости)"""
        if not skill:
            return ""

        normalized = VacancyParser.clean_highlighttext(skill).lower().strip()
        normalized = re.sub(r'^(опыт работы с|работа с|знание|владение|умение|должен|требуется|навык|навыки|умение работать с|опыт)\s+', '', normalized)
        normalized = re.sub(r'\s+(опыт|знание|владение|умение|плюсом|желательно|преимуществом|навык|навыки)$', '', normalized)

        if len(normalized.split()) > 4:
            return ""

        if normalized in {"инициатива", "мотивация", "коммуникация", "клиентами", "клиентам", "харизма", "многозадачность"}:
            return ""

        return normalized.strip()

    @staticmethod
    def is_valid_skill(skill: str) -> bool:
        """Старый метод - проверка валидности (для совместимости)"""
        if not skill or len(skill) < 3:
            return False
        return True

    @staticmethod
    def count_skills(skills_list: List[str]) -> Dict[str, int]:
        """Старый метод - подсчёт скиллов (для совместимости)"""
        normalized_skills = [VacancyParser.normalize_skill(s) for s in skills_list if s]
        skill_counts = Counter(normalized_skills)
        filtered = {k: v for k, v in skill_counts.items() if len(k) > 2}
        
        whitelist = load_it_skills()
        if whitelist:
            filtered = {k: v for k, v in filtered.items() if k in whitelist or k.lower() in whitelist}
        
        logger.info(f"После фильтрации осталось {len(filtered)} навыков")
        return filtered

    @staticmethod
    def extract_skills_from_text(vacancies: List[Dict[str, Any]]) -> List[str]:
        """Старый метод - извлечение из текста (для совместимости)"""
        logger.warning("extract_skills_from_text deprecated, используйте extract_skills_from_vacancies")
        return []

    # =========================================================================
    # EXCEL
    # =========================================================================

    def aggregate_to_dataframe(self, vacancies: Union[List[Dict], List[Vacancy]]) -> pd.DataFrame:
        """Агрегирует данные в DataFrame для Excel"""
        rows = []

        for vac in vacancies:
            if isinstance(vac, Vacancy):
                row = {
                    'Вакансия': vac.name,
                    'Компания': vac.employer.name,
                    'Регион': vac.area.name,
                    'ID': vac.id,
                    'Зарплата': str(vac.salary) if vac.salary else 'Не указана',
                    'Навыков': len(vac.key_skills),
                    'Навыки': ', '.join(vac.get_skill_names())
                }
            else:
                vac_name = vac.get('name', 'Unknown')
                employer_name = vac.get('employer', {}).get('name', 'Unknown')
                area_name = vac.get('area', {}).get('name', 'Unknown')
                skills = [s['name'] for s in vac.get('key_skills', [])]

                row = {
                    'Вакансия': vac_name,
                    'Компания': employer_name,
                    'Регион': area_name,
                    'ID': vac.get('id'),
                    'Зарплата': 'Не указана',
                    'Навыков': len(skills),
                    'Навыки': ', '.join(skills)
                }
            
            rows.append(row)

        return pd.DataFrame(rows)

    def save_to_excel(self, df: pd.DataFrame, filename: str):
        """Сохраняет DataFrame в Excel"""
        filepath = config.DATA_PROCESSED_DIR / filename
        df.to_excel(filepath, index=False, engine='openpyxl')
        logger.info(f"Excel файл сохранён в {filepath}")

    def print_vacancies_list(self, vacancies: Union[List[Dict], List[Vacancy]]):
        """Выводит список вакансий"""
        for i, vac in enumerate(vacancies[:20], 1):
            if isinstance(vac, Vacancy):
                print(f"{i}. {vac.name} @ {vac.employer.name} ({vac.area.name})")
                if vac.key_skills:
                    print(f"   Навыки: {', '.join(vac.get_skill_names()[:5])}")
            else:
                vac_name = vac.get('name', 'Unknown')
                employer = vac.get('employer', {}).get('name', 'Unknown')
                area = vac.get('area', {}).get('name', 'Unknown')
                print(f"{i}. {vac_name} @ {employer} ({area})")