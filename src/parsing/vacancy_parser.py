"""
Парсер вакансий с поддержкой как старых dict, так и новых типизированных моделей.
Исправленная версия с корректным подсчётом частот навыков.
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
    Исправлен подсчёт частот навыков (считает вхождения, а не уникальные навыки).
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
    # ИЗВЛЕЧЕНИЕ НАВЫКОВ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    # =========================================================================

    def extract_skills_from_vacancies(
        self,
        vacancies: Union[List[Dict], List[Vacancy]]
    ) -> Dict[str, int]:
        """
        Извлекает навыки с ПОДСЧЁТОМ ЧАСТОТ (сколько раз навык встречается во всех вакансиях)
        
        Порядок обработки:
        1. Парсинг (извлечение всех найденных навыков с дубликатами)
        2. Нормализация (python3 -> python)
        3. Валидация (удаление мусора)
        4. Подсчёт частоты (сколько раз каждый навык встречается)
        5. Возврат словаря {навык: частота}
        
        Args:
            vacancies: Список вакансий (dict или Vacancy объекты)
        
        Returns:
            Словарь {навык: частота упоминаний}
        """
        logger.info(f"Начинаем извлечение навыков из {len(vacancies)} вакансий...")
        
        # Конвертируем в Vacancy если нужно
        vacancy_objects = []
        invalid_count = 0
        for vac in vacancies:
            if isinstance(vac, dict):
                try:
                    vacancy_objects.append(Vacancy.from_api(vac))
                except ValueError as e:
                    logger.debug(f"Невалидная вакансия: {e}")
                    invalid_count += 1
                    continue
            else:
                vacancy_objects.append(vac)
        
        if invalid_count > 0:
            logger.warning(f"Пропущено {invalid_count} невалидных вакансий")
        
        logger.info(f"Обрабатывается {len(vacancy_objects)} валидных вакансий")
        
        # === ШАГ 1: ПАРСИНГ (собираем ВСЕ навыки, включая дубликаты) ===
        all_extracted_skills = []
        vacancies_with_skills = 0
        
        for vacancy in vacancy_objects:
            extracted = self.skill_parser.parse_vacancy(vacancy)
            if extracted:
                vacancies_with_skills += 1
                all_extracted_skills.extend(extracted)
        
        logger.info(f"Парсинг завершён:")
        logger.info(f"  - Вакансий с навыками: {vacancies_with_skills}/{len(vacancy_objects)}")
        logger.info(f"  - Извлечено сырых навыков: {len(all_extracted_skills)}")
        
        if not all_extracted_skills:
            logger.warning("Не удалось извлечь ни одного навыка!")
            return {}
        
        # === ШАГ 2: НОРМАЛИЗАЦИЯ (приводим к единому формату) ===
        skill_texts = [s.text for s in all_extracted_skills]
        normalized_skills = SkillNormalizer.normalize_batch(skill_texts)
        
        unique_before_norm = len(set(skill_texts))
        unique_after_norm = len(set(normalized_skills))
        logger.info(f"Нормализация: {unique_before_norm} → {unique_after_norm} уникальных")
        
        # === ШАГ 3: ВАЛИДАЦИЯ (удаляем мусор) ===
        valid_skills, validation_results = self.skill_validator.validate_batch(normalized_skills)
        
        rejection_report = self.skill_validator.get_rejection_report(validation_results)
        logger.info(f"Валидация завершена:")
        logger.info(f"  - Всего проверено: {rejection_report['total_validated']}")
        logger.info(f"  - Валидных: {rejection_report['valid']}")
        logger.info(f"  - Отклонено: {rejection_report['rejected']}")
        
        if rejection_report['rejection_reasons']:
            logger.info(f"  - Причины отклонений: {rejection_report['rejection_reasons']}")
        
        if not valid_skills:
            logger.warning("После валидации не осталось ни одного навыка!")
            return {}
        
        # === ШАГ 4: ПОДСЧЁТ ЧАСТОТЫ (важно: считаем количество вхождений, а не уникальных) ===
        skill_freq = Counter(valid_skills)
        
        # === ШАГ 5: ФИЛЬТРАЦИЯ ПО МИНИМАЛЬНОЙ ЧАСТОТЕ (опционально) ===
        # Удаляем навыки, которые встречаются слишком редко (меньше 3 раз)
        min_frequency = 3
        filtered_freq = {k: v for k, v in skill_freq.items() if v >= min_frequency}
        
        removed_rare = len(skill_freq) - len(filtered_freq)
        if removed_rare > 0:
            logger.info(f"Удалено редких навыков (менее {min_frequency} упоминаний): {removed_rare}")
        
        # === ИТОГОВАЯ СТАТИСТИКА ===
        total_occurrences = sum(filtered_freq.values())
        logger.info(f"\n{'='*60}")
        logger.info(f"ИТОГОВАЯ СТАТИСТИКА ИЗВЛЕЧЕНИЯ НАВЫКОВ:")
        logger.info(f"{'='*60}")
        logger.info(f"  Уникальных навыков: {len(filtered_freq)}")
        logger.info(f"  Всего упоминаний: {total_occurrences}")
        logger.info(f"  Средняя частота: {total_occurrences / len(filtered_freq):.1f}")
        
        # Выводим топ-20 для проверки
        top_skills = sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        logger.info(f"\nТОП-20 НАВЫКОВ ПО ЧАСТОТЕ:")
        for i, (skill, count) in enumerate(top_skills, 1):
            logger.info(f"  {i:2}. {skill:<30} → {count:>3} упоминаний")
        
        return filtered_freq

    # =========================================================================
    # МЕТОД ДЛЯ ОТЛАДКИ (показывает статистику по источникам навыков)
    # =========================================================================

    def extract_skills_with_stats(
        self,
        vacancies: Union[List[Dict], List[Vacancy]]
    ) -> Dict[str, Any]:
        """
        Расширенная версия extract_skills_from_vacancies с подробной статистикой.
        Возвращает словарь с навыками и метаданными.
        
        Returns:
            {
                'frequencies': Dict[str, int],
                'stats': {
                    'total_extracted': int,
                    'by_source': Dict[str, int],
                    'by_confidence': Dict[str, int]
                }
            }
        """
        # Конвертируем в Vacancy если нужно
        vacancy_objects = []
        for vac in vacancies:
            if isinstance(vac, dict):
                try:
                    vacancy_objects.append(Vacancy.from_api(vac))
                except ValueError:
                    continue
            else:
                vacancy_objects.append(vac)
        
        # Собираем все навыки с метаданными
        all_skills = []
        stats_by_source = Counter()
        stats_by_confidence = Counter()
        
        for vacancy in vacancy_objects:
            extracted = self.skill_parser.parse_vacancy(vacancy)
            for skill in extracted:
                all_skills.append(skill)
                stats_by_source[skill.source.value] += 1
                # Категоризируем по уверенности
                if skill.confidence >= 0.9:
                    stats_by_confidence['high'] += 1
                elif skill.confidence >= 0.7:
                    stats_by_confidence['medium'] += 1
                else:
                    stats_by_confidence['low'] += 1
        
        # Нормализуем и валидируем
        skill_texts = [s.text for s in all_skills]
        normalized = SkillNormalizer.normalize_batch(skill_texts)
        valid, _ = self.skill_validator.validate_batch(normalized)
        
        # Подсчитываем частоты
        skill_freq = Counter(valid)
        
        logger.info(f"\n{'='*60}")
        logger.info("ДЕТАЛЬНАЯ СТАТИСТИКА ИЗВЛЕЧЕНИЯ:")
        logger.info(f"{'='*60}")
        logger.info(f"Источники навыков:")
        for source, count in stats_by_source.most_common():
            logger.info(f"  {source}: {count}")
        logger.info(f"\nУверенность извлечения:")
        for level, count in stats_by_confidence.items():
            logger.info(f"  {level}: {count}")
        
        return {
            'frequencies': dict(skill_freq),
            'stats': {
                'total_extracted': len(all_skills),
                'by_source': dict(stats_by_source),
                'by_confidence': dict(stats_by_confidence)
            }
        }

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