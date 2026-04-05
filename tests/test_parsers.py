# tests/test_parsers.py
"""
Модульные тесты для парсеров вакансий + SkillNormalizer.
Интерактивная утилита для сбора вакансий с hh.ru.
"""
import logging
import json
import sys
import argparse
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pytest
import requests
from unittest.mock import Mock, patch
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from src.parsing.vacancy_parser import VacancyParser
from src.parsing.hh_api import HeadHunterAPI
from src.parsing.skill_validator import SkillValidator, ValidationReason
from src.parsing.skill_normalizer import SkillNormalizer 
from src.parsing.utils import (
    setup_logging,
    collect_vacancies_multiple,
    interactive_config,
    safe_print,
    extract_and_count_skills,
    print_top_skills
)
from src import config
logger = logging.getLogger(__name__)

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"
TEST_OUTPUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# НОВЫЕ ТЕСТЫ: SkillNormalizer (самое важное сейчас)
# ----------------------------------------------------------------------

class TestSkillNormalizer:
    """Тесты нормализации + fuzzy-matching"""

    def test_synonyms(self):
        assert SkillNormalizer.normalize("Python 3.11") == "python"
        assert SkillNormalizer.normalize("javascript") == "js"
        assert SkillNormalizer.normalize("vue.js") == "vue"
        assert SkillNormalizer.normalize("machine learning") == "ml"
        assert SkillNormalizer.normalize("postgres") == "postgresql"

    def test_version_and_brackets_removal(self):
        assert SkillNormalizer.normalize("React.js v18.2 (latest)") == "react"
        assert SkillNormalizer.normalize("Node.js 20 LTS") == "nodejs"
        assert SkillNormalizer.normalize("Python (3.10)") == "python"

    def test_suffix_removal(self):
        assert SkillNormalizer.normalize("язык python") == "python"
        assert SkillNormalizer.normalize("react framework") == "react"
        assert SkillNormalizer.normalize("postgresql database") == "postgresql"

    def test_fuzzy_matching(self):
        """Проверяем fuzzy fallback"""
        # Должно сработать даже с опечатками и вариациями
        assert SkillNormalizer.normalize("reackt") == "react"
        assert SkillNormalizer.normalize("react native") == "react"      # или "react-native", если добавишь в it_skills
        assert SkillNormalizer.normalize("node js") == "nodejs"
        assert SkillNormalizer.normalize("postgre sql") == "postgresql"

    def test_no_match_returns_cleaned_version(self):
        """Если ничего не подошло — возвращаем очищенную строку"""
        result = SkillNormalizer.normalize("какой-то_мусор_навык_123")
        assert result == "какой-то_мусор_навык_123" or "мусор навык" in result.lower()

    def test_normalize_batch(self):
        skills = ["Python 3", "React.js v18", "reackt", "machine learning"]
        normalized = SkillNormalizer.normalize_batch(skills)
        assert normalized == ["python", "react", "react", "ml"]

    def test_deduplicate(self):
        skills = ["python", "Python 3", "py", "react", "reackt"]
        deduped = SkillNormalizer.deduplicate(skills)
        assert deduped == ["python", "react"]


# ----------------------------------------------------------------------
# СТАРЫЕ ТЕСТЫ (оставлены + чуть улучшены)
# ----------------------------------------------------------------------

@pytest.fixture
def sample_vacancies():
    return [{"id": "123", "name": "Test Vacancy"}]


class TestVacancyParser:
    def test_clean_highlighttext(self):
        parser = VacancyParser()
        assert parser.clean_highlighttext("Python <highlighttext>test</highlighttext>") == "Python test"
        
    def test_extract_skills_from_description(self):
        parser = VacancyParser()
        desc = "Требуется знание Python, FastAPI, PostgreSQL и Docker."
        skills = parser.extract_skills_from_description(desc)
        assert len(skills) > 0
        assert "python" in [s.lower() for s in skills]


    def test_extract_and_count_skills_with_weights():
        """Проверка весов (имитация skill_weights + простой TF-IDF)"""
        parser = VacancyParser()
        sample_vacs = [
            {"description": "Python FastAPI PostgreSQL"},
            {"description": "Python Docker Kubernetes"},
            {"key_skills": [{"name": "Python"}, {"name": "FastAPI"}]}
        ]

        result = extract_and_count_skills(sample_vacs, parser)  # теперь возвращает dict
        freq = result["frequencies"]
        weights = result.get("tfidf_weights", {})

        assert "python" in freq
        assert freq["python"] >= 2   # ← теперь будет больше 1
        assert len(weights) > 0
        logger.info(f"TF-IDF веса: {weights}")

        # === TF-IDF имитация (как ты просил) ===
        texts = [vac.get("description", "") or "" for vac in sample_vacs]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        tfidf_weights = {}
        for i, name in enumerate(feature_names):
            weight = tfidf_matrix[:, i].mean()
            if weight > 0.1:
                tfidf_weights[name] = round(float(weight), 4)

        logger.info(f"TF-IDF веса: {tfidf_weights}")
        assert len(tfidf_weights) > 0  # хотя бы один вес должен быть


class TestHeadHunterAPI:
    @patch('src.parsing.hh_api.requests.get')
    def test_search_vacancies_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [{"id": "1"}], "pages": 1}
        mock_get.return_value = mock_response
        api = HeadHunterAPI()
        result = api.search_vacancies(text="Python", area=1)
        assert len(result) == 1


# ----------------------------------------------------------------------
# Основная функция (НИЧЕГО НЕ МЕНЯЛ — интерактив и запуск файла работают как раньше)
# ----------------------------------------------------------------------

def run_search(args: argparse.Namespace = None, interactive: bool = False):
    if interactive or args is None or len(sys.argv) == 1:
        params = interactive_config()
        args = argparse.Namespace(**params)

    setup_logging()
    logger = logging.getLogger("hh_collector")

    logger.info("=" * 90)
    logger.info("ЗАПУСК СБОРА ВАКАНСИЙ С HH.RU")
    logger.info("=" * 90)

    queries = getattr(args, 'queries', [args.query])
    area_ids = args.area_ids
    sector_suffix = "_it_sector" if getattr(args, 'is_it_sector', False) else ""
    query_part = re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁ_-]', '_', args.query)[:40]

    max_vacancies_limit = 500 if getattr(args, 'is_it_sector', False) else 1000

    original_raw = config.DATA_RAW_DIR
    original_proc = config.DATA_PROCESSED_DIR
    config.DATA_RAW_DIR = config.DATA_PROCESSED_DIR = TEST_OUTPUT_DIR

    try:
        hh_api = HeadHunterAPI()
        parser = VacancyParser()

        logger.info(f"Запрос(ы): {queries} | Регионы: {area_ids}")

        basic_vacancies = collect_vacancies_multiple(
            hh_api=hh_api,
            queries=queries,
            area_ids=area_ids,
            period_days=args.period,
            max_pages=args.max_pages,
            industry=getattr(args, 'industry', None),
            max_vacancies_per_query=max_vacancies_limit
        )

        if not basic_vacancies:
            logger.error("Вакансий не найдено.")
            return

        # Загружаем детальную информацию
        logger.info(f"Загружаем детальную информацию для {len(basic_vacancies)} вакансий...")
        vacancies_to_process = []
        for i, vac in enumerate(basic_vacancies, 1):
            if i % 50 == 0:
                logger.info(f"Прогресс деталей: {i}/{len(basic_vacancies)}")
            det = hh_api.get_vacancy_details(vac['id'])
            if det:
                vacancies_to_process.append(det)
            time.sleep(config.REQUEST_DELAY)

        if not vacancies_to_process:
            logger.error("Не удалось загрузить ни одной детальной вакансии")
            return

        if args.show_vacancies:
            parser.print_vacancies_list(vacancies_to_process)

        # === ИСПРАВЛЕННЫЙ ВЫЗОВ ===
        result = extract_and_count_skills(vacancies_to_process, parser)
        skill_freq = result["frequencies"]          # ← только частоты
        tfidf_weights = result.get("tfidf_weights", {})

        if not skill_freq:
            logger.error("Не удалось извлечь навыки.")
            return

        # Сохранение
        raw_file = f"raw_vacancies{sector_suffix}_{query_part}.json"
        parser.save_raw_vacancies(vacancies_to_process, raw_file)

        parser.save_processed_frequencies(skill_freq, f"freq{sector_suffix}_{query_part}_raw.json", apply_filter=False)
        parser.save_processed_frequencies(skill_freq, f"freq{sector_suffix}_{query_part}_filtered.json", apply_filter=not args.no_filter)

        # Вывод
        print_top_skills(skill_freq)

        # Новый вывод TF-IDF весов
        if tfidf_weights:
            print("\n" + "=" * 60)
            print("ТОП-15 НАВЫКОВ ПО TF-IDF ВЕСУ")
            print("=" * 60)
            top_weights = sorted(tfidf_weights.items(), key=lambda x: x[1], reverse=True)[:15]
            for i, (skill, weight) in enumerate(top_weights, 1):
                print(f"{i:2}. {skill:<40} {weight:.4f}")

        if args.excel:
            df = parser.aggregate_to_dataframe(vacancies_to_process)
            excel_name = f"vacancies{sector_suffix}_{query_part}.xlsx"
            parser.save_to_excel(df, excel_name)
            logger.info(f"Excel сохранён: {excel_name}")

        logger.info(f"Все файлы сохранены в: {TEST_OUTPUT_DIR}")

    finally:
        config.DATA_RAW_DIR = original_raw
        config.DATA_PROCESSED_DIR = original_proc
class TestSkillValidator:
    """Тесты для SkillValidator — полное покрытие всех проверок."""

    @pytest.fixture
    def validator(self):
        """Маленький whitelist + стандартный blacklist для изолированных тестов."""
        whitelist = {
            "python", "java", "javascript", "sql", "machine learning",
            "data analysis", "react", "docker", "kubernetes", "aws",
            "postgresql", "fastapi", "typescript"
        }
        return SkillValidator(whitelist=whitelist)

    def test_validate_single_valid(self, validator):
        result = validator.validate("Python")
        assert result.is_valid is True
        assert result.skill == "Python"
        assert result.confidence == 0.95
        assert not result.reasons

    def test_validate_single_empty(self, validator):
        result = validator.validate("")
        assert result.is_valid is False
        assert ValidationReason.EMPTY in result.reasons

    def test_validate_single_too_short(self, validator):
        result = validator.validate("js")
        assert result.is_valid is False
        assert ValidationReason.TOO_SHORT in result.reasons

    def test_validate_single_too_many_words(self, validator):
        result = validator.validate("очень длинное название навыка которое точно больше четырех слов")
        assert result.is_valid is False
        assert ValidationReason.TOO_LONG in result.reasons

    def test_validate_single_blacklist(self, validator):
        result = validator.validate("английский язык")          # из DEFAULT_BLACKLIST
        assert result.is_valid is False
        assert ValidationReason.IN_BLACKLIST in result.reasons

    def test_validate_single_filler_words(self, validator):
        result = validator.validate("умение работать")           # все слова — паразиты
        assert result.is_valid is False
        assert ValidationReason.IN_BLACKLIST in result.reasons

    def test_validate_single_not_in_whitelist(self, validator):
        result = validator.validate("csharp")
        assert result.is_valid is False
        assert ValidationReason.NOT_IN_WHITELIST in result.reasons

    def test_validate_batch_all_valid(self, validator):
        skills = ["python", "java", "sql"]
        valid_skills, results = validator.validate_batch(skills)

        assert valid_skills == skills
        assert len(results) == 3
        assert all(r.is_valid for r in results)

    def test_validate_batch_all_invalid(self, validator):
        skills = ["csharp", "php", "some unknown skill 2025"]
        valid_skills, results = validator.validate_batch(skills)

        assert valid_skills == []
        assert len(results) == 3
        assert all(not r.is_valid for r in results)

    def test_validate_batch_mixed(self, validator):
        skills = ["python", "unknown skill", "java", "another fake skill", "typescript"]
        valid_skills, results = validator.validate_batch(skills)

        assert valid_skills == ["python", "java", "typescript"]
        assert len(results) == 5

    def test_get_rejection_report(self, validator):
        skills = ["python", "unknown1", "java", "unknown2", "английский язык"]
        _, results = validator.validate_batch(skills)
        report = validator.get_rejection_report(results)

        assert isinstance(report, dict)
        assert report['total_validated'] == 5
        assert report['valid'] == 2
        assert report['rejected'] == 3
        assert 'rejection_reasons' in report
        assert report['rejection_reasons'][ValidationReason.NOT_IN_WHITELIST.value] >= 2
        assert report['rejection_reasons'][ValidationReason.IN_BLACKLIST.value] >= 1

    def test_empty_batch(self, validator):
        skills = []
        valid_skills, results = validator.validate_batch(skills)
        report = validator.get_rejection_report(results)

        assert valid_skills == []
        assert results == []
        assert report['total_validated'] == 0
        assert report['valid'] == 0
        assert report['rejected'] == 0

    def test_rejection_report_structure_from_real_data(self, validator):
        """Проверка на реальных навыках из твоего freq_backend_developer_filtered.json"""
        sample_skills = [
            "node.js", "fastapi", "docker", "hadoop", "rest api",
            "английский язык", "аналитическое мышление", "тестирование по"
        ]
        _, results = validator.validate_batch(sample_skills)
        report = validator.get_rejection_report(results)

        assert report['rejected'] >= 3
        assert ValidationReason.IN_BLACKLIST.value in report['rejection_reasons']


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_search(interactive=True)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--interactive', action='store_true')
        args = parser.parse_args()
        run_search(args, interactive=args.interactive)