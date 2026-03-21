import json
import re
from collections import Counter
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from pathlib import Path
from src import config
from src.parsing.utils import load_it_skills, filter_skills_by_whitelist

logger = logging.getLogger(__name__)

class VacancyParser:
    """
    Класс для обработки вакансий и извлечения навыков.
    """

    # Расширенный список маркеров для поиска навыков
    SKILL_MARKERS = [
        "ключевые навыки", "ключевые навыки:", "ключевые компетенции", "ключевые компетенции:",
        "требования", "требования:", "требования к кандидату", "требования к кандидату:",
        "необходимые навыки", "необходимые навыки:", "навыки", "навыки:",
        "мы ждем", "мы ждем:", "ожидаем от вас", "ожидаем от вас:",
        "что нужно знать", "что нужно знать:", "что вы умеете", "что вы умеете:",
        "профессиональные навыки", "профессиональные навыки:",
        "опыт работы с", "опыт работы:", "знание", "знание:",
        "уверенное владение", "владение", "понимание",
        "должен уметь", "должен знать",
        "stack", "технологии", "технологии:",
        "инструменты", "инструменты:"
    ]
    
    # Технические навыки для извлечения (популярные технологии)
    TECH_SKILLS = {
        # Языки программирования
        "python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby", "go", "golang", 
        "rust", "swift", "kotlin", "scala", "r", "matlab", "sql", "nosql",
        # Фреймворки и библиотеки
        "django", "flask", "fastapi", "spring", "spring boot", "react", "angular", "vue", "vue.js",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
        "spark", "hadoop", "kafka", "airflow", "docker", "kubernetes", "jenkins", "git", "github",
        # Базы данных
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "clickhouse", "oracle",
        # Облачные технологии
        "aws", "azure", "gcp", "yandex cloud", "cloud",
        # Data Science / ML
        "machine learning", "машинное обучение", "deep learning", "глубокое обучение", "nlp",
        "computer vision", "компьютерное зрение", "data science", "анализ данных", "data analysis",
        "big data", "большие данные", "etl", "data warehouse", "dwh"
    }

    def save_raw_vacancies(self, vacancies: List[Dict[str, Any]], filename: str = "hh_vacancies.json"):
        """Сохраняет сырые данные о вакансиях в JSON-файл."""
        filepath = config.DATA_RAW_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=2)
        logger.info(f"Сырые данные сохранены в {filepath} (вакансий: {len(vacancies)})")

    def save_processed_frequencies(self, frequencies: Dict[str, int], filename: str = "competency_frequency.json", apply_filter: bool = True):
        """Сохраняет частоты навыков в JSON."""
        if apply_filter:
            whitelist = load_it_skills()
            if whitelist:
                frequencies = filter_skills_by_whitelist(frequencies, whitelist)
                logger.info(f"Фильтрация применена, осталось {len(frequencies)} навыков")
            else:
                logger.warning("Белый список не загружен, фильтрация пропущена.")

        filepath = config.DATA_PROCESSED_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(frequencies, f, ensure_ascii=False, indent=2)
        logger.info(f"Частоты навыков сохранены в {filepath} (навыков: {len(frequencies)})")

    @staticmethod
    def extract_skills(vacancies: List[Dict[str, Any]]) -> List[str]:
        """Извлекает ключевые навыки из поля key_skills."""
        all_skills = []
        vacancies_with_skills = 0

        for vacancy in vacancies:
            skills_data = vacancy.get('key_skills', [])
            if skills_data:
                vacancies_with_skills += 1
                for skill_item in skills_data:
                    skill_name = skill_item.get('name', '')
                    if skill_name:
                        all_skills.append(skill_name)

        logger.info(f"Из {len(vacancies)} вакансий извлечены навыки из {vacancies_with_skills}")
        if all_skills:
            logger.debug(f"Пример первых 10 навыков: {all_skills[:10]}")

        return all_skills

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """Приводит навык к единому виду для корректного подсчёта."""
        normalized = skill.lower().strip()
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    @staticmethod
    def count_skills(skills_list: List[str]) -> Dict[str, int]:
        """Подсчитывает частоту каждого навыка после нормализации."""
        normalized_skills = [VacancyParser.normalize_skill(s) for s in skills_list]
        skill_counts = Counter(normalized_skills)
        # Фильтруем слишком короткие навыки
        filtered_counts = {k: v for k, v in skill_counts.items() if len(k) > 2}
        logger.info(f"Подсчитаны частоты навыков: {len(filtered_counts)} уникальных (было {len(skill_counts)})")
        return filtered_counts

    @staticmethod
    def extract_skills_from_text(vacancies: List[Dict[str, Any]]) -> List[str]:
        """Улучшенное извлечение навыков из текстовых полей вакансий."""
        all_skills = []
        
        for vacancy in vacancies:
            text_parts = []
            
            # Собираем текст из разных полей
            snippet = vacancy.get('snippet', {})
            if snippet:
                if snippet.get('requirement'):
                    text_parts.append(snippet['requirement'])
                if snippet.get('responsibility'):
                    text_parts.append(snippet['responsibility'])
            
            if vacancy.get('description'):
                text_parts.append(vacancy['description'])
            
            full_text = ' '.join(text_parts).lower()
            
            # 1. Ищем по маркерам
            skills_found = set()
            for marker in VacancyParser.SKILL_MARKERS:
                if marker in full_text:
                    # Находим текст после маркера
                    parts = full_text.split(marker, 1)
                    if len(parts) > 1:
                        after_marker = parts[1]
                        # Берем следующий абзац или до 200 символов
                        end_pos = min(500, len(after_marker))
                        skill_text = after_marker[:end_pos]
                        
                        # Разбиваем по строкам и запятым
                        lines = re.split(r'[\n,•\-*•]+', skill_text)
                        for line in lines:
                            line = line.strip()
                            if line and len(line) > 2 and len(line) < 100:
                                # Проверяем, содержит ли строка технические навыки
                                if any(tech in line for tech in VacancyParser.TECH_SKILLS):
                                    skills_found.add(line)
            
            # 2. Ищем по словарю технических навыков
            for tech_skill in VacancyParser.TECH_SKILLS:
                if tech_skill in full_text:
                    skills_found.add(tech_skill)
            
            # 3. Ищем паттерны типа "опыт работы с X"
            exp_pattern = r'опыт работы с ([^,.;\n]+)'
            matches = re.findall(exp_pattern, full_text)
            for match in matches:
                skill = match.strip()
                if len(skill) > 2 and len(skill) < 50:
                    skills_found.add(skill)
            
            # 4. Ищем паттерны "знание X"
            knowledge_pattern = r'знание ([^,.;\n]+)'
            matches = re.findall(knowledge_pattern, full_text)
            for match in matches:
                skill = match.strip()
                if len(skill) > 2 and len(skill) < 50:
                    skills_found.add(skill)
            
            # 5. Ищем паттерны "владение X"
            skill_pattern = r'владение ([^,.;\n]+)'
            matches = re.findall(skill_pattern, full_text)
            for match in matches:
                skill = match.strip()
                if len(skill) > 2 and len(skill) < 50:
                    skills_found.add(skill)
            
            all_skills.extend(list(skills_found))
        
        logger.info(f"Из текста извлечено {len(all_skills)} сырых навыков")
        return all_skills

    @staticmethod
    def print_vacancies_list(vacancies: List[Dict[str, Any]], limit: int = 30):
        """Выводит список вакансий в консоль."""
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
            employer_name = employer.get('name', 'N/A') if employer else 'N/A'
            if len(employer_name) > 32:
                employer_name = employer_name[:29] + "..."

            salary = item.get('salary')
            if salary and isinstance(salary, dict):
                salary_from = salary.get('from')
                salary_to = salary.get('to')
                currency = salary.get('currency', 'RUR')
                if salary_from and salary_to:
                    salary_str = f"{salary_from}-{salary_to} {currency}"
                elif salary_from:
                    salary_str = f"от {salary_from} {currency}"
                elif salary_to:
                    salary_str = f"до {salary_to} {currency}"
                else:
                    salary_str = "Не указана"
            else:
                salary_str = "Не указана"

            if len(salary_str) > 24:
                salary_str = salary_str[:21] + "..."

            skills_data = item.get('key_skills', [])
            skills_count = len(skills_data) if skills_data else 0

            print(f"{idx:<4} {vacancy_id:<10} {name:<50} {employer_name:<35} {salary_str:<25} {skills_count:<8}")

        if len(vacancies) > limit:
            print(f"... и ещё {len(vacancies) - limit} вакансий")
        print("=" * 140)

    def aggregate_to_dataframe(self, vacancies: List[Dict[str, Any]]) -> pd.DataFrame:
        """Преобразует список детальных вакансий в DataFrame."""
        if not vacancies:
            return pd.DataFrame()

        data = []
        for v in vacancies:
            salary = v.get('salary') if isinstance(v.get('salary'), dict) else None
            area = v.get('area') if isinstance(v.get('area'), dict) else {}
            employer = v.get('employer') if isinstance(v.get('employer'), dict) else {}
            snippet = v.get('snippet') if isinstance(v.get('snippet'), dict) else {}
            schedule = v.get('schedule') if isinstance(v.get('schedule'), dict) else {}
            employment = v.get('employment') if isinstance(v.get('employment'), dict) else {}
            experience = v.get('experience') if isinstance(v.get('experience'), dict) else {}

            salary_from = None
            salary_to = None
            salary_currency = None
            if salary:
                salary_from = salary.get('from')
                salary_to = salary.get('to')
                salary_currency = salary.get('currency')

            key_skills_list = v.get('key_skills', [])
            if key_skills_list and isinstance(key_skills_list, list):
                key_skills = ', '.join([
                    s.get('name', '') for s in key_skills_list
                    if isinstance(s, dict) and s.get('name')
                ])
            else:
                key_skills = ''

            data.append({
                'id': v.get('id'),
                'name': v.get('name'),
                'area_id': area.get('id') if area else None,
                'area_name': area.get('name') if area else None,
                'employer_id': employer.get('id') if employer else None,
                'employer_name': employer.get('name') if employer else None,
                'salary_from': salary_from,
                'salary_to': salary_to,
                'salary_currency': salary_currency,
                'experience': experience.get('name') if experience else None,
                'employment': employment.get('name') if employment else None,
                'schedule': schedule.get('name') if schedule else None,
                'description': v.get('description'),
                'requirement': snippet.get('requirement') if snippet else None,
                'responsibility': snippet.get('responsibility') if snippet else None,
                'key_skills': key_skills,
                'published_at': v.get('published_at'),
                'created_at': v.get('created_at'),
                'alternate_url': v.get('alternate_url')
            })

        df = pd.DataFrame(data)
        logger.info(f"Создан DataFrame с {len(df)} вакансиями и {len(df.columns)} колонками")
        return df

    def save_to_excel(self, df: pd.DataFrame, filename: str = "vacancies_result.xlsx"):
        """Сохраняет DataFrame в Excel-файл."""
        if df.empty:
            logger.warning("DataFrame пуст, Excel не создан")
            return
        filepath = config.DATA_PROCESSED_DIR / filename
        df.to_excel(filepath, index=False)
        logger.info(f"Результаты сохранены в Excel: {filepath}")