import json
import re
from collections import Counter
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from pathlib import Path
from src import config
from src.parsing.utils import load_it_skills, filter_skills_by_whitelist
import nltk

logger = logging.getLogger(__name__)


class VacancyParser:
    """
    Класс для обработки вакансий и извлечения навыков.
    Улучшенная версия: полная очистка от <highlighttext>, лучшая нормализация,
    фильтрация мусора и более точные паттерны.
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

    # Технические навыки для извлечения (множество для быстрого поиска)
    TECH_SKILLS = {
        # Языки программирования
        "python", "python3", "py", "java", "javascript", "js", "typescript", "ts", "c++", "cpp",
        "c#", "csharp", "php", "ruby", "go", "golang", "rust", "swift", "kotlin", "scala",
        "r", "matlab", "sql", "nosql",
        # Фреймворки и библиотеки
        "django", "flask", "fastapi", "spring", "spring boot", "react", "angular", "vue", "vue.js",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
        "spark", "hadoop", "kafka", "airflow", "docker", "kubernetes", "k8s", "jenkins", "git", "github",
        "gitlab", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "clickhouse", "oracle",
        # Облачные технологии
        "aws", "azure", "gcp", "yandex cloud", "cloud",
        # Data Science / ML
        "machine learning", "машинное обучение", "deep learning", "глубокое обучение", "nlp",
        "computer vision", "компьютерное зрение", "data science", "анализ данных", "data analysis",
        "big data", "большие данные", "etl", "data warehouse", "dwh"
    }

    # Синонимы для финальной нормализации (можно расширять)
    SYNONYMS = {
        "javascript": "js",
        "typescript": "ts",
        "golang": "go",
        "vue.js": "vue",
        "postgresql": "postgres",
        "c++": "cpp",
        "c#": "csharp",
        "kubernetes": "k8s",
        "machine learning": "ml",
        "машинное обучение": "ml",
    }

    # Чёрный список для фильтрации мусорных фраз (вынесен на уровень класса)
    BAD_TERMS = {
        "оценка потребностей клиентов", "развитие ключевых клиентов", "проведение презентаций",
        "проведение переговоров", "подготовка коммерческих предложений", "навыки межличностного общения",
        "межличностное общение", "мотивация персонала", "стратегическое мышление", "аналитическое мышление",
        "командная работа", "клиентоориентированность", "ориентация на результат", "продвижение бренда",
        "развитие продаж", "традиционная розница", "собственная розница", "клиентами", "клиентам",
        "инициатива", "харизма", "многозадачность", "бухгалтерская отчетность", "бухгалтерский учет",
        "английского языка", "русского языка", "высшее образование", "знание английского",
        "опыт работы", "умение", "желательно", "продажи", "маркетинг", "smm", "smm-стратегия",
        "hr-аналитика", "управление персоналом", "административная поддержка", "ведение проектов",
        "пластичные смазки", "автомобильные перевозки", "организация клиентских мероприятий",
        "аналитика маркетплейсов", "аналитика маркетплейса", "контроль и анализ ценообразования",
        "построение воронки продаж", "мониторинг рынка", "мониторинг цен", "анализ продаж"
    }

    @staticmethod
    def clean_highlighttext(text: str) -> str:
        """Удаляет все теги <highlighttext> и </highlighttext>, которые вставляет HH.ru."""
        if not text:
            return ""
        # Удаляем открывающие и закрывающие теги (с учётом возможных атрибутов)
        text = re.sub(r'</?highlighttext[^>]*>', '', text, flags=re.IGNORECASE)
        return text.strip()

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
        """Извлекает ключевые навыки из поля key_skills (официальное поле HH)."""
        all_skills = []
        vacancies_with_skills = 0

        for vacancy in vacancies:
            skills_data = vacancy.get('key_skills', [])
            if skills_data:
                vacancies_with_skills += 1
                for skill_item in skills_data:
                    skill_name = skill_item.get('name', '')
                    if skill_name:
                        # Очищаем от возможных highlight-тегов (на всякий случай)
                        clean_name = VacancyParser.clean_highlighttext(skill_name)
                        all_skills.append(clean_name)

        logger.info(f"Из {len(vacancies)} вакансий извлечены навыки из {vacancies_with_skills} (key_skills)")
        return all_skills

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """Радикальная нормализация: жёстко отсекаем мусор и длинные фразы."""
        if not skill:
            return ""

        normalized = VacancyParser.clean_highlighttext(skill).lower().strip()

        # Удаляем типичные префиксы и суффиксы
        normalized = re.sub(r'^(опыт работы с|работа с|знание|владение|умение|должен|требуется|навык|навыки|умение работать с|опыт)\s+', '', normalized)
        normalized = re.sub(r'\s+(опыт|знание|владение|умение|плюсом|желательно|преимуществом|навык|навыки)$', '', normalized)

        # Если фраза слишком длинная (> 4 слов) — почти всегда мусор (например "развитие ключевых клиентов")
        if len(normalized.split()) > 4:
            return ""

        # Убираем одиночные подозрительные слова
        if normalized in {"инициатива", "мотивация", "коммуникация", "клиентами", "клиентам", "харизма", "многозадачность"}:
            return ""

        return normalized.strip()

    @staticmethod
    def is_valid_skill(skill: str) -> bool:
        """Проверяет, что навык не является мусором."""
        if not skill or len(skill) < 3:
            return False
        skill_lower = skill.lower()
        # Проверка на вхождение любого чёрного термина
        if any(bad in skill_lower for bad in VacancyParser.BAD_TERMS):
            return False
        # Дополнительно можно отсечь слишком длинные фразы (>4 слов)
        if len(skill_lower.split()) > 4:
            return False
        return True

    @staticmethod
    def count_skills(skills_list: List[str]) -> Dict[str, int]:
        """Радикальная фильтрация: whitelist + большой blacklist."""
        normalized_skills = [VacancyParser.normalize_skill(s) for s in skills_list if s]
        skill_counts = Counter(normalized_skills)

        # Сначала отсеиваем по blacklist (используем BAD_TERMS)
        filtered = {
            k: v for k, v in skill_counts.items()
            if len(k) > 2 
            and k not in VacancyParser.BAD_TERMS 
            and not any(bad in k for bad in VacancyParser.BAD_TERMS)
        }

        # Затем применяем whitelist (it_skills.json)
        whitelist = load_it_skills()
        if whitelist:
            filtered = {k: v for k, v in filtered.items() if k in whitelist or k.lower() in whitelist}

        logger.info(f"После радикальной фильтрации осталось {len(filtered)} навыков (отфильтровано {len(skill_counts) - len(filtered)} мусорных)")
        return filtered

    @staticmethod
    def extract_skills_from_text(vacancies: List[Dict[str, Any]]) -> List[str]:
        """Извлечение навыков из текстовых полей + n-grams (1–3) + улучшенная фильтрация."""
        
        from nltk.util import ngrams
        
        all_skills = []

        def generate_ngrams(text: str):
            tokens = re.findall(r'\b[a-zA-Zа-яА-Я0-9\+#\.]+\b', text.lower())
            result = set()

            for n in (1, 2, 3):
                for gram in ngrams(tokens, n):
                    phrase = " ".join(gram)
                    if 2 <= len(phrase) <= 50:
                        result.add(phrase)

            return result

        for vacancy in vacancies:
            text_parts = []

            # === сбор текста ===
            snippet = vacancy.get('snippet', {})
            if snippet:
                if snippet.get('requirement'):
                    text_parts.append(VacancyParser.clean_highlighttext(snippet['requirement']))
                if snippet.get('responsibility'):
                    text_parts.append(VacancyParser.clean_highlighttext(snippet['responsibility']))

            if vacancy.get('description'):
                text_parts.append(VacancyParser.clean_highlighttext(vacancy['description']))

            full_text = ' '.join(text_parts).lower()

            if not full_text:
                continue

            skills_found = set()

            # === 1. n-grams (основа) ===
            ngram_phrases = generate_ngrams(full_text)

            for phrase in ngram_phrases:
                if any(tech in phrase for tech in VacancyParser.TECH_SKILLS):
                    skills_found.add(phrase)

            # === 2. Поиск по маркерам ===
            for marker in VacancyParser.SKILL_MARKERS:
                if marker in full_text:
                    parts = full_text.split(marker, 1)
                    if len(parts) > 1:
                        after_marker = parts[1][:600]
                        lines = re.split(r'[\n,•\-*;]+', after_marker)

                        for line in lines:
                            line = line.strip()
                            if 3 < len(line) < 120:
                                if any(tech in line for tech in VacancyParser.TECH_SKILLS):
                                    skills_found.add(line)

            # === 3. Regex паттерны ===
            patterns = [
                r'(?:опыт работы с|опыт с|работа с)\s+([^,.;\n]{3,90})',
                r'(?:знание|владение|умение)\s+([^,.;\n]{3,90})',
                r'(?:должен (?:знать|уметь))\s+([^,.;\n]{3,90})'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, full_text)
                for match in matches:
                    skill = match.strip()
                    if 3 < len(skill) < 90:
                        skills_found.add(skill)

            # === 4. Прямой поиск (как fallback) ===
            for tech_skill in VacancyParser.TECH_SKILLS:
                if re.search(rf'\b{re.escape(tech_skill)}\b', full_text):
                    skills_found.add(tech_skill)

            # === 5. Финальная нормализация ===
            cleaned_skills = set()

            for skill in skills_found:
                norm = VacancyParser.normalize_skill(skill)

                if not norm:
                    continue

                if len(norm) < 2:
                    continue

                if any(bad in norm for bad in [
                    "английского языка", "русского языка", "высшее", "образование"
                ]):
                    continue

                cleaned_skills.add(norm)

            all_skills.extend(cleaned_skills)

        logger.info(f"Из текста извлечено {len(all_skills)} навыков (с n-grams)")
        return all_skills