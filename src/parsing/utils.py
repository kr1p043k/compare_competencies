# src/parsing/utils.py
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .skill_normalizer import SkillNormalizer
    from .vacancy_parser import VacancyParser
    from .skill_validator import SkillValidator, ValidationReason

from src import config


# ----------------------------------------------------------------------
# Базовые утилиты (логирование, чтение/запись JSON)
# ----------------------------------------------------------------------

def setup_logging() -> None:
    """Настраивает логирование: вывод в консоль (INFO) и в файл (DEBUG)."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def read_json(filepath: Path) -> Any:
    """Безопасно читает JSON-файл."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Чтение JSON из {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка чтения {filepath}: {e}")
        return None


def write_json(data: Any, filepath: Path) -> None:
    """Безопасно записывает данные в JSON-файл."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Запись JSON в {filepath}")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка записи в {filepath}: {e}")


# ----------------------------------------------------------------------
# Фильтрация навыков по белому списку
# ----------------------------------------------------------------------

def load_it_skills() -> Set[str]:
    """Загружает список допустимых IT-навыков из data/it_skills.json."""
    logger = logging.getLogger(__name__)
    skills_file = config.DATA_DIR / "it_skills.json"
    if not skills_file.exists():
        logger.warning(f"Файл с IT-навыками не найден: {skills_file}. Фильтрация отключена.")
        return set()

    try:
        skills_list = read_json(skills_file)
        if not isinstance(skills_list, list):
            logger.error("Файл it_skills.json должен содержать список строк.")
            return set()
        skills_set = {skill.strip().lower() for skill in skills_list if isinstance(skill, str)}
        logger.info(f"Загружено {len(skills_set)} допустимых IT-навыков.")
        return skills_set
    except Exception as e:
        logger.error(f"Ошибка при загрузке it_skills.json: {e}")
        return set()


def filter_skills_by_whitelist(skills_dict: Dict[str, int], whitelist: Set[str]) -> Dict[str, int]:
    """Оставляет только навыки из whitelist."""
    if not whitelist:
        return skills_dict.copy()
    filtered = {
        skill: count for skill, count in skills_dict.items()
        if skill.lower().strip() in whitelist
    }
    logger = logging.getLogger(__name__)
    logger.info(f"Фильтрация: осталось {len(filtered)} навыков из {len(skills_dict)}")
    return filtered


# ----------------------------------------------------------------------
# Сбор вакансий по множественным запросам/регионам
# ----------------------------------------------------------------------

def collect_vacancies_multiple(
    hh_api,
    queries: List[str],
    area_ids: List[int],
    period_days: int,
    max_pages: int,
    industry: Optional[int] = None,
    max_vacancies_per_query: int = 1000000
) -> List[Dict[str, Any]]:
    """
    Собирает вакансии по комбинациям запросов и регионов.
    Если ожидается больше 2000 вакансий, автоматически разбивает период на интервалы.
    """
    all_vacancies = []
    seen_ids: Set[str] = set()
    logger = logging.getLogger("collector")

    # Порог, после которого включаем разбивку по датам (например, 2000)
    CHUNK_THRESHOLD = 2000
    DATE_CHUNK_DAYS = 5

    for query in queries:
        query_vacancies = []
        for area_id in area_ids:
            logger.info(f"Поиск: '{query}', регион ID {area_id}")

            # Пробный запрос с одной страницей, чтобы оценить количество
            test_vacs = hh_api.search_vacancies(
                text=query,
                area=area_id,
                period_days=period_days,
                max_pages=1,
                per_page=100,
                industry=industry
            )
            last_resp = getattr(hh_api, 'last_response', None)
            total_found = last_resp.get('found', 0) if last_resp else 0

            if total_found <= CHUNK_THRESHOLD or period_days <= DATE_CHUNK_DAYS:
                # Обычный сбор, если вакансий мало или период короткий
                vacs = hh_api.search_vacancies(
                    text=query,
                    area=area_id,
                    period_days=period_days,
                    max_pages=max_pages,
                    per_page=100,
                    industry=industry
                )
                for vac in vacs:
                    vid = vac.get('id')
                    if vid and vid not in seen_ids:
                        seen_ids.add(vid)
                        query_vacancies.append(vac)
                        if len(query_vacancies) >= max_vacancies_per_query:
                            break
                if len(query_vacancies) >= max_vacancies_per_query:
                    break
            else:
                # Разбиваем период на интервалы
                chunks = date_chunks(period_days, DATE_CHUNK_DAYS)
                logger.info(f"Разбиваем запрос на {len(chunks)} интервалов (общий период {period_days} дней)")
                for date_from, date_to in chunks:
                    vacs = hh_api.search_vacancies(
                        text=query,
                        area=area_id,
                        date_from=date_from,
                        date_to=date_to,
                        max_pages=max_pages,
                        per_page=100,
                        industry=industry
                    )
                    for vac in vacs:
                        vid = vac.get('id')
                        if vid and vid not in seen_ids:
                            seen_ids.add(vid)
                            query_vacancies.append(vac)
                            if len(query_vacancies) >= max_vacancies_per_query:
                                break
                    if len(query_vacancies) >= max_vacancies_per_query:
                        break
                    time.sleep(config.REQUEST_DELAY)

            time.sleep(config.REQUEST_DELAY)

        all_vacancies.extend(query_vacancies[:max_vacancies_per_query])
        logger.info(f"Для запроса '{query}' собрано {len(query_vacancies[:max_vacancies_per_query])} вакансий")

    logger.info(f"Всего собрано уникальных вакансий: {len(all_vacancies)}")
    return all_vacancies

def load_queries_from_file(filepath: Path) -> List[str]:
    """Загружает список запросов из текстового файла."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Ошибка чтения файла запросов {filepath}: {e}")
        return []


# ----------------------------------------------------------------------
# Интерактивный режим (используется в test_parsers.py и main.py)
# ----------------------------------------------------------------------

def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(re.sub(r'[^\x00-\x7F]+', '', text))


def input_int(prompt: str, default: int = 30, min_val: int = 1, max_val: int = 30) -> int:
    while True:
        val = input(prompt).strip()
        if not val:
            return default
        try:
            num = int(val)
            if min_val <= num <= max_val:
                return num
            print(f"Введите число от {min_val} до {max_val}")
        except ValueError:
            print("Введите целое число")


def input_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = " (y/n, по умолчанию y)" if default else " (y/n, по умолчанию n)"
    ans = input(prompt + default_text).strip().lower()
    if not ans:
        return default
    return ans in ('y', 'yes', 'да')


def select_from_list(items: List[str], prompt: str) -> str:
    print(prompt)
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    while True:
        try:
            idx = int(input("> ").strip())
            if 1 <= idx <= len(items):
                return items[idx-1]
        except:
            print("Некорректный ввод")


def interactive_config() -> Dict[str, Any]:
    """Интерактивный сбор параметров для поиска."""
    print("\n" + "=" * 90)
    print("ИНТЕРАКТИВНЫЙ СБОР ВАКАНСИЙ С HH.RU")
    print("=" * 90)

    mode_options = [
        "1. Data Scientist",
        "2. Python Developer",
        "3. Java Developer",
        "4. Frontend Developer",
        "5. Backend Developer",
        "6. DevOps Engineer",
        "7. Machine Learning Engineer",
        "8. QA Engineer",
        "9. Системный аналитик",
        "10. Другое (ввести свой запрос)",
        "11. Поиск по всему IT-сектору (industry=7)"
    ]

    selected_mode = select_from_list(mode_options, "\nВыберите вариант поиска:")

    if selected_mode == "11. Поиск по всему IT-сектору (industry=7)":
        print("\nРежим: Поиск по всему IT-сектору")
        positions = [
           # Data & AI
            "Data Scientist", "Data Analyst", "Machine Learning Engineer",
            "Computer Vision Engineer", "NLP Engineer", "Data Architect", "ETL Developer",
            # Development
            "Python Developer", "Java Developer", "Frontend Developer",
            "Backend Developer", "Fullstack Developer", "DevOps Engineer",
            "Embedded Developer", "Blockchain Developer",
            # Mobile
            "iOS Developer", "Android Developer", "React Native Developer", "Flutter Developer",
            # QA
            "QA Engineer", "Automation QA Engineer", "Performance QA Engineer",
            # Security
            "Специалист по кибербезопасности", "Security Engineer", "DevSecOps Engineer",
            # Infrastructure & Administration
            "SRE инженер", "Системный администратор", "Облачный инженер",
            "Сетевой инженер", "Администратор баз данных",
            # Architecture & Management
            "Системный аналитик", "Бизнес-аналитик", "Архитектор программного обеспечения",
            "Solution Architect", "Team Lead", "Tech Lead", "Project Manager IT", "Scrum Master",
            # Design
            "UX/UI дизайнер", "Product Designer",
            # Game Development
            "Unity Developer", "Unreal Engine Developer",
            # Other
            "Technical Writer", "MLops engineer"
        ]
        print("Позиции для поиска:")
        for p in positions:
            print(f"  • {p}")
        queries = positions
        industry = 7
        is_it_sector = True
        query = "IT_Sector_Multiple"
    elif selected_mode == "10. Другое (ввести свой запрос)":
        query = input("\nВведите поисковый запрос: ").strip() or "Data Scientist"
        industry = None
        is_it_sector = False
        queries = [query]
    else:
        query = selected_mode.split('. ', 1)[1]
        industry = None
        is_it_sector = False
        queries = [query]

    # Регионы
    region_options = [
        ("Москва", 1), ("Санкт-Петербург", 2), ("Екатеринбург", 3),
        ("Новосибирск", 4), ("Казань", 88), ("Нижний Новгород", 66),
        ("Ростов-на-Дону", 76), ("Вся Россия", 0)
    ]
    region_names = [f"{name} (ID {rid})" for name, rid in region_options]
    print("\nВыберите регионы (можно несколько, введите номера через пробел):")
    try:
        indices = list(map(int, input("> ").split()))
        selected_regions = [region_names[i-1] for i in indices if 1 <= i <= len(region_names)]
    except:
        selected_regions = [region_names[0]]

    area_ids = [int(re.search(r'ID (\d+)', s).group(1)) for s in selected_regions if re.search(r'ID (\d+)', s)]
    if not area_ids:
        area_ids = [1]

    # Параметры
    if is_it_sector:
        period = 30
        max_pages = 50
        skip_details = False
        show_list = False
        print("\nОграничение: 500 вакансий на одну позицию")
        apply_filter = input_yes_no("Применять фильтрацию по белому списку?", default=False)
        max_vacancies = 100000
    else:
        period = input_int("\nПериод поиска в днях (по умолчанию 30): ", default=30)
        max_pages = input_int("Максимальное количество страниц (по умолчанию 20): ", default=20, max_val=20)
        skip_details = not input_yes_no("Загружать полную информацию по каждой вакансии?", default=True)
        show_list = input_yes_no("Показывать список найденных вакансий?", default=False)
        apply_filter = input_yes_no("Применять фильтрацию по белому списку?", default=True)
        max_vacancies = 2000

    save_excel = input_yes_no("Сохранить результаты в Excel?", default=True)

    return {
        "query": query,
        "queries": queries,
        "area_ids": area_ids,
        "industry": industry,
        "period": period,
        "max_pages": max_pages,
        "skip_details": skip_details,
        "show_vacancies": show_list,
        "excel": save_excel,
        "no_filter": not apply_filter,
        "is_it_sector": is_it_sector,
        "max_vacancies_per_query": max_vacancies
    }


# ----------------------------------------------------------------------
# Обработка навыков (извлечение, подсчёт, маппинг на компетенции)
# ----------------------------------------------------------------------

def normalize_skill_for_matching(skill: str) -> str:
    """Нормализует навык для сопоставления с маппингом."""
    normalized = skill.lower().strip()
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def extract_and_count_skills(
    vacancies: List[Dict[str, Any]],
    parser: "VacancyParser"
) -> Dict[str, Any]:   # теперь возвращает dict с frequencies + tfidf_weights
    logger = logging.getLogger(__name__)

    if not vacancies:
        return {"frequencies": {}, "tfidf_weights": {}}

    try:
        return parser.extract_skills_from_vacancies(vacancies)
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return {"frequencies": {}, "tfidf_weights": {}}

def map_to_competencies(
    skill_frequencies: Dict[str, int],
    mapping: Dict[str, List[str]]
) -> Counter:
    """Сопоставляет рыночные навыки с учебными компетенциями."""
    skill_to_comp = {}
    for comp, keywords in mapping.items():
        for keyword in keywords:
            normalized_keyword = normalize_skill_for_matching(keyword)
            skill_to_comp.setdefault(normalized_keyword, []).append(comp)

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
            # Частичное совпадение
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

    logger = logging.getLogger(__name__)
    logger.info(f"Сопоставлено навыков с компетенциями: {matched_skills} из {len(skill_frequencies)}")
    if unmatched_skills:
        logger.info(f"Примеры несопоставленных навыков: {unmatched_skills[:5]}")
    return comp_counter


def print_top_skills(skill_frequencies: Dict[str, int], top_n: int = 20) -> None:
    """Выводит топ-N навыков в консоль."""
    top = sorted(skill_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print("\n" + "=" * 60)
    print(f"ТОП-{top_n} НАИБОЛЕЕ ВОСТРЕБОВАННЫХ НАВЫКОВ")
    print("=" * 60)
    for i, (skill, count) in enumerate(top, 1):
        print(f"{i:2}. {skill:<50} {count:>4} упоминаний")


def print_top_competencies(comp_counter: Counter, top_n: int = 20) -> None:
    """Выводит топ-N компетенций в консоль."""
    top = comp_counter.most_common(top_n)
    print("\n" + "=" * 60)
    print(f"ТОП-{top_n} УЧЕБНЫХ КОМПЕТЕНЦИЙ НА РЫНКЕ")
    print("=" * 60)
    for i, (comp, freq) in enumerate(top, 1):
        print(f"{i:2}. {comp:<25} {freq:>4} суммарных упоминаний")

def date_chunks(days: int, chunk_size: int = 5) -> List[Tuple[int, int]]:
    """
    Разбивает общий период поиска (в днях) на интервалы по chunk_size дней.
    Возвращает список кортежей (date_from, date_to) в формате Unix timestamp.
    """
    end_date = datetime.now()
    chunks = []
    for offset in range(0, days, chunk_size):
        to_date = end_date - timedelta(days=offset)
        from_date = to_date - timedelta(days=min(chunk_size, days - offset))
        chunks.append((int(from_date.timestamp()), int(to_date.timestamp())))
    return chunks