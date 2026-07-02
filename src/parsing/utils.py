# src/parsing/utils.py
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from .skills.vacancy_parser import VacancyParser

from src import Err, Ok, Result, config
from src.errors import DomainError
from src.utils import atomic_read_json, atomic_write_json

logger = structlog.get_logger(__name__)


def _write_progress_pct(pct: int, message: str):
    from src.pipeline.progress import write
    write(pct, message)

__all__ = [
    "setup_logging",
    "read_json",
    "write_json",
    "load_it_skills",
    "filter_skills_by_whitelist",
    "collect_vacancies_multiple",
    "load_queries_from_file",
    "safe_print",
    "input_int",
    "input_yes_no",
    "select_from_list",
    "interactive_config",
    "normalize_skill_for_matching",
    "extract_and_count_skills",
    "map_to_competencies",
    "print_top_skills",
    "print_top_competencies",
    "date_chunks",
    "get_last_parsed_id",
    "save_last_parsed_id",
    "ParsingCheckpoint",
    "save_checkpoint",
    "load_checkpoint",
    "resume_from_checkpoint",
]


# ----------------------------------------------------------------------
# Базовые утилиты (логирование, чтение/запись JSON)
# ----------------------------------------------------------------------


def setup_logging() -> None:
    """Настраивает логирование: вывод в консоль (INFO) и в файл (DEBUG)."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def read_json(filepath: Path) -> Result[Any, DomainError]:
    """Безопасно читает JSON-файл."""
    logger.debug("reading_json", path=str(filepath))
    try:
        with open(filepath, encoding="utf-8") as f:
            return Ok(json.load(f))
    except Exception as e:
        return Err(DomainError(message="JSON read error", detail=str(e)))


def write_json(data: Any, filepath: Path) -> Result[None, DomainError]:
    """Безопасно записывает данные в JSON-файл."""
    logger.debug("writing_json", path=str(filepath))
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return Ok(None)
    except Exception as e:
        return Err(DomainError(message="JSON write error", detail=str(e)))


# ----------------------------------------------------------------------
# Фильтрация навыков по белому списку
# ----------------------------------------------------------------------


def load_it_skills_result() -> Result[set[str], DomainError]:
    skills_file = config.IT_SKILLS_PATH
    if not skills_file.exists():
        return Err(DomainError(message="IT skills file not found", detail=str(skills_file)))
    match read_json(skills_file):
        case Ok(data):
            if not isinstance(data, list):
                return Err(DomainError(message="IT skills invalid format"))
            skills_set = {skill.strip().lower() for skill in data if isinstance(skill, str)}
            return Ok(skills_set)
        case Err(e):
            return Err(DomainError(message="IT skills load error", detail=str(e)))


def load_it_skills() -> set[str]:
    match load_it_skills_result():
        case Ok(skills_set):
            return skills_set
        case Err(e):
            logger.warning("it_skills_load_failed", error=str(e))
            return set()


def filter_skills_by_whitelist(skills_dict: dict[str, int], whitelist: set[str]) -> dict[str, int]:
    """Оставляет только навыки из whitelist."""
    if not whitelist:
        return skills_dict.copy()
    filtered = {skill: count for skill, count in skills_dict.items() if skill.lower().strip() in whitelist}
    logger.info("whitelist_filter_applied", remaining=len(filtered), total=len(skills_dict))
    return filtered


# ----------------------------------------------------------------------
# Сбор вакансий по множественным запросам/регионам
# ----------------------------------------------------------------------


IT_PROFESSIONAL_ROLES = [
    "10", "157", "156", "150", "36", "125", "165", "160",
    "104", "25", "116", "112", "164", "73", "96", "107", "113",
    "114", "148", "121", "126", "124",
]


def collect_vacancies_multiple(
    hh_api,
    queries: list[str],
    area_ids: list[int],
    period_days: int,
    max_pages: int,
    industry: int | None = None,
    max_vacancies_per_query: int = 1000000,
    professional_role: str | None = None,
) -> list[dict[str, Any]]:
    """
    Собирает вакансии по комбинациям запросов и регионов.
    Если ожидается больше 2000 вакансий, автоматически разбивает период на интервалы.
    """
    all_vacancies = []
    seen_ids: set[str] = set()

    chunk_threshold = 2000
    date_chunk_days = 5

    total_combos = len(queries) * len(area_ids)
    combo_idx = 0

    for query in queries:
        query_vacancies = []
        for area_id in area_ids:
            combo_idx += 1
            _write_progress_pct(5 + int(combo_idx / total_combos * 5), f"Поиск: {query[:40]} (регион {area_id})")
            logger.info("search_started", query=query, area_id=area_id)

            match hh_api.search_vacancies(
                text=query, area=area_id, period_days=period_days, max_pages=1, per_page=100,
                industry=industry, professional_role=professional_role,
            ):
                case Ok(_):
                    last_resp = getattr(hh_api, "last_response", None)
                    total_found = last_resp.get("found", 0) if last_resp else 0
                case Err(e):
                    logger.warning("search_estimate_failed", query=query, area=area_id, error=str(e))
                    total_found = 0

            if total_found <= chunk_threshold or period_days <= date_chunk_days:
                match hh_api.search_vacancies(
                    text=query, area=area_id, period_days=period_days, max_pages=max_pages,
                    per_page=100, industry=industry, professional_role=professional_role,
                ):
                    case Ok(vacs):
                        for vac in vacs:
                            vid = vac.get("id")
                            if vid and vid not in seen_ids:
                                seen_ids.add(vid)
                                query_vacancies.append(vac)
                                if len(query_vacancies) >= max_vacancies_per_query:
                                    break
                    case Err(e):
                        logger.warning("search_failed", query=query, area=area_id, error=str(e))
                if len(query_vacancies) >= max_vacancies_per_query:
                    break
            else:
                chunks = date_chunks(period_days, date_chunk_days)
                logger.info("period_chunked", chunks=len(chunks), total_days=period_days)
                for ci, (date_from, date_to) in enumerate(chunks):
                    _write_progress_pct(
                        5 + int(combo_idx / total_combos * 5),
                        f"Поиск {query[:30]}... интервал {ci + 1}/{len(chunks)} ({date_from}..{date_to})",
                    )
                    match hh_api.search_vacancies(
                        text=query, area=area_id, date_from=date_from, date_to=date_to,
                        max_pages=max_pages, per_page=100, industry=industry,
                        professional_role=professional_role,
                    ):
                        case Ok(vacs):
                            for vac in vacs:
                                vid = vac.get("id")
                                if vid and vid not in seen_ids:
                                    seen_ids.add(vid)
                                    query_vacancies.append(vac)
                                    if len(query_vacancies) >= max_vacancies_per_query:
                                        break
                        case Err(e):
                            logger.warning("search_chunk_failed", query=query, area=area_id, error=str(e))
                    if len(query_vacancies) >= max_vacancies_per_query:
                        break
                    time.sleep(config.REQUEST_DELAY)

            time.sleep(config.REQUEST_DELAY)

        all_vacancies.extend(query_vacancies[:max_vacancies_per_query])
        _write_progress_pct(
            5 + int(combo_idx / total_combos * 5),
            f"Найдено {len(all_vacancies)} вакансий ({query[:30]})",
        )
        logger.info("query_completed", query=query, vacancies=len(query_vacancies[:max_vacancies_per_query]))

    logger.info("collection_completed", total_unique=len(all_vacancies))
    return all_vacancies


def load_queries_from_file(filepath: Path) -> Result[list[str], DomainError]:
    """Загружает список запросов из текстового файла."""
    try:
        with open(filepath, encoding="utf-8") as f:
            return Ok([line.strip() for line in f if line.strip()])
    except Exception as e:
        return Err(DomainError(message="Queries file read error", detail=str(e)))


# ----------------------------------------------------------------------
# Инкрементальный парсинг (Incremental Parsing, #11)
# ----------------------------------------------------------------------


def get_last_parsed_id() -> Result[int, DomainError]:
    id_file = config.DATA_PROCESSED_DIR / "last_parsed_id.txt"
    if not id_file.exists():
        return Err(DomainError(message="Last parsed ID file not found"))
    try:
        raw = id_file.read_text(encoding="utf-8").strip()
        return Ok(int(raw)) if raw else Err(DomainError(message="Empty last parsed ID file"))
    except Exception as e:
        return Err(DomainError(message="Get last parsed ID error", detail=str(e)))


def save_last_parsed_id(vacancy_id: int) -> Result[None, DomainError]:
    id_file = config.DATA_PROCESSED_DIR / "last_parsed_id.txt"
    try:
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text(str(vacancy_id), encoding="utf-8")
        return Ok(None)
    except Exception as e:
        return Err(DomainError(message="Save last parsed ID error", detail=str(e)))


# ----------------------------------------------------------------------
# Checkpoint / Resume (#12)
# ----------------------------------------------------------------------


@dataclass
class ParsingCheckpoint:
    queries_done: list[str]
    total_collected: int
    errors: int
    elapsed_seconds: float
    timestamp: str


def save_checkpoint(checkpoint: ParsingCheckpoint) -> Result[None, DomainError]:
    path = config.DATA_CACHE_DIR / "parsing_checkpoint.json"
    try:
        atomic_write_json({
            "queries_done": checkpoint.queries_done,
            "total_collected": checkpoint.total_collected,
            "errors": checkpoint.errors,
            "elapsed_seconds": checkpoint.elapsed_seconds,
            "timestamp": checkpoint.timestamp,
        }, path)
        return Ok(None)
    except Exception as e:
        return Err(DomainError(message="Checkpoint save error", detail=str(e)))


def load_checkpoint() -> Result[ParsingCheckpoint, DomainError]:
    path = config.DATA_CACHE_DIR / "parsing_checkpoint.json"
    try:
        data = atomic_read_json(path)
        if data is None:
            return Err(DomainError(message="Checkpoint not found"))
        return Ok(ParsingCheckpoint(
            queries_done=data["queries_done"],
            total_collected=data["total_collected"],
            errors=data["errors"],
            elapsed_seconds=data["elapsed_seconds"],
            timestamp=data["timestamp"],
        ))
    except Exception as e:
        return Err(DomainError(message="Checkpoint load error", detail=str(e)))


def resume_from_checkpoint(queries: list[str]) -> Result[tuple[list[str], ParsingCheckpoint], DomainError]:
    match load_checkpoint():
        case Ok(checkpoint):
            remaining = [q for q in queries if q not in checkpoint.queries_done]
            logger.info(
                "checkpoint_resume",
                total_queries=len(queries),
                done=len(checkpoint.queries_done),
                remaining=len(remaining),
            )
            return Ok((remaining, checkpoint))
        case Err(e):
            return Err(e)


# ----------------------------------------------------------------------
# Интерактивный режим (используется в test_parsers.py и main.py)
# ----------------------------------------------------------------------


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(re.sub(r"[^\x00-\x7F]+", "", text))


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
    return ans in ("y", "yes", "да")


def select_from_list(items: list[str], prompt: str) -> str:
    print(prompt)
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    while True:
        try:
            idx = int(input("> ").strip())
            if 1 <= idx <= len(items):
                return items[idx - 1]
        except Exception:
            print("Некорректный ввод")


def interactive_config() -> dict[str, Any]:
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
        "11. Поиск по всему IT-сектору (industry=7)",
    ]

    selected_mode = select_from_list(mode_options, "\nВыберите вариант поиска:")

    if selected_mode == "11. Поиск по всему IT-сектору (industry=7)":
        print("\nРежим: Поиск по всему IT-сектору")
        positions = [
            "Data Scientist",
            "Data Analyst",
            "Machine Learning Engineer",
            "Computer Vision Engineer",
            "NLP Engineer",
            "Data Architect",
            "ETL Developer",
            "Python Developer",
            "Java Developer",
            "Frontend Developer",
            "Backend Developer",
            "Fullstack Developer",
            "DevOps Engineer",
            "Embedded Developer",
            "Blockchain Developer",
            "iOS Developer",
            "Android Developer",
            "React Native Developer",
            "Flutter Developer",
            "QA Engineer",
            "Automation QA Engineer",
            "Performance QA Engineer",
            "Специалист по кибербезопасности",
            "Security Engineer",
            "DevSecOps Engineer",
            "SRE инженер",
            "Системный администратор",
            "Облачный инженер",
            "Сетевой инженер",
            "Администратор баз данных",
            "Системный аналитик",
            "Бизнес-аналитик",
            "Архитектор программного обеспечения",
            "Solution Architect",
            "Team Lead",
            "Tech Lead",
            "Project Manager IT",
            "Scrum Master",
            "UX/UI дизайнер",
            "Product Designer",
            "Unity Developer",
            "Unreal Engine Developer",
            "Technical Writer",
            "MLops engineer",
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
        query = selected_mode.split(". ", 1)[1]
        industry = None
        is_it_sector = False
        queries = [query]

    region_options = [
        ("Москва", 1),
        ("Санкт-Петербург", 2),
        ("Екатеринбург", 3),
        ("Новосибирск", 4),
        ("Казань", 88),
        ("Нижний Новгород", 66),
        ("Ростов-на-Дону", 76),
        ("Вся Россия", 0),
    ]
    region_names = [f"{name} (ID {rid})" for name, rid in region_options]
    print("\nВыберите регионы (можно несколько, введите номера через пробел):")
    try:
        indices = list(map(int, input("> ").split()))
        selected_regions = [region_names[i - 1] for i in indices if 1 <= i <= len(region_names)]
    except (ValueError, IndexError):
        selected_regions = [region_names[0]]

    area_ids = [int(re.search(r"ID (\d+)", s).group(1)) for s in selected_regions if re.search(r"ID (\d+)", s)]
    if not area_ids:
        area_ids = [1]

    if is_it_sector:
        period = 30
        max_pages = 50
        skip_details = False
        show_list = False
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
        "max_vacancies_per_query": max_vacancies,
    }


# ----------------------------------------------------------------------
# Обработка навыков (извлечение, подсчёт, маппинг на компетенции)
# ----------------------------------------------------------------------


def normalize_skill_for_matching(skill: str) -> str:
    """Нормализует навык для сопоставления с маппингом."""
    normalized = skill.lower().strip()
    normalized = re.sub(r"[^\w\s-]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def extract_and_count_skills(vacancies: list[dict[str, Any]], parser: VacancyParser) -> Result[dict[str, Any], DomainError]:
    logger.info("extracting_skills_from_vacancies", count=len(vacancies))
    if not vacancies:
        return Ok({"frequencies": {}, "tfidf_weights": {}})
    return parser.extract_skills_from_vacancies(vacancies)


def map_to_competencies(skill_frequencies: dict[str, int], mapping: dict[str, list[str]]) -> Counter:
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
            found = False
            for keyword in skill_to_comp:
                if keyword in normalized_skill or normalized_skill in keyword:
                    matched_skills += 1
                    for comp in skill_to_comp[keyword]:
                        comp_counter[comp] += freq
                    found = True
                    break
            if not found and len(unmatched_skills) < 10:
                unmatched_skills.append(skill)

    logger.info(
        "skills_mapped_to_competencies",
        matched=matched_skills,
        total=len(skill_frequencies),
    )
    if unmatched_skills:
        logger.info("unmatched_skills_sample", sample=unmatched_skills[:5])
    return comp_counter


def print_top_skills(skill_frequencies: dict[str, int], top_n: int = 20) -> None:
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


def date_chunks(days: int, chunk_size: int = 5) -> list[tuple[str, str]]:
    """Разбивает период поиска на интервалы. Возвращает (date_from, date_to) в YYYY-MM-DD."""
    end_date = datetime.now()
    chunks = []
    for offset in range(0, days, chunk_size):
        to_date = end_date - timedelta(days=offset)
        from_date = to_date - timedelta(days=min(chunk_size, days - offset))
        chunks.append((from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d")))
    return chunks
