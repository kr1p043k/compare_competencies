import json
import logging
import collections
from pathlib import Path
from typing import Any
from typing import Dict, Set
from src import config

def setup_logging():
    """
    Настраивает логирование: вывод в консоль (INFO) и в файл (DEBUG).
    Файл лога: logs/app.log.
    """
    # Создаём логгер
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Формат лога: время - уровень - имя модуля - сообщение
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Хендлер для файла (записывает всё от DEBUG и выше)
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Хендлер для консоли (только INFO и выше, чтобы не засорять вывод)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Добавляем хендлеры в корневой логгер
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def read_json(filepath: Path) -> Any:
    """Безопасно читает JSON-файл с логированием."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Чтение JSON из {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data: Any, filepath: Path):
    """Безопасно записывает данные в JSON-файл с логированием."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Запись JSON в {filepath}")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# ===== НОВЫЕ ФУНКЦИИ ДЛЯ ВАЛИДАЦИИ НАВЫКОВ =====

def load_it_skills() -> Set[str]:
    """
    Загружает список допустимых IT-навыков из файла data/it_skills.json.
    Возвращает множество строк в нижнем регистре (для быстрого поиска).
    Если файл не найден или повреждён, возвращает пустое множество и логирует предупреждение.
    """
    logger = logging.getLogger(__name__)
    if not config.IT_SKILLS_FILE.exists():
        logger.warning(f"Файл с IT-навыками не найден: {config.IT_SKILLS_FILE}. Фильтрация отключена.")
        return set()

    try:
        skills_list = read_json(config.IT_SKILLS_FILE)
        if not isinstance(skills_list, list):
            logger.error("Файл it_skills.json должен содержать список строк.")
            return set()
        # Приводим всё к нижнему регистру и удаляем лишние пробелы
        skills_set = {skill.strip().lower() for skill in skills_list if isinstance(skill, str)}
        logger.info(f"Загружено {len(skills_set)} допустимых IT-навыков.")
        return skills_set
    except Exception as e:
        logger.error(f"Ошибка при загрузке it_skills.json: {e}")
        return set()

def filter_skills_by_whitelist(skills_dict: Dict[str, int], whitelist: Set[str]) -> Dict[str, int]:
    """
    Оставляет в словаре только те навыки, которые присутствуют в whitelist (множество строк в нижнем регистре).
    Возвращает новый словарь.
    """
    if not whitelist:
        # Если белый список пуст, возвращаем исходный словарь (фильтрация отключена)
        return skills_dict.copy()

    filtered = {}
    for skill, count in skills_dict.items():
        skill_lower = skill.lower().strip()
        if skill_lower in whitelist:
            filtered[skill] = count
        else:
            logging.getLogger(__name__).debug(f"Навык отфильтрован: '{skill}'")

    logging.getLogger(__name__).info(f"Фильтрация: осталось {len(filtered)} навыков из {len(skills_dict)}")
    return filtered