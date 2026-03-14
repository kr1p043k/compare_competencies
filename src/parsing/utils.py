import json
import logging
from pathlib import Path
from typing import Any
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