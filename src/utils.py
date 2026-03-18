import logging
import json
from src.config import *

def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Возвращает логгер с указанным именем, который пишет в LOG_FILE
    и выводит в консоль сообщения уровня INFO и выше.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Файловый обработчик
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def load_competency_mapping():
    with open(COMPETENCY_MAPPING_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)