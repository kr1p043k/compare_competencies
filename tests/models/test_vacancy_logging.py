"""
Проверка: debug у src.models.vacancy (vacancy_parsed_from_api и
vacancy_created на каждую вакансию) приглушён до INFO.

Прогрев после каждого деплоя парсит десятки тысяч вакансий — без приглушения
это ~90k debug-строк и ~50МБ лога за один запуск. Уровень выставляется на
stdlib-логгере: он действует независимо от того, успел ли setup_structlog()
перенастроить structlog-прокси.

Warning'и об ошибках парсинга (invalid_skill_in_vacancy) и
vacancy_parsing_failed (error) остаются слышны — приглушается только debug.
"""

import logging

import src.models.vacancy  # noqa: F401  # импорт и есть проверка сетАпа


def test_vacancy_debug_is_muted():
    logger = logging.getLogger("src.models.vacancy")

    assert logger.getEffectiveLevel() >= logging.INFO, (
        f"уровень должен быть >= INFO (debug приглушён), получили {logger.getEffectiveLevel()}"
    )
    assert not logger.isEnabledFor(logging.DEBUG), (
        "debug-события src.models.vacancy не должны попадать в лог прод-прогрева"
    )
