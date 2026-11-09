"""
Проверка: debug-шум skill_normalizer (fuzzy_match/no_fuzzy_match при прогреве
после каждого деплоя) заглушен — уровень stdlib-логгера модуля поднят до INFO,
глобальные настройки не тронуты.

Отдельно важен сам факт импорта: уровень ставится через logging.getLogger(...),
а не через structlog-прокси — у прокси нет setLevel, пока setup_structlog() не
выполнен (модуль импортируется раньше старта приложения).
"""

import logging

from src.parsing.skills.skill_normalizer import SkillNormalizer


def test_skill_normalizer_debug_is_muted():
    # Сбрасываем кэши до прогрева, чтобы прошла ветка fuzzy/no_fuzzy.
    SkillNormalizer._canonical_map = None
    SkillNormalizer._whitelist = None

    logger = logging.getLogger("src.parsing.skills.skill_normalizer")

    assert logger.getEffectiveLevel() >= logging.INFO, (
        f"ожидали эффективный уровень >= INFO (debug заглушен), получили {logger.getEffectiveLevel()}"
    )
    assert not logger.isEnabledFor(logging.DEBUG), (
        "debug-события skill_normalizer должны отбрасываться на уровне логгера"
    )
