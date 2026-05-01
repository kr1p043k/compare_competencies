#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт авторасширения it_skills.json из новых вакансий.
Извлекает навыки из файла вакансий, нормализует,
сравнивает с текущим белым списком и добавляет отсутствующие.
Поддерживает интерактивный и автоматический режимы.
"""
import re
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Set, List

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsing.vacancy_parser import VacancyParser
from src.parsing.skill_normalizer import SkillNormalizer
from src.parsing.skill_validator import SkillValidator
from src.parsing.utils import load_it_skills, read_json
from src import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_all_skills(vacancies: List[dict]) -> Set[str]:
    """
    Извлекает все уникальные нормализованные навыки из списка вакансий.
    Использует усиленную фильтрацию через SkillValidator + дополнительные проверки.
    """
    parser = VacancyParser()
    validator = SkillValidator(whitelist=None)
    
    # Дополнительные фильтры
    MIN_LENGTH = 3
    MAX_LENGTH = 50
    MAX_WORDS = 4
    
    # Ключевые слова-паразиты (если навык состоит ТОЛЬКО из них или начинается с них)
    STOP_WORDS = {
        'опыт', 'знание', 'работа', 'лет', 'навык', 'работы', 'знания',
        'использования', 'проведения', 'поддержка', 'развитие', 'создание',
        'внедрение', 'участие', 'обеспечение', 'управление', 'анализ',
        'разработка', 'программирование', 'тестирование', 'построение',
        'организация', 'контроль', 'планирование', 'оценка', 'учёт',
        'хорошее', 'отличное', 'базовое', 'уверенное', 'понимание',
        'умение', 'владение', 'наличие', 'отсутствие', 'способность',
        'обязательно', 'желательно', 'приветствуется', 'используется'
    }
    
    # Шаблоны мусора (регулярные выражения)
    BAD_PATTERNS = [
        r'^[\.\-\d]',                    # начинается с точки, дефиса или цифры
        r'^.{1,2}$',                     # 1-2 символа
        r'^[a-z]$',                      # одиночная буква
        r'["\'«»„"]',                    # содержит кавычки
        r'stro$',                        # обрывок "stro"
        r'quot$',                        # обрывок "quot"
        r'^.{50,}$',                     # длиннее 50 символов
    ]
    
    all_skills = set()
    
    for vac in vacancies:
        skills = []
        
        # Из key_skills
        key_skills = vac.get('key_skills', [])
        if key_skills:
            skills.extend([s['name'] for s in key_skills if isinstance(s, dict) and 'name' in s])
        
        # Из текста
        desc = vac.get('description', '') or ''
        snippet = vac.get('snippet', {}) or {}
        req = snippet.get('requirement', '') or ''
        resp = snippet.get('responsibility', '') or ''
        text_skills = parser.extract_skills_from_description(f"{desc} {req} {resp}")
        skills.extend(text_skills)
        
        for skill in skills:
            norm = SkillNormalizer.normalize(skill)
            if not norm:
                continue
            
            # 1. Базовые проверки
            if len(norm) < MIN_LENGTH or len(norm) > MAX_LENGTH:
                continue
            
            words = norm.split()
            if len(words) > MAX_WORDS:
                continue
            
            # 2. Проверка через SkillValidator (blacklist, generic и т.д.)
            if not validator.validate(norm).is_valid:
                continue
            
            # 3. Проверка на стоп-слова (если навык — это только стоп-слово)
            if all(w in STOP_WORDS for w in words):
                continue
            
            # 4. Проверка через регулярки
            if any(re.search(pattern, norm) for pattern in BAD_PATTERNS):
                continue
            
            # 5. Дополнительная проверка: если навык слишком общий
            if len(words) == 1 and len(norm) < 4 and norm not in {'r', 'c', 'go', 'sql'}:
                continue
            
            # 6. Не начинается с цифры
            if norm[0].isdigit():
                continue
            
            all_skills.add(norm)
    
    # 7. Пост-фильтрация: убираем дубликаты с разным регистром
    result = set()
    seen_lower = set()
    for skill in sorted(all_skills):
        skill_lower = skill.lower()
        if skill_lower not in seen_lower:
            seen_lower.add(skill_lower)
            result.add(skill)
    
    return result


def extend_it_skills(
    vacancies_path: Path,
    output_path: Path,
    interactive: bool = True,
    min_frequency: int = 1
) -> int:
    """
    Основная функция: загружает вакансии, находит новые навыки,
    предлагает добавить их в it_skills.json.
    
    Returns:
        Количество добавленных навыков
    """
    # Загружаем текущий белый список
    current_skills = load_it_skills()
    if not current_skills:
        logger.error("Не удалось загрузить текущий it_skills.json")
        return 0
    
    logger.info(f"Текущий белый список содержит {len(current_skills)} навыков")
    
    # Загружаем вакансии
    if not vacancies_path.exists():
        logger.error(f"Файл вакансий не найден: {vacancies_path}")
        logger.info("Доступные файлы:")
        for p in config.DATA_RAW_DIR.glob("*.json"):
            logger.info(f"  • {p}")
        for p in config.DATA_RESULT_DIR.glob("*.json"):
            logger.info(f"  • {p}")
        return 0
    
    vacancies = read_json(vacancies_path)
    if not vacancies:
        logger.error(f"Не удалось загрузить вакансии из {vacancies_path}")
        return 0
    
    logger.info(f"Загружено {len(vacancies)} вакансий")
    
    # Извлекаем все навыки из вакансий
    logger.info("Извлечение навыков из вакансий...")
    extracted_skills = extract_all_skills(vacancies)
    logger.info(f"Извлечено {len(extracted_skills)} уникальных валидных навыков")
    
    # Находим новые навыки (отсутствующие в белом списке)
    new_skills = extracted_skills - current_skills
    if not new_skills:
        logger.info("✅ Новых навыков не найдено – белый список актуален")
        return 0
    
    logger.info(f"Найдено {len(new_skills)} новых навыков:")
    for i, skill in enumerate(sorted(new_skills), 1):
        logger.info(f"  {i:3d}. {skill}")
    
    if interactive:
        print(f"\nНайдено {len(new_skills)} новых навыков.")
        print("Добавить их в it_skills.json?")
        print("  y = добавить все")
        print("  n = не добавлять")
        print("  m = показать ещё раз и выбрать вручную (y/n для каждого)")
        choice = input("Ваш выбор [y/n/m]: ").strip().lower()
        
        if choice == 'n':
            logger.info("Операция отменена пользователем")
            return 0
        elif choice == 'm':
            approved = set()
            for skill in sorted(new_skills):
                ans = input(f"  Добавить '{skill}'? [y/n]: ").strip().lower()
                if ans == 'y':
                    approved.add(skill)
            new_skills = approved
            if not new_skills:
                logger.info("Ни один навык не выбран")
                return 0
        # при 'y' – добавляем все
    
    # Добавляем новые навыки в белый список
    updated_skills = sorted(current_skills | new_skills)
    
    # Сохраняем
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_skills, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Белый список обновлён и сохранён в {output_path}")
    logger.info(f"   Было: {len(current_skills)} навыков")
    logger.info(f"   Добавлено: {len(new_skills)} навыков")
    logger.info(f"   Стало: {len(updated_skills)} навыков")
    
    return len(new_skills)


def main():
    parser = argparse.ArgumentParser(description="Авторасширение it_skills.json из вакансий")
    parser.add_argument(
        '--vacancies', '-v',
        type=Path,
        default=config.DATA_RESULT_DIR / "hh_vacancies_detailed.json",
        help="Путь к JSON-файлу с детальными вакансиями (по умолчанию data/result/hh_vacancies_detailed.json)"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=config.IT_SKILLS_PATH,
        help="Путь к выходному файлу (по умолчанию data/it_skills.json)"
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help="Автоматически добавить все новые навыки (без интерактивного подтверждения)"
    )
    parser.add_argument(
        '--backup', '-b',
        action='store_true',
        default=True,
        help="Создать резервную копию текущего it_skills.json перед изменением (по умолчанию да)"
    )
    
    args = parser.parse_args()
    
    # Проверяем существование файла вакансий
    if not args.vacancies.exists():
        logger.error(f"Файл вакансий не найден: {args.vacancies}")
        logger.info("Попробуйте указать другой файл через --vacancies, например:")
        logger.info("  python scripts/extend_it_skills.py --vacancies data/result/hh_vacancies_detailed.json")
        logger.info("Или сначала выполните сбор вакансий с флагом --skip-collection")
        sys.exit(1)
    
    # Создаём бэкап текущего белого списка
    if args.backup and args.output.exists():
        backup_path = args.output.with_suffix('.backup.json')
        import shutil
        shutil.copy(args.output, backup_path)
        logger.info(f"Резервная копия сохранена: {backup_path}")
    
    # Запускаем расширение
    added = extend_it_skills(
        vacancies_path=args.vacancies,
        output_path=args.output,
        interactive=not args.yes
    )
    
    if added > 0:
        print(f"\n✅ Добавлено {added} новых навыков.")
        print("⚠️  Не забудьте очистить кэш перед следующим запуском:")
        print("   rm data/processed/parsed_skills.pkl")
        print("   rm -r data/embeddings/cache/")
    else:
        print("\nБелый список уже актуален.")


if __name__ == "__main__":
    main()