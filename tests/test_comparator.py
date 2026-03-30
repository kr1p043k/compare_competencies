import pytest
import logging
import traceback
import sys

# Настраиваем логирование сразу при запуске теста
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)
logger = logging.getLogger("test_comparator")

from src.analyzers.comparator import CompetencyComparator


@pytest.mark.parametrize("use_embeddings,level", [
    (False, "middle"),
    (True, "junior"),
    (True, "middle"),
    (True, "senior")
])
def test_comparator_both_modes(use_embeddings, level):
    """Тест обоих режимов с подробным логированием и отловом ошибок"""
    logger.info("=" * 80)
    logger.info(f"▶️  Запуск теста | embeddings={use_embeddings} | level={level}")
    logger.info("=" * 80)

    try:
        # 1. Создаём comparator
        logger.info("Создаём CompetencyComparator...")
        comparator = CompetencyComparator(
            use_embeddings=use_embeddings,
            level=level
        )
        logger.info("✅ Comparator создан")

        # 2. Пример данных
        vacancies_skills = [
            ["python", "sql", "pandas", "machine learning"],
            ["python", "pytorch", "ml", "rest api"],
            ["sql", "django", "fastapi"]
        ]
        student_skills = ["python", "sql", "pytorch", "ml"]

        # 3. fit_market
        logger.info("Вызываем fit_market()...")
        success = comparator.fit_market(vacancies_skills)
        assert success is True, "fit_market вернул False"
        logger.info("✅ fit_market прошёл успешно")

        # 4. compare
        logger.info("Вызываем compare()...")
        score, confidence = comparator.compare(student_skills)
        logger.info(f"Результат: score={score:.4f} | confidence={confidence:.4f}")

        # 5. Проверки
        assert 0.0 <= score <= 1.0, f"Некорректный score: {score}"
        assert 0.0 <= confidence <= 1.0, f"Некорректный confidence: {confidence}"
        assert score > 0.3, f"Слишком низкий score: {score} (ожидали >0.3)"

        # 6. Статистика
        stats = comparator.get_stats()
        logger.info(f"Статистика: {stats}")

        logger.info(f"✅ ТЕСТ ПРОШЁЛ УСПЕШНО для {level} (embeddings={use_embeddings})")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("❌ ОШИБКА В ТЕСТЕ")
        logger.error(f"Тип: {type(e).__name__}")
        logger.error(f"Сообщение: {e}")
        logger.error("-" * 60)
        traceback.print_exc(file=sys.stderr)
        logger.error("-" * 60)
        raise  # чтобы pytest показал полный traceback