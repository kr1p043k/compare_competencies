"""CLI dispatcher: python -m src.cli <command> [args]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main() -> None:
    import argparse
    import asyncio

    from src.cli import (
        backup_db,
        compute_competency_trends,
        compute_competency_vectors,
        create_user,
        embeddings,
        export_json,
        export_results,
        export_vacancies,
        extend_skills,
        import_students,
        rebuild,
        seed_db,
        teacher_analysis,
    )

    parser = argparse.ArgumentParser(description="compare_competencies CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("seed-db", help="Наполнить БД из JSON")
    p.add_argument("--drop", action="store_true")
    p.add_argument("--version", help="Parse version label")
    p.set_defaults(func=lambda a: seed_db.main(drop=a.drop, version=a.version))

    p = sub.add_parser("create-user", help="Создать пользователя")
    p.add_argument("email")
    p.add_argument("password")
    p.add_argument("--role", default="teacher", choices=["admin", "teacher"])
    p.add_argument("--name", default="")
    p.set_defaults(func=lambda a: create_user.main(a.email, a.password, a.role, a.name))

    p = sub.add_parser("embeddings", help="Сгенерировать эмбеддинги навыков")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=lambda a: embeddings.main(force=a.force))

    p = sub.add_parser("import-students", help="Импорт студентов из CSV")
    p.add_argument("csv", help="Path to CSV file")
    p.set_defaults(func=lambda a: import_students.main(a.csv))

    p = sub.add_parser("export-json", help="Экспорт БД в JSON")
    p.set_defaults(func=lambda a: export_json.main())

    p = sub.add_parser("extend-skills", help="Расширить it_skills таксономию")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--coverage", action="store_true")
    p.add_argument("--dead", action="store_true")
    p.add_argument("--min-frequency", type=int, default=2)
    p.set_defaults(func=lambda a: extend_skills.main(a))

    p = sub.add_parser("rebuild", help="Полная пересборка")
    p.set_defaults(func=lambda a: rebuild.main())

    p = sub.add_parser("backup", help="Бэкап БД")
    p.add_argument("--restore", nargs="?", const="latest", help="Восстановить из бэкапа")
    p.set_defaults(func=lambda a: backup_db.main(restore=a.restore))

    p = sub.add_parser("export-results", help="Экспорт JSON-результатов в БД")
    p.set_defaults(func=lambda a: export_results.main())

    p = sub.add_parser("compute-competency-vectors", help="Вычислить эмбеддинги компетенций через mean pool навыков")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=lambda a: compute_competency_vectors.main(force=a.force))

    p = sub.add_parser("compute-competency-trends", help="Вычислить тренды компетенций из снимков рынка")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=lambda a: asyncio.run(compute_competency_trends.main(force=a.force)))

    p = sub.add_parser("teacher-analysis", help="Запустить преподавательский анализ (gap + embedding + SHAP)")
    p.add_argument("--direction", default="09.03.02", help="Код направления (09.03.02)")
    p.add_argument("--discipline", help="Фильтр по дисциплине (необязательно)")
    p.set_defaults(func=lambda a: teacher_analysis.main(direction=a.direction, discipline=a.discipline))

    p = sub.add_parser("export-vacancies", help="Экспорт JSON-вакансий в БД")
    p.add_argument("--basic", help="Путь к hh_vacancies_basic.json")
    p.add_argument("--detailed", help="Путь к hh_vacancies_detailed.json")
    p.set_defaults(func=lambda a: export_vacancies.main(basic_path=a.basic, detailed_path=a.detailed))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
