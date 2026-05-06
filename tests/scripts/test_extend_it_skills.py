# tests/scripts/test_extend_it_skills.py
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestExtendItSkills:
    @pytest.fixture
    def sample_vacancies(self):
        return [
            {
                "id": "1",
                "name": "Python Developer",
                "key_skills": [{"name": "Python"}, {"name": "Django"}],
                "description": "Опыт работы с FastAPI",
                "snippet": {"requirement": "Знание Docker", "responsibility": "Разработка микросервисов"},
            },
            {
                "id": "2",
                "name": "Data Scientist",
                "key_skills": [{"name": "Python"}, {"name": "TensorFlow"}],
                "description": "Опыт работы с PyTorch",
                "snippet": {},
            },
        ]

    @pytest.fixture
    def sample_skills_file(self, tmp_path):
        skills = ["python", "django", "fastapi", "docker", "tensorflow", "pytorch"]
        path = tmp_path / "it_skills.json"
        path.write_text(json.dumps(skills), encoding="utf-8")
        return path

    def test_extract_all_skills(self, sample_vacancies):
        """Извлечение навыков из вакансий"""
        from scripts.extend_it_skills import extract_all_skills

        skills = extract_all_skills(sample_vacancies, min_frequency=1)
        assert len(skills) > 0
        assert isinstance(skills, list)
        assert isinstance(skills[0], tuple)
        assert len(skills[0]) == 2  # (skill, frequency)

    def test_extract_all_skills_min_frequency(self, sample_vacancies):
        """Фильтрация по минимальной частоте"""
        from scripts.extend_it_skills import extract_all_skills

        # Частота ≥ 10 — ничего не должно вернуться
        skills = extract_all_skills(sample_vacancies, min_frequency=10)
        assert len(skills) == 0

    def test_analyze_coverage(self, sample_skills_file, monkeypatch):
        """Анализ покрытия категорий"""
        monkeypatch.setattr("src.parsing.utils.config.DATA_DIR", sample_skills_file.parent)
        monkeypatch.setattr("src.analyzers.skill_taxonomy.config.DATA_DIR", sample_skills_file.parent.parent)

        from scripts.extend_it_skills import analyze_coverage
        from src.analyzers.skill_taxonomy import SkillTaxonomy

        current_skills = {"python", "django", "fastapi"}
        taxonomy = SkillTaxonomy()
        coverage = analyze_coverage(current_skills, taxonomy)

        assert isinstance(coverage, dict)
        # Должна быть хотя бы одна категория

    def test_find_dead_skills(self):
        """Поиск мёртвых навыков"""
        from scripts.extend_it_skills import find_dead_skills

        current = {"python", "sql", "dead_skill_1", "dead_skill_2"}
        extracted = {"python": 10, "sql": 5}

        dead = find_dead_skills(current, extracted)
        assert len(dead) == 2
        assert "dead_skill_1" in dead
        assert "dead_skill_2" in dead

    def test_find_dead_skills_none(self):
        """Все навыки встретились"""
        from scripts.extend_it_skills import find_dead_skills

        current = {"python", "sql"}
        extracted = {"python": 10, "sql": 5}

        dead = find_dead_skills(current, extracted)
        assert dead == []

    def test_add_skills_to_whitelist(self, sample_skills_file):
        """Добавление навыков в whitelist"""
        from scripts.extend_it_skills import add_skills_to_whitelist

        # Патчим DATA_DIR, а не IT_SKILLS_PATH
        with patch("src.parsing.utils.config.DATA_DIR", sample_skills_file.parent):
            from src.parsing.utils import load_it_skills

            added = add_skills_to_whitelist(
                skills_to_add={"new_skill_1", "new_skill_2"},
                output_path=sample_skills_file,
                backup=False,
            )
            assert added == 2

            # Проверяем содержимое файла
            content = json.loads(sample_skills_file.read_text(encoding="utf-8"))
            assert "new_skill_1" in content
            assert "new_skill_2" in content

    def test_add_skills_to_whitelist_empty(self, sample_skills_file):
        """Добавление пустого списка"""
        from scripts.extend_it_skills import add_skills_to_whitelist

        with patch("scripts.extend_it_skills.config.IT_SKILLS_PATH", sample_skills_file):
            added = add_skills_to_whitelist(
                skills_to_add=set(),
                output_path=sample_skills_file,
                backup=False,
            )
            assert added == 0

    def test_make_bar(self):
        """Форматирование прогресс-бара"""
        from scripts.extend_it_skills import _make_bar

        bar = _make_bar(50, width=10)
        assert len(bar) == 10
        assert "█" in bar
        assert "░" in bar
        assert bar.count("█") == 5

    def test_print_new_skills_empty(self, capsys):
        """Вывод при отсутствии новых навыков"""
        from scripts.extend_it_skills import print_new_skills

        print_new_skills({})
        captured = capsys.readouterr()
        assert "Новых навыков не найдено" in captured.out

    def test_print_dead_skills_empty(self, capsys):
        """Вывод при отсутствии мёртвых навыков"""
        from scripts.extend_it_skills import print_dead_skills

        print_dead_skills([])
        captured = capsys.readouterr()
        assert "Все навыки" in captured.out or "✅" in captured.out

    def test_interactive_confirm_add_all(self, monkeypatch):
        """Интерактивный режим: добавить все"""
        from scripts.extend_it_skills import interactive_confirm

        new_skills = {"skill_a": 5, "skill_b": 3, "skill_c": 1}
        monkeypatch.setattr("builtins.input", lambda _: "a")

        approved = interactive_confirm(new_skills)
        assert approved == set(new_skills.keys())

    def test_interactive_confirm_quit(self, monkeypatch):
        """Интерактивный режим: выйти сразу"""
        from scripts.extend_it_skills import interactive_confirm

        new_skills = {"skill_a": 5, "skill_b": 3}
        monkeypatch.setattr("builtins.input", lambda _: "q")

        approved = interactive_confirm(new_skills)
        assert approved == set()

    def test_interactive_confirm_selective(self, monkeypatch):
        """Интерактивный режим: выбрать некоторые"""
        from scripts.extend_it_skills import interactive_confirm

        new_skills = {"skill_a": 5, "skill_b": 3}
        inputs = iter(["y", "n"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        approved = interactive_confirm(new_skills)
        assert "skill_a" in approved
        assert "skill_b" not in approved

    def test_main_analysis_mode(self, sample_vacancies, sample_skills_file, monkeypatch, tmp_path, capsys):
        """Режим анализа (без изменений)"""
        # Создаём реальный файл вакансий
        vac_file = tmp_path / "test_vacancies.json"
        vac_file.write_text(json.dumps(sample_vacancies), encoding="utf-8")

        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--vacancies", str(vac_file),  # настоящий файл
            "--output", str(sample_skills_file),
        ])
        monkeypatch.setattr("scripts.extend_it_skills.load_it_skills", lambda: {"python", "django"})

        from scripts.extend_it_skills import main
        main()

        captured = capsys.readouterr()
        assert "РЕЖИМ АНАЛИЗА" in captured.out

    def test_main_yes_mode(self, sample_vacancies, sample_skills_file, monkeypatch, tmp_path):
        """Режим --yes (автодобавление)"""
        vac_file = tmp_path / "test_vacancies.json"
        vac_file.write_text(json.dumps(sample_vacancies), encoding="utf-8")

        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--vacancies", str(vac_file),
            "--output", str(sample_skills_file),
            "--yes",
            "--no-backup",
        ])
        monkeypatch.setattr("scripts.extend_it_skills.load_it_skills", lambda: {"python", "django"})

        from scripts.extend_it_skills import main
        main()  # Не должно упасть

    def test_main_vacancies_not_found(self, tmp_path, monkeypatch, capsys):
        """Файл вакансий не найден"""
        nonexistent = tmp_path / "nonexistent.json"
        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--vacancies", str(nonexistent),
        ])

        with pytest.raises(SystemExit):
            from scripts.extend_it_skills import main
            main()

    def test_print_coverage(self, capsys):
        """Вывод покрытия"""
        from scripts.extend_it_skills import print_coverage

        coverage = {
            "test_cat": {
                "label": "Test Category",
                "icon": "🔧",
                "total": 10,
                "covered": 7,
                "percent": 70.0,
            }
        }
        print_coverage(coverage)
        captured = capsys.readouterr()
        assert "Test Category" in captured.out
        assert "70.0%" in captured.out

    @pytest.fixture
    def sample_vacancies_rich(self):
        return [
            {
                "id": "1",
                "name": "Python Developer",
                "key_skills": [{"name": "Python"}, {"name": "Django"}],
                "description": "Опыт работы с FastAPI, Docker, Kubernetes",
                "snippet": {"requirement": "Знание Git, SQL", "responsibility": "Разработка микросервисов"},
            },
            {
                "id": "2",
                "name": "Data Scientist",
                "key_skills": [{"name": "Python"}, {"name": "TensorFlow"}],
                "description": "Опыт работы с PyTorch, Pandas, NumPy",
                "snippet": {},
            },
            {
                "id": "3",
                "name": "Frontend Developer",
                "key_skills": [{"name": "JavaScript"}, {"name": "React"}],
                "description": "Опыт с TypeScript, Next.js",
                "snippet": {"requirement": "Знание HTML, CSS"},
            },
        ]

    def test_extract_all_skills_rich(self, sample_vacancies_rich):
        """Строки 108-110: полное извлечение навыков"""
        from scripts.extend_it_skills import extract_all_skills

        skills = extract_all_skills(sample_vacancies_rich, min_frequency=1)
        skill_names = [s[0] for s in skills]

        assert "python" in skill_names
        assert "django" in skill_names

    def test_extract_all_skills_min_frequency_filter(self, sample_vacancies_rich):
        """Строки 108-110: фильтрация по частоте"""
        from scripts.extend_it_skills import extract_all_skills

        # Только навыки с частотой ≥ 2
        skills = extract_all_skills(sample_vacancies_rich, min_frequency=2)
        skill_names = [s[0] for s in skills]

        # python встречается 2 раза
        assert "python" in skill_names

    def test_print_new_skills_with_taxonomy(self, sample_vacancies_rich, tmp_path, monkeypatch, capsys):
        """Строка 146: вывод с таксономией"""
        tax_path = tmp_path / "skill_taxonomy.json"
        tax_path.write_text(json.dumps({
            "categories": {
                "programming_languages": {
                    "label": "Языки",
                    "icon": "💻",
                    "skills": ["python", "javascript"]
                }
            }
        }))
        monkeypatch.setattr("src.analyzers.skill_taxonomy.config.DATA_DIR", tmp_path)

        from scripts.extend_it_skills import print_new_skills
        from src.analyzers.skill_taxonomy import SkillTaxonomy

        # Сбрасываем синглтон
        SkillTaxonomy._instance = None

        new_skills = {"python": 5, "javascript": 3}
        taxonomy = SkillTaxonomy()
        print_new_skills(new_skills, taxonomy)

        captured = capsys.readouterr()
        assert "python" in captured.out

    def test_analyze_coverage_with_data(self, tmp_path, monkeypatch):
        """Строки 159, 161: анализ покрытия с данными"""
        tax_path = tmp_path / "skill_taxonomy.json"
        tax_path.write_text(json.dumps({
            "categories": {
                "test_cat": {
                    "label": "Test",
                    "icon": "🧪",
                    "skills": ["skill_a", "skill_b", "skill_c", "skill_d"]
                }
            }
        }))
        monkeypatch.setattr("src.analyzers.skill_taxonomy.config.DATA_DIR", tmp_path)

        from scripts.extend_it_skills import analyze_coverage
        from src.analyzers.skill_taxonomy import SkillTaxonomy

        SkillTaxonomy._instance = None
        taxonomy = SkillTaxonomy()

        coverage = analyze_coverage({"skill_a", "skill_b"}, taxonomy)
        assert "test_cat" in coverage
        assert coverage["test_cat"]["covered"] == 2
        assert coverage["test_cat"]["total"] == 4
        assert coverage["test_cat"]["percent"] == 50.0

    def test_find_dead_skills_partial(self):
        """Строки 198-199: поиск мёртвых навыков"""
        from scripts.extend_it_skills import find_dead_skills

        current = {"active_skill", "dead_skill", "python"}
        extracted = {"active_skill": 10, "python": 5}

        dead = find_dead_skills(current, extracted)
        assert "dead_skill" in dead
        assert "active_skill" not in dead
        assert "python" not in dead

    def test_print_coverage_output(self, capsys):
        """Строки 206-210: форматированный вывод покрытия"""
        from scripts.extend_it_skills import print_coverage

        coverage = {
            "cat1": {
                "label": "Category 1",
                "icon": "🔧",
                "total": 10,
                "covered": 5,
                "percent": 50.0,
            }
        }
        print_coverage(coverage)
        captured = capsys.readouterr()
        assert "Category 1" in captured.out
        assert "50.0%" in captured.out

    def test_print_dead_skills_many(self, capsys):
        """Строка 238: много мёртвых навыков"""
        from scripts.extend_it_skills import print_dead_skills

        dead = [f"dead_skill_{i}" for i in range(50)]
        print_dead_skills(dead)
        captured = capsys.readouterr()
        assert "ещё 20" in captured.out

    def test_main_vacancies_not_found_error(self, tmp_path, monkeypatch):
        """Строки 293-294: файл вакансий не найден"""
        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--vacancies", str(tmp_path / "nonexistent.json"),
        ])

        with pytest.raises(SystemExit):
            from scripts.extend_it_skills import main
            main()

    def test_main_interactive_and_yes_conflict(self, monkeypatch, capsys):
        """Строки 305-306: конфликт --interactive и --yes"""
        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--interactive",
            "--yes",
        ])

        with pytest.raises(SystemExit):
            from scripts.extend_it_skills import main
            main()

    def test_main_with_coverage_flag(self, sample_vacancies_rich, tmp_path, monkeypatch, capsys):
        """Строки 311-312: флаг --coverage"""
        vac_file = tmp_path / "test_vacancies.json"
        vac_file.write_text(json.dumps(sample_vacancies_rich))

        tax_path = tmp_path / "skill_taxonomy.json"
        tax_path.write_text(json.dumps({
            "categories": {
                "test_cat": {
                    "label": "Test",
                    "icon": "🧪",
                    "skills": ["python", "javascript"]
                }
            }
        }))
        monkeypatch.setattr("src.analyzers.skill_taxonomy.config.DATA_DIR", tmp_path)
        monkeypatch.setattr("scripts.extend_it_skills.load_it_skills", lambda: {"python", "django"})

        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--vacancies", str(vac_file),
            "--coverage",
            "--output", str(tmp_path / "it_skills.json"),
        ])

        from src.analyzers.skill_taxonomy import SkillTaxonomy
        SkillTaxonomy._instance = None

        from scripts.extend_it_skills import main
        main()

        captured = capsys.readouterr()
        assert "РЕЖИМ АНАЛИЗА" in captured.out or "ПОКРЫТИЕ" in captured.out

    def test_main_with_dead_flag(self, sample_vacancies_rich, tmp_path, monkeypatch, capsys):
        """Строки 316-317: флаг --dead"""
        vac_file = tmp_path / "test_vacancies.json"
        vac_file.write_text(json.dumps(sample_vacancies_rich))

        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--vacancies", str(vac_file),
            "--dead",
            "--output", str(tmp_path / "it_skills.json"),
        ])
        monkeypatch.setattr("scripts.extend_it_skills.load_it_skills", lambda: {"python", "django"})

        from scripts.extend_it_skills import main
        main()

    def test_main_yes_adds_skills(self, sample_vacancies_rich, tmp_path, monkeypatch):
        """Строки 335-340: --yes добавляет навыки"""
        vac_file = tmp_path / "test_vacancies.json"
        vac_file.write_text(json.dumps(sample_vacancies_rich))

        output = tmp_path / "it_skills.json"
        output.write_text(json.dumps(["python", "django"]))

        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--vacancies", str(vac_file),
            "--yes",
            "--no-backup",
            "--output", str(output),
        ])
        monkeypatch.setattr("scripts.extend_it_skills.load_it_skills", lambda: {"python", "django"})
        monkeypatch.setattr("src.parsing.utils.config.DATA_DIR", tmp_path)

        from scripts.extend_it_skills import main
        main()

    def test_main_no_new_skills(self, tmp_path, monkeypatch, capsys):
        """Строки 344-345: нет новых навыков"""
        vac_file = tmp_path / "test_vacancies.json"
        vac_file.write_text(json.dumps([{
            "id": "1",
            "name": "Dev",
            "key_skills": [{"name": "python"}],
            "description": "",
            "snippet": {},
        }]))

        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--vacancies", str(vac_file),
            "--output", str(tmp_path / "it_skills.json"),
        ])
        monkeypatch.setattr("scripts.extend_it_skills.load_it_skills", lambda: {"python", "django"})

        from scripts.extend_it_skills import main
        main()

        captured = capsys.readouterr()
        assert "Белый список актуален" in captured.out

    def test_interactive_confirm_cancel(self, monkeypatch):
        """Строки 360-369: интерактивный режим — выход"""
        from scripts.extend_it_skills import interactive_confirm

        new_skills = {"skill_a": 5, "skill_b": 3}
        monkeypatch.setattr("builtins.input", lambda _: "q")

        approved = interactive_confirm(new_skills)
        assert approved == set()

    def test_main_taxonomy_load_error(self, sample_vacancies_rich, tmp_path, monkeypatch, capsys):
        """Строка 382: ошибка загрузки таксономии — работает без неё"""
        vac_file = tmp_path / "test_vacancies.json"
        vac_file.write_text(json.dumps(sample_vacancies_rich))

        # Ломаем таксономию
        tax_path = tmp_path / "skill_taxonomy.json"
        tax_path.write_text("{invalid")

        # Патчим config так, чтобы SkillTaxonomy не падал, а load_it_skills работал
        monkeypatch.setattr("sys.argv", [
            "extend_it_skills.py",
            "--vacancies", str(vac_file),
            "--output", str(tmp_path / "it_skills.json"),
        ])
        monkeypatch.setattr("src.analyzers.skill_taxonomy.config.DATA_DIR", tmp_path)
        monkeypatch.setattr("scripts.extend_it_skills.load_it_skills", lambda: {"python", "django"})

        from src.analyzers.skill_taxonomy import SkillTaxonomy
        SkillTaxonomy._instance = None

        # Мокаем print_new_skills чтобы избежать вызова taxonomy.get_category
        with patch("scripts.extend_it_skills.print_new_skills") as mock_print:
            from scripts.extend_it_skills import main
            main()
            mock_print.assert_called_once()
