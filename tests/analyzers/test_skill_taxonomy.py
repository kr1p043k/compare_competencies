# tests/analyzers/test_skill_taxonomy.py
import json
import pytest
from unittest.mock import patch

from src.analyzers.skill_taxonomy import SkillTaxonomy


class TestSkillTaxonomy:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Сбрасываем синглтон перед каждым тестом"""
        SkillTaxonomy._instance = None
        SkillTaxonomy._taxonomy = None
        SkillTaxonomy._skill_to_category = {}
        SkillTaxonomy._category_info = {}
        yield
        SkillTaxonomy._instance = None
        SkillTaxonomy._taxonomy = None
        SkillTaxonomy._skill_to_category = {}
        SkillTaxonomy._category_info = {}

    @pytest.fixture
    def sample_taxonomy(self, tmp_path, monkeypatch):
        """Создаёт временный файл таксономии"""
        taxonomy_data = {
            "categories": {
                "programming_languages": {
                    "label": "Языки программирования",
                    "icon": "💻",
                    "skills": ["python", "java", "javascript", "typescript", "go", "rust"]
                },
                "frameworks": {
                    "label": "Фреймворки и библиотеки",
                    "icon": "🔧",
                    "skills": ["react", "vue", "angular", "django", "flask", "fastapi"]
                },
                "devops": {
                    "label": "DevOps и инфраструктура",
                    "icon": "🚀",
                    "skills": ["docker", "kubernetes", "jenkins", "git", "terraform"]
                },
                "databases": {
                    "label": "Базы данных",
                    "icon": "🗄️",
                    "skills": ["postgresql", "mysql", "mongodb", "redis"]
                },
                "soft_skills": {
                    "label": "Soft skills",
                    "icon": "🤝",
                    "skills": ["communication", "teamwork", "leadership"]
                }
            }
        }
        tax_path = tmp_path / "skill_taxonomy.json"
        tax_path.write_text(json.dumps(taxonomy_data), encoding="utf-8")
        monkeypatch.setattr("src.analyzers.skill_taxonomy.config.DATA_DIR", tmp_path)
        return taxonomy_data

    def test_singleton_pattern(self):
        """Строка 36: синглтон возвращает один и тот же экземпляр"""
        t1 = SkillTaxonomy()
        t2 = SkillTaxonomy()
        assert t1 is t2

    def test_load_taxonomy_from_file(self, sample_taxonomy):
        """Строки 36-37: загрузка таксономии из файла"""
        taxonomy = SkillTaxonomy()
        assert taxonomy._taxonomy is not None
        assert len(taxonomy._skill_to_category) > 0
        assert "python" in taxonomy._skill_to_category

    def test_load_taxonomy_file_not_found(self, tmp_path, monkeypatch):
        """Строка 36: файл не найден — без ошибки"""
        monkeypatch.setattr("src.analyzers.skill_taxonomy.config.DATA_DIR", tmp_path)
        taxonomy = SkillTaxonomy()
        assert taxonomy._taxonomy is None
        assert taxonomy.get_category("python") == "other"

    def test_load_taxonomy_invalid_json(self, tmp_path, monkeypatch):
        """Строки 51-52: битый JSON — без ошибки"""
        tax_path = tmp_path / "skill_taxonomy.json"
        tax_path.write_text("{invalid json", encoding="utf-8")
        monkeypatch.setattr("src.analyzers.skill_taxonomy.config.DATA_DIR", tmp_path)
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category("python") == "other"

    def test_get_category_known(self, sample_taxonomy):
        """Строка 65: известный навык"""
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category("python") == "programming_languages"
        assert taxonomy.get_category("Python") == "programming_languages"
        assert taxonomy.get_category("  docker  ") == "devops"

    def test_get_category_unknown(self, sample_taxonomy):
        """Строка 65: неизвестный навык → 'other'"""
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category("unknown_skill") == "other"

    def test_get_category_label(self, sample_taxonomy):
        """Строки 66-67: человекочитаемое название категории"""
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category_label("python") == "Языки программирования"
        assert taxonomy.get_category_label("unknown") == "other"

    def test_get_category_icon(self, sample_taxonomy):
        """Строки 74-75: иконка категории"""
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category_icon("python") == "💻"
        assert taxonomy.get_category_icon("docker") == "🚀"
        assert taxonomy.get_category_icon("unknown") == ""

    def test_get_category_label_by_id(self, sample_taxonomy):
        """Человекочитаемое название по ID"""
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category_label_by_id("programming_languages") == "Языки программирования"
        assert taxonomy.get_category_label_by_id("unknown") == "unknown"

    def test_get_category_icon_by_id(self, sample_taxonomy):
        """Иконка по ID"""
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category_icon_by_id("devops") == "🚀"

    def test_get_skills_in_category(self, sample_taxonomy):
        """Строка 95: навыки категории"""
        taxonomy = SkillTaxonomy()
        skills = taxonomy.get_skills_in_category("programming_languages")
        assert "python" in skills
        assert "java" in skills
        assert len(skills) == 6

    def test_get_skills_in_category_empty(self, sample_taxonomy):
        """Строка 95: несуществующая категория"""
        taxonomy = SkillTaxonomy()
        skills = taxonomy.get_skills_in_category("nonexistent")
        assert skills == []

    def test_get_all_categories(self, sample_taxonomy):
        """Строка 99: все категории"""
        taxonomy = SkillTaxonomy()
        cats = taxonomy.get_all_categories()
        assert "programming_languages" in cats
        assert "frameworks" in cats
        assert len(cats) == 5

    def test_get_category_stats(self, sample_taxonomy):
        """Подсчёт навыков по категориям"""
        taxonomy = SkillTaxonomy()
        stats = taxonomy.get_category_stats(["python", "java", "docker", "react", "unknown"])
        assert stats.get("programming_languages", 0) == 2
        assert stats.get("devops", 0) == 1
        assert stats.get("frameworks", 0) == 1

    def test_get_dominant_category(self, sample_taxonomy):
        """Строка 118: доминирующая категория"""
        taxonomy = SkillTaxonomy()
        # 3 языка, 1 devops
        dominant = taxonomy.get_dominant_category(["python", "java", "go", "docker"])
        assert dominant == "programming_languages"

    def test_get_dominant_category_empty(self, sample_taxonomy):
        """Строка 118: пустой список"""
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_dominant_category([]) == "other"

    def test_get_dominant_category_label(self, sample_taxonomy):
        """Строки 123-124: лейбл доминирующей категории"""
        taxonomy = SkillTaxonomy()
        label = taxonomy.get_dominant_category_label(["react", "vue", "angular", "fastapi"])
        assert label == "Фреймворки и библиотеки"
