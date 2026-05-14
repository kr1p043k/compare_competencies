# tests/analyzers/test_skill_taxonomy.py
import json
import pytest

from src.analyzers.skills.skill_taxonomy import SkillTaxonomy


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
        """Создаёт временный файл таксономии и подменяет SKILL_TAXONOMY_PATH"""
        taxonomy_data = {
            "categories": {
                "programming_languages": {
                    "label": "Языки программирования",
                    "icon": "💻",
                    "skills": ["Python", "Java", "JavaScript", "TypeScript", "Go", "Rust"]
                },
                "frameworks": {
                    "label": "Фреймворки и библиотеки",
                    "icon": "🔧",
                    "skills": ["React", "Vue", "Angular", "Django", "Flask", "FastAPI"]
                },
                "devops": {
                    "label": "DevOps и инфраструктура",
                    "icon": "🚀",
                    "skills": ["Docker", "Kubernetes", "Jenkins", "Git", "Terraform"]
                },
                "databases": {
                    "label": "Базы данных",
                    "icon": "🗄️",
                    "skills": ["PostgreSQL", "MySQL", "MongoDB", "Redis"]
                },
                "soft_skills": {
                    "label": "Soft skills",
                    "icon": "🤝",
                    "skills": ["Communication", "Teamwork", "Leadership"]
                }
            }
        }
        tax_path = tmp_path / "skill_taxonomy.json"
        tax_path.write_text(json.dumps(taxonomy_data), encoding="utf-8")
        # Патчим конечный путь, а не DATA_DIR
        monkeypatch.setattr("src.config.SKILL_TAXONOMY_PATH", tax_path)
        return taxonomy_data

    def test_singleton_pattern(self):
        t1 = SkillTaxonomy()
        t2 = SkillTaxonomy()
        assert t1 is t2

    def test_load_taxonomy_from_file(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        assert taxonomy._taxonomy is not None
        assert len(taxonomy._skill_to_category) > 0
        assert "python" in taxonomy._skill_to_category

    def test_load_taxonomy_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.config.SKILL_TAXONOMY_PATH", tmp_path / "nonexistent.json")
        taxonomy = SkillTaxonomy()
        assert taxonomy._taxonomy is None
        assert taxonomy.get_category("python") == "other"

    def test_load_taxonomy_invalid_json(self, tmp_path, monkeypatch):
        tax_path = tmp_path / "skill_taxonomy.json"
        tax_path.write_text("{invalid json", encoding="utf-8")
        monkeypatch.setattr("src.config.SKILL_TAXONOMY_PATH", tax_path)
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category("python") == "other"

    def test_get_category_known(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category("python") == "programming_languages"
        assert taxonomy.get_category("Python") == "programming_languages"
        assert taxonomy.get_category("  docker  ") == "devops"

    def test_get_category_unknown(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category("unknown_skill") == "other"

    def test_get_category_label(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category_label("python") == "Языки программирования"
        assert taxonomy.get_category_label("unknown") == "other"

    def test_get_category_icon(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category_icon("python") == "💻"
        assert taxonomy.get_category_icon("docker") == "🚀"
        assert taxonomy.get_category_icon("unknown") == ""

    def test_get_category_label_by_id(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category_label_by_id("programming_languages") == "Языки программирования"
        assert taxonomy.get_category_label_by_id("unknown") == "unknown"

    def test_get_category_icon_by_id(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_category_icon_by_id("devops") == "🚀"

    def test_get_skills_in_category(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        skills = taxonomy.get_skills_in_category("programming_languages")
        assert "Python" in skills
        assert "Java" in skills
        assert len(skills) == 6   # теперь совпадает с sample

    def test_get_skills_in_category_empty(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        skills = taxonomy.get_skills_in_category("nonexistent")
        assert skills == []

    def test_get_all_categories(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        cats = taxonomy.get_all_categories()
        assert "programming_languages" in cats
        assert "frameworks" in cats
        assert len(cats) == 5   # ровно 5 категорий из sample

    def test_get_category_stats(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        stats = taxonomy.get_category_stats(["python", "java", "docker", "react", "unknown"])
        assert stats.get("programming_languages", 0) == 2
        assert stats.get("devops", 0) == 1
        assert stats.get("frameworks", 0) == 1   # react теперь в frameworks

    def test_get_dominant_category(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        dominant = taxonomy.get_dominant_category(["python", "java", "go", "docker"])
        assert dominant == "programming_languages"

    def test_get_dominant_category_empty(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        assert taxonomy.get_dominant_category([]) == "other"

    def test_get_dominant_category_label(self, sample_taxonomy):
        taxonomy = SkillTaxonomy()
        label = taxonomy.get_dominant_category_label(["react", "vue", "angular", "fastapi"])
        assert label == "Фреймворки и библиотеки"   # fastapi попадает в frameworks
