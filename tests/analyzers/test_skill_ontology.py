from unittest.mock import MagicMock, patch
import json

import pytest

from src import Ok, Err
from src.analyzers.skills.skill_ontology import SkillOntology, seed_ontology
from src.errors import DomainError


class TestSkillOntology:
    @pytest.fixture
    def ont(self):
        o = SkillOntology()
        o.load_seed_data()
        return o

    def test_init_default(self):
        o = SkillOntology()
        assert o._data is not None
        assert "categories" in o._data
        assert "skills" in o._data
        assert "relations" in o._data

    def test_init_from_path(self, tmp_path):
        data = {"categories": {}, "skills": {}, "relations": {}}
        p = tmp_path / "ont.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        o = SkillOntology(str(p))
        assert o._data is not None

    def test_category(self, ont):
        cat = ont.category("programming")
        assert cat is not None
        assert cat["label"] is not None

    def test_category_missing(self, ont):
        assert ont.category("nonexistent") is None

    def test_add_skill(self, ont):
        ont.add_skill("rust", "programming", ["rust-lang"])
        assert ont.category_of("rust") == "programming"

    def test_add_skill_new_category(self, ont):
        ont.add_skill("figma", "design")
        assert ont.category_of("figma") == "design"

    def test_category_of_known(self, ont):
        assert ont.category_of("python") == "programming"
        assert ont.category_of("sql") == "databases"
        assert ont.category_of("docker") == "devops"

    def test_category_of_unknown(self, ont):
        assert ont.category_of("nonexistent") is None

    def test_category_of_case_insensitive(self, ont):
        assert ont.category_of("Python") == "programming"

    def test_add_relation_prerequisite(self, ont):
        ont.add_relation("prerequisite", "sql", "mongodb")
        prereqs = ont.prerequisites("mongodb")
        assert "sql" in prereqs

    def test_prerequisites_chain(self, ont):
        prereqs = ont.prerequisites("machine learning")
        assert "python" in prereqs
        assert "statistics" in prereqs
        assert "linear algebra" in prereqs

    def test_prerequisites_none(self, ont):
        assert ont.prerequisites("python") == []

    def test_similar(self, ont):
        sim = ont.similar("tensorflow")
        assert "pytorch" in sim

    def test_similar_reverse(self, ont):
        sim = ont.similar("pytorch")
        assert "tensorflow" in sim

    def test_similar_none(self, ont):
        assert ont.similar("python") == []

    def test_to_dict(self, ont):
        d = ont.to_dict()
        assert "categories" in d
        assert "skills" in d
        assert "relations" in d

    def test_save_and_reload(self, ont, tmp_path):
        p = tmp_path / "saved.json"
        ont.save(str(p))
        assert p.exists()
        o2 = SkillOntology(str(p))
        assert o2.category_of("python") == "programming"

    def test_save_no_path(self, ont):
        ont._path = None
        ont.save()

    def test_find_query_plan(self):
        plan = SkillOntology.find_query_plan("machine learning")
        assert "python" in plan
        assert "statistics" in plan
        assert "linear algebra" in plan
        assert "machine learning" in plan

    def test_find_query_plan_unknown(self):
        assert SkillOntology.find_query_plan("nonexistent") == []

    def test_seed_ontology(self, tmp_path):
        p = tmp_path / "seeded.json"
        result = seed_ontology(str(p))
        assert result.is_ok()
        assert p.exists()
        o = SkillOntology(str(p))
        assert o.category_of("python") is not None

    def test_seed_ontology_invalid_path(self):
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = seed_ontology("/invalid/path.json")
            assert result.is_err()
