"""Тесты VacancySchema."""
from datetime import datetime
from src.models.vacancy_schema import VacancySchema, SalarySchema, KeySkillSchema


class TestVacancySchema:
    def test_from_api_minimal(self):
        data = {"id": 1, "name": "Python Developer"}
        schema = VacancySchema.from_api(data)
        assert schema.id == 1
        assert schema.name == "Python Developer"

    def test_from_api_full(self):
        data = {
            "id": 123,
            "name": "  Senior Python Dev  ",
            "area": {"id": 1, "name": "Moscow"},
            "salary": {"from": 100000, "to": 200000, "currency": "RUB"},
            "keySkills": [{"name": "Python"}],
        }
        schema = VacancySchema.from_api(data)
        assert schema.name == "Senior Python Dev"
        assert schema.area is not None
        assert schema.area.name == "Moscow"
        assert schema.salary is not None
        assert schema.salary.from_amount == 100000
        assert schema.key_skills[0].name == "Python"

    def test_name_not_empty_validator(self):
        import pytest
        with pytest.raises(Exception):
            VacancySchema.from_api({"id": 1, "name": "   "})

    def test_skill_name_stripped(self):
        skill = KeySkillSchema(name="  Python  ")
        assert skill.name == "Python"

    def test_empty_name_skill_raises(self):
        import pytest
        with pytest.raises(Exception, match="cannot be empty"):
            KeySkillSchema(name="  ")

    def test_salary_aliases(self):
        data = {"id": 1, "name": "Dev", "salary": {"from": 50000, "to": 150000}}
        schema = VacancySchema.from_api(data)
        assert schema.salary is not None
        assert schema.salary.from_amount == 50000
        assert schema.salary.to_amount == 150000
