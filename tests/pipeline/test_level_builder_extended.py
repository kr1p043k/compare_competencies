from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src import LevelBuildError, Ok, Err
from src.models.enums import ExperienceLevel
from src.models.vacancy import Vacancy, Area, Employer, KeySkill, Experience


@pytest.fixture
def builder():
    from src.pipeline.level_builder import LevelBuilder
    return LevelBuilder()


@pytest.fixture
def area():
    return Area(id=1, name="Moscow")


@pytest.fixture
def employer():
    return Employer(id="123", name="Test Corp")


class TestLevelBuilderBuild:
    def test_empty_vacancies(self, builder):
        result = builder.build([], None)
        assert result.is_ok()
        levels, skills = result.unwrap()
        assert levels == []
        assert skills == []

    def test_vacancy_with_key_skills(self, builder, area, employer):
        vac = Vacancy(
            id="1", name="Python Dev", area=area, employer=employer,
            key_skills=[KeySkill(name="Python"), KeySkill(name="Django")],
            description="Do python stuff",
        )
        result = builder.build([vac], None)
        assert result.is_ok()
        levels, skills = result.unwrap()
        assert len(levels) == 1
        assert levels[0]["skills"] == ["Python", "Django"]
        assert levels[0]["description"] == "Do python stuff"
        assert levels[0]["experience"] == ExperienceLevel.MIDDLE

    def test_vacancy_with_extracted_skills(self, builder, area, employer):
        class VacLike(Vacancy):
            pass
        vac = VacLike(
            id="1", name="Py Dev", area=area, employer=employer,
            key_skills=[],
        )
        vac.extracted_skills = ["Py", "Dj"]
        result = builder.build([vac], None)
        assert result.is_ok()
        levels, skills = result.unwrap()
        assert levels[0]["skills"] == ["Py", "Dj"]

    def test_vacancy_with_experience_obj_junior_id(self, builder, area, employer):
        exp = Experience(id="no_experience", name="No exp")
        vac = Vacancy(
            id="1", name="Trainee", area=area, employer=employer,
            key_skills=[KeySkill(name="Python")], experience=exp,
        )
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.JUNIOR

    def test_vacancy_with_experience_obj_senior_id(self, builder, area, employer):
        exp = Experience(id="morethan10", name="10+ years")
        vac = Vacancy(
            id="1", name="Senior Engineer", area=area, employer=employer,
            key_skills=[KeySkill(name="Go")], experience=exp,
        )
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.SENIOR

    def test_vacancy_with_experience_str_junior(self, builder, area, employer):
        vac = Vacancy(
            id="1", name="Dev", area=area, employer=employer,
            key_skills=[KeySkill(name="Python")],
        )
        vac.experience = "junior"
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.JUNIOR

    def test_vacancy_with_experience_str_senior(self, builder, area, employer):
        vac = Vacancy(
            id="1", name="Dev", area=area, employer=employer,
            key_skills=[KeySkill(name="Python")],
        )
        vac.experience = "senior"
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.SENIOR

    def test_vacancy_name_overrides_middle_to_junior(self, builder, area, employer):
        vac = Vacancy(
            id="1", name="Junior Python Dev", area=area, employer=employer,
            key_skills=[KeySkill(name="Python")],
        )
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.JUNIOR

    def test_vacancy_name_overrides_middle_to_senior(self, builder, area, employer):
        vac = Vacancy(
            id="1", name="Senior Python Dev", area=area, employer=employer,
            key_skills=[KeySkill(name="Python")],
        )
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.SENIOR

    def test_vacancy_name_intern_overrides(self, builder, area, employer):
        vac = Vacancy(
            id="1", name="Intern Python Dev", area=area, employer=employer,
            key_skills=[KeySkill(name="Python")],
        )
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.JUNIOR

    def test_vacancy_no_skills_not_added(self, builder, area, employer):
        vac = Vacancy(
            id="1", name="No Skills", area=area, employer=employer,
        )
        result = builder.build([vac], None)
        assert result.is_ok()
        levels, _ = result.unwrap()
        assert len(levels) == 0

    def test_dict_vacancy_with_key_skills(self, builder):
        vac = {"key_skills": [{"name": "Python"}, {"name": "SQL"}], "description": "test", "name": "Dev"}
        result = builder.build([vac], None)
        assert result.is_ok()
        levels, _ = result.unwrap()
        assert levels[0]["skills"] == ["Python", "SQL"]
        assert levels[0]["experience"] == ExperienceLevel.MIDDLE

    def test_dict_vacancy_with_extracted_skills(self, builder):
        vac = {"extracted_skills": ["Py"], "description": "", "name": "Dev"}
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["skills"] == ["Py"]

    def test_dict_vacancy_experience_dict_junior(self, builder):
        vac = {"extracted_skills": ["Py"], "description": "", "name": "Dev",
               "experience": {"id": "less1", "name": "<1 year"}}
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.JUNIOR

    def test_dict_vacancy_experience_dict_senior(self, builder):
        vac = {"extracted_skills": ["Py"], "description": "", "name": "Dev",
               "experience": {"id": "between6and10", "name": "6-10 years"}}
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.SENIOR

    def test_dict_vacancy_experience_str_junior(self, builder):
        vac = {"extracted_skills": ["Py"], "description": "", "name": "Dev",
               "experience": "junior"}
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.JUNIOR

    def test_dict_vacancy_experience_str_senior(self, builder):
        vac = {"extracted_skills": ["Py"], "description": "", "name": "Dev",
               "experience": "senior"}
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.SENIOR

    def test_dict_vacancy_name_junior_override(self, builder):
        vac = {"extracted_skills": ["Py"], "description": "", "name": "Junior Dev"}
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.JUNIOR

    def test_dict_vacancy_name_senior_override(self, builder):
        vac = {"extracted_skills": ["Py"], "description": "", "name": "Senior Dev"}
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.SENIOR

    def test_dict_vacancy_name_intern_override(self, builder):
        vac = {"extracted_skills": ["Py"], "description": "", "name": "Intern Dev"}
        result = builder.build([vac], None)
        assert result.is_ok()
        assert result.unwrap()[0][0]["experience"] == ExperienceLevel.JUNIOR

    def test_dict_vacancy_no_skills_not_added(self, builder):
        vac = {"description": "", "name": "Dev"}
        result = builder.build([vac], None)
        assert result.is_ok()
        assert len(result.unwrap()[0]) == 0

    def test_exception_returns_err(self, builder):
        class RaisyIter:
            def __iter__(self):
                raise ValueError("boom in iteration")
        result = builder.build(RaisyIter(), None)
        assert result.is_err()
        assert isinstance(result.err(), LevelBuildError)
