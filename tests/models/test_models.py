import pytest
from src.models.student import StudentProfile
from src.models.vacancy import Vacancy
from pydantic import ValidationError

def test_student_profile_validation_success(sample_student):
    assert sample_student.name == "Анна Иванова"
    assert len(sample_student.competencies) >= 4

def test_student_profile_validation_error():
    with pytest.raises(ValidationError):
        StudentProfile(name="", competencies=[])

def test_vacancy_model_creation():
    vac = Vacancy(title="Test Vacancy", skills=["Python"], salary_from=100000)
    assert vac.title == "Test Vacancy"