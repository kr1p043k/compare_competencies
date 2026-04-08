# tests/models/test_models.py
import pytest
from src.models.student import StudentProfile
from src.models.vacancy import Vacancy, Area, Employer, KeySkill
from pydantic import ValidationError


def test_student_profile_validation_success(sample_student):
    assert sample_student.profile_name == "base"
    assert len(sample_student.skills) == 5


def test_student_profile_validation_error():
    """Теперь правильно вызывает ошибку (обязательное поле profile_name)"""
    with pytest.raises(ValidationError):
        StudentProfile(                     # ← без profile_name
            competencies=[],
            skills=[],
            target_level="junior"
        )


def test_vacancy_model_creation():
    area = Area(id=1, name="Москва")
    employer = Employer(id="123", name="Test Corp")
    vac = Vacancy(
        id="test-1",
        name="Test Vacancy",          # ← name, а не title
        area=area,
        employer=employer
    )
    assert vac.name == "Test Vacancy"