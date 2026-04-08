# tests/loaders/test_loaders.py
import pytest
from src.models.student import StudentProfile


def test_student_loader_loads_single_profile(student_loader):
    """Тест загрузки одного профиля"""
    student = student_loader.load_student("base")
    assert student is not None
    assert isinstance(student, StudentProfile)
    assert student.profile_name == "base"
    assert len(student.competencies) > 0 or len(student.skills) > 0


def test_student_loader_loads_all_profiles(student_loader):
    """Тест загрузки всех профилей"""
    students = student_loader.load_all_students()
    assert len(students) == 3
    assert all(isinstance(s, StudentProfile) for s in students)
    assert any(s.profile_name == "base" for s in students)