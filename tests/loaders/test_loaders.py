import pytest
from pathlib import Path
from src.models.student import StudentProfile
from src.loaders_student.student_loader import StudentLoader
from src.loaders_student.student_loader import generate_profiles_from_csv
def test_student_loader_loads_single_profile(StudentLoader):
    student = StudentLoader.load_student("base")
    assert student is not None
    assert isinstance(student, StudentProfile)
    assert len(student.competencies) > 0


def test_student_loader_loads_all_profiles(StudentLoader):
    students = StudentLoader.load_all_students()
    assert len(students) == 3
    assert all(isinstance(s, StudentProfile) for s in students)


def test_generate_profiles_from_csv_creates_files(StudentLoader, tmp_path):
    # Тест с временной папкой
    csv_path = Path(__file__).parent.parent.parent / "data" / "raw" / "competency_matrix.csv"
    if not csv_path.exists():
        pytest.skip("CSV файл отсутствует")
    
    output_dir = tmp_path / "students"
    result = StudentLoader.generate_profiles_from_csv(  # если метод доступен через класс
        csv_path=csv_path, output_dir=output_dir, save_copy=False
    )
    assert isinstance(result, dict)
    assert len(result) > 0