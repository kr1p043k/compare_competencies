# tests/loaders/test_loaders.py
import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.loaders_student.student_loader import StudentLoader, generate_profiles_from_csv
from src.models.student import StudentProfile


class TestStudentLoader:
    def test_load_student_file_not_exists(self, tmp_path):
        loader = StudentLoader(students_dir=tmp_path)
        student = loader.load_student("nonexistent")
        assert student is None

    def test_load_student_invalid_json(self, tmp_path):
        file_path = tmp_path / "invalid_competency.json"
        file_path.write_text("{invalid json", encoding='utf-8')
        loader = StudentLoader(students_dir=tmp_path)
        with pytest.raises(json.JSONDecodeError):
            loader.load_student("invalid")

    def test_load_student_missing_skills_field(self, tmp_path):
        file_path = tmp_path / "empty_competency.json"
        file_path.write_text('{"другие_данные": []}', encoding='utf-8')
        loader = StudentLoader(students_dir=tmp_path)
        student = loader.load_student("empty")
        assert student is not None
        assert student.skills == []

    def test_load_student_uses_skills_as_competencies(self, tmp_path):
        file_path = tmp_path / "test_competency.json"
        file_path.write_text('{"навыки": ["Python", "SQL"]}', encoding='utf-8')
        loader = StudentLoader(students_dir=tmp_path)
        student = loader.load_student("test")
        assert student.competencies == ["Python", "SQL"]
        assert student.skills == ["Python", "SQL"]
        assert student.profile_name == "test"
    def test_load_all_students_loads_all_profiles(self, tmp_path):
        # Создаём все три JSON-файла
        profiles = ["base", "dc", "top_dc"]
        for p in profiles:
            (tmp_path / f"{p}_competency.json").write_text('{"навыки": ["A"]}', encoding='utf-8')

        loader = StudentLoader(students_dir=tmp_path)
        students = loader.load_all_students()

        assert len(students) == 3
        loaded_names = {s.profile_name for s in students}
        assert loaded_names == {"base", "dc", "top_dc"}
    def test_load_student_sets_target_level(self, tmp_path):
        file_path = tmp_path / "test_competency.json"
        file_path.write_text('{"навыки": ["Python"]}', encoding='utf-8')
        loader = StudentLoader(students_dir=tmp_path)
        student = loader.load_student("test")
        assert student.target_level == "middle"   


class TestGenerateProfilesFromCSV:
    def test_csv_not_found(self, tmp_path):
        nonexistent = tmp_path / "no_such_file.csv"
        with pytest.raises(FileNotFoundError):
            generate_profiles_from_csv(csv_path=nonexistent, output_dir=tmp_path, save_copy=False)

    def test_generate_profiles_success(self, tmp_path):
        # Важно: одинаковое количество запятых во всех строках (4 поля)
        csv_content = """\
,Заголовок отчёта (игнорируется),,
,SS1.1 описание,SS2.1 описание,
1,Дисциплина 1,Б,
2,Дисциплина 2,,П
3,Дисциплина 3,Б,Б
"""
        csv_path = tmp_path / "competency_matrix.csv"
        csv_path.write_text(csv_content, encoding='utf-8')

        output_dir = tmp_path / "students"
        output_dir.mkdir()

        with patch('src.loaders_student.student_loader.PROFILES_DISCIPLINES',
                   {'base': [1, 2], 'dc': [3]}):
            profiles = generate_profiles_from_csv(csv_path, output_dir, save_copy=False)

        assert 'base' in profiles
        assert 'dc' in profiles
        assert set(profiles['base']) == {'SS1.1', 'SS2.1'}
        assert set(profiles['dc']) == {'SS1.1', 'SS2.1'}

        base_json = output_dir / "base_competency.json"
        assert base_json.exists()
        with open(base_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert sorted(data["навыки"]) == ['SS1.1', 'SS2.1']

    def test_generate_profiles_handles_unicode_error(self, tmp_path):
        csv_path = tmp_path / "competency_matrix.csv"
        # Запишем в cp1251 (3 поля: пусто, индикатор, пусто)
        content = ",Заголовок,\n,Код1 описание,\n1,Дисц1,Б\n".encode('cp1251')
        csv_path.write_bytes(content)

        output_dir = tmp_path / "students"
        output_dir.mkdir()

        with patch('src.loaders_student.student_loader.PROFILES_DISCIPLINES', {'base': [1]}):
            profiles = generate_profiles_from_csv(csv_path, output_dir, save_copy=False)

        assert 'base' in profiles
        assert 'Код1' in profiles['base']

    def test_generate_profiles_save_copy(self, tmp_path):
        # Минимальный CSV с одинаковым числом полей (3 поля)
        csv_content = ",h1,\n,ind,\n1,Дисц,Б\n"
        csv_path = tmp_path / "competency_matrix.csv"
        csv_path.write_text(csv_content, encoding='utf-8')
        output_dir = tmp_path / "students"
        output_dir.mkdir()

        with patch('src.loaders_student.student_loader.PROFILES_DISCIPLINES', {'base': [1]}):
            with patch('src.loaders_student.student_loader.LAST_UPLOADED_DIR', tmp_path / "last"):
                generate_profiles_from_csv(csv_path, output_dir, save_copy=True)
                last_csv = tmp_path / "last" / "competency_matrix.csv"
                assert last_csv.exists()

    def test_generate_profiles_empty_disciplines(self, tmp_path):
        # Только заголовки, без дисциплин (2 строки, 3 поля)
        csv_content = ",h1,\n,SS1.1,\n"
        csv_path = tmp_path / "competency_matrix.csv"
        csv_path.write_text(csv_content, encoding='utf-8')
        output_dir = tmp_path / "students"
        output_dir.mkdir()

        with patch('src.loaders_student.student_loader.PROFILES_DISCIPLINES', {'empty': [99]}):
            profiles = generate_profiles_from_csv(csv_path, output_dir, save_copy=False)
        assert profiles['empty'] == []

    def test_generate_profiles_handles_nan_values(self, tmp_path):
        # Пустое значение в столбце индикатора
        csv_content = ",h1,\n,SS1.1,\n1,Дисц1,Б\n2,Дисц2,\n"
        csv_path = tmp_path / "competency_matrix.csv"
        csv_path.write_text(csv_content, encoding='utf-8')
        output_dir = tmp_path / "students"
        output_dir.mkdir()

        with patch('src.loaders_student.student_loader.PROFILES_DISCIPLINES', {'base': [1, 2]}):
            profiles = generate_profiles_from_csv(csv_path, output_dir, save_copy=False)
        assert profiles['base'] == ['SS1.1']

    def test_generate_profiles_with_missing_values(self, tmp_path):
        # Проверяем обработку пропущенных значений (NaN) в CSV
        csv_content = """,h1,
    ,SS1.1,
    1,Дисц1,Б
    2,Дисц2,
    """
        csv_path = tmp_path / "competency_matrix.csv"
        csv_path.write_text(csv_content, encoding='utf-8')
        output_dir = tmp_path / "students"
        output_dir.mkdir()

        with patch('src.loaders_student.student_loader.PROFILES_DISCIPLINES', {'base': [1, 2]}):
            profiles = generate_profiles_from_csv(csv_path, output_dir, save_copy=False)

        # Только первая дисциплина имеет отметку 'Б'
        assert profiles['base'] == ['SS1.1']
    def test_generate_profiles_save_copy_creates_dir(self, tmp_path):
        csv_content = ",h1,\n,ind,\n1,Дисц,Б\n"
        csv_path = tmp_path / "competency_matrix.csv"
        csv_path.write_text(csv_content, encoding='utf-8')
        output_dir = tmp_path / "students"
        output_dir.mkdir()

        last_uploaded_dir = tmp_path / "last_uploaded"  # ещё не существует

        with patch('src.loaders_student.student_loader.PROFILES_DISCIPLINES', {'base': [1]}):
            with patch('src.loaders_student.student_loader.LAST_UPLOADED_DIR', last_uploaded_dir):
                generate_profiles_from_csv(csv_path, output_dir, save_copy=True)

        assert last_uploaded_dir.exists()
        assert (last_uploaded_dir / "competency_matrix.csv").exists()
        
    def test_generate_profiles_logs_warning_for_empty_profile(self, tmp_path, caplog):
        csv_content = ",h1,\n,ind,\n1,Дисц1,Б\n"
        csv_path = tmp_path / "competency_matrix.csv"
        csv_path.write_text(csv_content, encoding='utf-8')
        output_dir = tmp_path / "students"
        output_dir.mkdir()

        with patch('src.loaders_student.student_loader.PROFILES_DISCIPLINES', {'empty': [99]}):
            with caplog.at_level('WARNING'):
                generate_profiles_from_csv(csv_path, output_dir, save_copy=False)

        assert "не найдено ни одной дисциплины" in caplog.text
    
    def test_generate_profiles_non_numeric_discipline_id(self, tmp_path):
        csv_content = """,h1,
    ,ind,
    abc,Дисц1,Б
    """
        csv_path = tmp_path / "competency_matrix.csv"
        csv_path.write_text(csv_content, encoding='utf-8')
        output_dir = tmp_path / "students"
        output_dir.mkdir()

        with patch('src.loaders_student.student_loader.PROFILES_DISCIPLINES', {'base': [0]}):
            profiles = generate_profiles_from_csv(csv_path, output_dir, save_copy=False)
        # Дисциплина с id=0 будет обработана, навыки извлекутся
        assert profiles['base'] == ['ind']

    def test_generate_profiles_read_csv_exception(self, tmp_path):
        csv_path = tmp_path / "dummy.csv"
        csv_path.write_text("dummy", encoding='utf-8')
        output_dir = tmp_path / "students"
        output_dir.mkdir()

        with patch('pandas.read_csv', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                generate_profiles_from_csv(csv_path, output_dir, save_copy=False)

class TestStudentLoaderEdgeCases:
    def test_load_all_students_skips_missing(self, tmp_path):
        (tmp_path / "base_competency.json").write_text('{"навыки": ["A"]}', encoding='utf-8')
        loader = StudentLoader(students_dir=tmp_path)
        students = loader.load_all_students()
        assert len(students) == 1
        assert students[0].profile_name == "base"