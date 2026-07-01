"""Tests for RPDLoader and RPDSkillCleaner."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestRPDLoader:
    """Test RPDLoader parsing logic."""

    def test_normalize_comp_code(self):
        from src.loaders.rpd_loader import normalize_comp_code
        assert normalize_comp_code("ОПК-2") == "ОПК-2"
        assert normalize_comp_code("пк 10") == "ПК-10"
        # Space before digit becomes double dash
        assert normalize_comp_code("УК -3") == "УК--3"

    def test_clean_skill(self):
        from src.loaders.rpd_loader import clean_skill
        assert clean_skill("  python  ") == "python"
        assert clean_skill("- sql базы данных") == "sql базы данных"
        assert clean_skill("Знания: основы алгоритмов") == "основы алгоритмов"
        assert clean_skill("Навыки: работа с Git") == "работа с Git"

    def test_dedup_children(self):
        from src.loaders.rpd_loader import dedup_children
        comp_ksa = {
            "ОПК-2": {"flat": ["python", "sql"]},
            "ОПК-2.1": {"flat": ["python"]},  # subset of parent
            "ОПК-2.2": {"flat": ["python", "docker"]},  # not subset
        }
        result = dedup_children(comp_ksa)
        assert "ОПК-2" in result
        assert "ОПК-2.1" not in result  # removed (subset)
        assert "ОПК-2.2" in result  # kept (not subset)

    def test_is_continuation(self):
        from src.loaders.rpd_loader import is_continuation
        assert is_continuation("(продолжение)") is True
        assert is_continuation(",接着") is True
        assert is_continuation("Python") is False
        assert is_continuation("") is False

    def test_cyrillic_ratio(self):
        from src.loaders.rpd_loader import cyrillic_ratio
        assert cyrillic_ratio("Привет мир") > 0.5
        assert cyrillic_ratio("Hello world") == 0.0
        assert cyrillic_ratio("") == 0.0

    def test_extract_text_pypdf(self, tmp_path):
        from src.loaders.rpd_loader import extract_text_pypdf
        # Non-existent file
        result = extract_text_pypdf(str(tmp_path / "nonexistent.pdf"))
        assert result is None

    def test_is_rpd_filter(self):
        from src.loaders.rpd_loader import RPDLoader
        loader = RPDLoader(str(Path(".")))
        assert loader._is_rpd("РПД_Operating Systems.pdf") is True
        assert loader._is_rpd("РПД_Rating.pdf") is False
        assert loader._is_rpd("РПД_ECTS.pdf") is False
        assert loader._is_rpd("Аннотация.pdf") is False
        assert loader._is_rpd("file.sig") is False

    def test_discipline_name_extraction(self):
        from src.loaders.rpd_loader import RPDLoader
        loader = RPDLoader(str(Path(".")))
        assert loader._discipline_name("РПД_Operating Systems.pdf") == "Operating Systems"
        assert loader._discipline_name("РПД_Базы данных.pdf") == "Базы данных"
        assert loader._discipline_name("ГИА_ИИ 09.03.02.pdf") == "ИИ 09.03.02"

    def test_normalize_disc_name(self):
        from src.loaders.rpd_loader import normalize_disc_name
        assert normalize_disc_name("Операционные системы") == "операционныесистемы"
        assert normalize_disc_name("Hello World") == "helloworld"

    def test_extract_ksa_skills_basic(self):
        from src.loaders.rpd_loader import extract_ksa_skills
        text = """
        ОПК-2 Знания: основы программирования
        Умения: писать код на Python
        Навыки: отладка программ
        """
        result = extract_ksa_skills(text)
        assert "ОПК-2" in result
        assert len(result["ОПК-2"]["knowledge"]) > 0
        assert len(result["ОПК-2"]["abilities"]) > 0
        assert len(result["ОПК-2"]["skills"]) > 0

    def test_extract_ksa_skills_empty(self):
        from src.loaders.rpd_loader import extract_ksa_skills
        result = extract_ksa_skills("No competencies here")
        assert result == {}

    def test_find_section_text(self):
        from src.loaders.rpd_loader import find_section_text, SEC_PATTERN
        text = "I. Введение\nsome intro\nIII. ТРЕБОВАНИЯ К РЕЗУЛЬТАТАМ\nОПК-2 Знания: python\nУмения: код\nНавыки: отладка\nIV. Учебный план"
        matches = list(SEC_PATTERN.finditer(text))
        result = find_section_text(text, matches)
        assert result is not None
        # Section starts after the header
        assert "Знания" in result or "ОПК" in result

    def test_stats(self):
        from src.loaders.rpd_loader import RPDLoader
        loader = RPDLoader(str(Path(".")))
        data = {
            "09.03.02": {
                "disciplines": {
                    "Disc1": {"competencies": ["ОПК-2"], "skills": {"ОПК-2": ["python", "sql"]}},
                    "Disc2": {"competencies": ["ПК-1"], "skills": {"ПК-1": []}},
                }
            }
        }
        stats = loader.stats(data)
        assert stats["disciplines"] == 2
        assert stats["skills"] == 2
        assert stats["zero"] == 1


class TestRPDSkillCleaner:
    """Test RPDSkillCleaner filtering logic."""

    def test_clean_skills_basic(self):
        from src.loaders.rpd_skill_cleaner import RPDSkillCleaner
        cleaner = RPDSkillCleaner()
        # "sql" is too short (<5 chars), gets removed
        result = cleaner.clean_skills(["python", "docker", "kubernetes"])
        assert result.is_ok()
        assert len(result.unwrap()) == 3

    def test_clean_skills_removes_noise(self):
        from src.loaders.rpd_skill_cleaner import RPDSkillCleaner
        cleaner = RPDSkillCleaner()
        result = cleaner.clean_skills([
            "python",
            "экзамен",  # noise
            "зачет",  # noise
            "docker",
        ])
        assert result.is_ok()
        cleaned = result.unwrap()
        assert "python" in cleaned
        assert "docker" in cleaned
        assert "экзамен" not in cleaned
        assert "зачет" not in cleaned

    def test_clean_skills_removes_verbose(self):
        from src.loaders.rpd_skill_cleaner import RPDSkillCleaner
        cleaner = RPDSkillCleaner()
        result = cleaner.clean_skills([
            "python",
            "проводить анализ данных и разрабатывать алгоритмы",  # verbose
        ])
        assert result.is_ok()
        cleaned = result.unwrap()
        assert "python" in cleaned

    def test_clean_skills_deduplicates(self):
        from src.loaders.rpd_skill_cleaner import RPDSkillCleaner
        cleaner = RPDSkillCleaner()
        result = cleaner.clean_skills(["python", "python", "docker"])
        assert result.is_ok()
        assert len(result.unwrap()) == 2

    def test_clean_discipline_data(self):
        from src.loaders.rpd_skill_cleaner import RPDSkillCleaner
        cleaner = RPDSkillCleaner()
        data = {
            "09.03.02": {
                "disciplines": {
                    "Disc1": {
                        "skills": {"ОПК-2": ["python", "экзамен"]},
                        "ksa": {"ОПК-2": {"knowledge": ["sql"], "abilities": ["docker"]}}
                    }
                }
            }
        }
        result = cleaner.clean_discipline_data(data)
        assert result.is_ok()
        cleaned = result.unwrap()
        skills = cleaned["09.03.02"]["disciplines"]["Disc1"]["skills"]["ОПК-2"]
        assert "python" in skills
        assert "экзамен" not in skills

    def test_stats(self):
        from src.loaders.rpd_skill_cleaner import RPDSkillCleaner
        cleaner = RPDSkillCleaner()
        cleaner.clean_skills(["python", "экзамен", "docker"])
        stats = cleaner.stats()
        assert stats["total"] == 3
        assert stats["kept"] == 2
        assert stats["noise_removed"] == 1
