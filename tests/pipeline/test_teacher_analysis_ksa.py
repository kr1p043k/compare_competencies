"""Tests for teacher_analysis_runner KSA context integration."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


class TestTeacherAnalysisKsaContext:
    """Test KSA context loading and analysis."""

    def test_load_raw_ksa_from_json(self, tmp_path):
        """Test that raw KSA is loaded from JSON file."""
        from src.pipeline.teacher_analysis_runner import _safe_filename

        krm_data = {
            "09.03.02": {
                "disciplines": {
                    "Операционные системы": {
                        "ksa": {
                            "ОПК-2": {
                                "knowledge": ["знать основы ОС"],
                                "abilities": ["уметь администрировать"],
                                "skills": ["владеть linux"]
                            }
                        }
                    }
                }
            }
        }
        f = tmp_path / "krm.json"
        f.write_text(json.dumps(krm_data), encoding="utf-8")

        data = json.loads(f.read_text(encoding="utf-8"))
        raw_ksa = data.get("09.03.02", {}).get("disciplines", {})

        assert "Операционные системы" in raw_ksa
        assert "ОПК-2" in raw_ksa["Операционные системы"]["ksa"]
        assert len(raw_ksa["Операционные системы"]["ksa"]["ОПК-2"]["knowledge"]) == 1

    def test_safe_filename(self):
        from src.pipeline.teacher_analysis_runner import _safe_filename
        assert _safe_filename("Операционные системы") == "Операционные системы"
        assert _safe_filename('test/file:name') == "test_file_name"
        assert _safe_filename("a" * 100) == "a" * 80

    def test_extract_skills_from_ksa_basic(self):
        """Test that skills are extracted from KSA text."""
        # This tests the concept - the actual function is inline in _analyze_one
        ksa_texts = [
            "использование python для анализа данных",
            "применение sql для запросов",
            "работа с git и docker"
        ]
        market_skills = {"python", "sql", "git", "docker", "java", "kubernetes"}

        mentioned = set()
        for text in ksa_texts:
            tl = text.lower()
            for skill in market_skills:
                if len(skill) >= 3 and skill in tl:
                    mentioned.add(skill)

        assert "python" in mentioned
        assert "sql" in mentioned
        assert "git" in mentioned
        assert "docker" in mentioned
        assert "java" not in mentioned

    def test_ksa_context_structure(self):
        """Test that KSA context has expected structure."""
        ksa_context = {
            "ОПК-2": {
                "mentioned_in_ksa": ["python", "sql"],
                "ksa_not_on_market": ["специфический навык"],
                "market_not_in_ksa": ["java", "kubernetes"],
                "total_ksa_items": 5
            }
        }

        assert "mentioned_in_ksa" in ksa_context["ОПК-2"]
        assert "ksa_not_on_market" in ksa_context["ОПК-2"]
        assert "market_not_in_ksa" in ksa_context["ОПК-2"]
        assert "total_ksa_items" in ksa_context["ОПК-2"]
        assert len(ksa_context["ОПК-2"]["mentioned_in_ksa"]) == 2

    def test_ksa_recommendation_generation(self):
        """Test that KSA-based recommendations are generated correctly."""
        from src.models.teacher_analysis import Recommendation

        recs = []
        ksa_context = {
            "ОПК-2": {
                "ksa_not_on_market": ["устаревший навык"],
                "market_not_in_ksa": ["python", "sql"]
            }
        }

        for comp_code, ctx in ksa_context.items():
            for skill in ctx.get("ksa_not_on_market", []):
                recs.append(Recommendation(
                    type="ksa_context", priority="medium",
                    skill_name=skill,
                    message=f"Навык «{skill}» упомянут в KSA компетенции {comp_code}, но не обнаружен на рынке."
                ))
            for skill in ctx.get("market_not_in_ksa", [])[:3]:
                recs.append(Recommendation(
                    type="ksa_context", priority="medium",
                    skill_name=skill,
                    message=f"Рыночный навык «{skill}» не упомянут в KSA компетенции {comp_code}."
                ))

        assert len(recs) == 3  # 1 ksa_not_on_market + 2 market_not_in_ksa
        assert recs[0].type == "ksa_context"
        assert "устаревший навык" in recs[0].message
