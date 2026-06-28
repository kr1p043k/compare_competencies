"""Tests for DisciplineAwareScorer."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile

import numpy as np
import pytest

from src import config
from src.analyzers.discipline_relevance import DisciplineAwareScorer, DisciplineRelevance


class TestDisciplineRelevance:
    def test_levels(self):
        assert DisciplineRelevance(0.8).level == "CORE"
        assert DisciplineRelevance(0.5).level == "RELATED"
        assert DisciplineRelevance(0.25).level == "ADJACENT"
        assert DisciplineRelevance(0.1).level == "UNRELATED"
        assert DisciplineRelevance(-0.1).level == "UNRELATED"

    def test_combined_clamped(self):
        r = DisciplineRelevance(-0.5)
        assert r.combined == 0.0
        r2 = DisciplineRelevance(0.3)
        assert r2.combined == 0.3


class TestDisciplineAwareScorer:
    def test_load_missing_file(self):
        scorer = DisciplineAwareScorer()
        scorer.load(Path(tempfile.gettempdir()) / "nonexistent.json")
        assert len(scorer.get_discipline_names()) == 0

    def test_load_from_mock_data(self, tmp_path):
        krm = {
            "09.03.02": {
                "disciplines": {
                    "Операционные системы": {
                        "competencies": ["ОПК-2"],
                        "skills": {"ОПК-2": ["linux", "unix", "процессы", "память"]},
                    },
                    "Базы данных": {
                        "competencies": ["ПК-10"],
                        "skills": {"ПК-10": ["sql", "postgresql", "nosql", "индексы"]},
                    },
                }
            }
        }
        f = tmp_path / "krm_disciplines.json"
        f.write_text(json.dumps(krm, ensure_ascii=False), encoding="utf-8")
        scorer = DisciplineAwareScorer()
        scorer.load(f)
        names = scorer.get_discipline_names()
        assert "Операционные системы" in names
        assert "Базы данных" in names
        assert len(names) == 2

    def test_compute_relevance_no_model(self):
        scorer = DisciplineAwareScorer()
        result = scorer.compute_relevance("python", "Базы данных")
        assert result.combined == 0.0

    def test_compute_relevance_no_disciplines(self):
        scorer = DisciplineAwareScorer()
        result = scorer.compute_relevance("python")
        assert result.combined == 0.0
