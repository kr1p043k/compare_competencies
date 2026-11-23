"""Сквозная проверка: роль меняет веса навыков через смесь источников.

Кластерные модели не загружаются (patch load_model) - тест проверяет только
проводку весов в GapAnalyzer и не зависит от артефактов на диске.
"""

from __future__ import annotations

import pytest

from src.analyzers.gap import profile_evaluator as pe
from src.analyzers.skills.science_weights import RoleSourceMix
from src.models.student import StudentProfile

WEIGHTS_BY_LEVEL = {
    "junior": {"python": 0.8, "sql": 0.6},
    "middle": {"python": 1.0, "sql": 0.6},
    "senior": {"python": 0.9, "sql": 0.4},
}


@pytest.fixture(autouse=True)
def no_cluster_models(monkeypatch):
    monkeypatch.setattr(pe.VacancyClusterer, "load_model", lambda self, level="all": False)


def _evaluator(source_mix):
    return pe.ProfileEvaluator(
        skill_weights={"python": 1.0, "sql": 0.6},
        vacancies_skills=[["python", "sql"]],
        vacancies_skills_dict=[{"skills": ["python", "sql"], "experience": "middle"}],
        skill_weights_by_level=WEIGHTS_BY_LEVEL,
        source_mix=source_mix,
    )


def _student(profession: str) -> StudentProfile:
    return StudentProfile(
        profile_name=f"p_{profession or 'none'}",
        skills=["git"],
        target_level="middle",
        target_profession=profession,
    )


MIX = RoleSourceMix(
    article_weights={"data scientist": {"python": 1.0, "sql": 1.0}},
    role_lambdas={"data scientist": 0.5},
    default_lambda=0.0,
    source="test",
)


def test_science_role_gets_higher_demand_than_market_role():
    evaluator = _evaluator(MIX)
    ds = evaluator.evaluate_profile(_student("Data Scientist")).unwrap()
    backend = evaluator.evaluate_profile(_student("Backend Developer")).unwrap()

    ds_sql = ds["skill_metrics"]["sql"]["demand_m"]
    backend_sql = backend["skill_metrics"]["sql"]["demand_m"]
    # DS: 0.5*0.6 + 0.5*1.0 = 0.8; backend: lambda 0 -> рыночные 0.6
    assert ds_sql == pytest.approx(0.8)
    assert backend_sql == pytest.approx(0.6)
    assert ds_sql > backend_sql
    # python и рынок, и наука = 1.0 -> роль его не двигает
    assert ds["skill_metrics"]["python"]["demand_m"] == pytest.approx(1.0)


def test_disabled_mix_reproduces_market_numbers():
    evaluator = _evaluator(RoleSourceMix.disabled())
    ds = evaluator.evaluate_profile(_student("Data Scientist")).unwrap()
    assert ds["skill_metrics"]["sql"]["demand_m"] == pytest.approx(0.6)
    assert ds["skill_metrics"]["python"]["demand_m"] == pytest.approx(1.0)


def test_unknown_role_uses_default_lambda():
    mix = RoleSourceMix(
        article_weights={"global-ish": {"sql": 1.0}},
        global_weights={"sql": 1.0},
        role_lambdas={},
        default_lambda=0.25,
        source="test",
    )
    evaluator = _evaluator(mix)
    out = evaluator.evaluate_profile(_student("Unknown Profession")).unwrap()
    # 0.25 научной половины: 0.6*0.75 + 1.0*0.25 = 0.7
    assert out["skill_metrics"]["sql"]["demand_m"] == pytest.approx(0.7)
