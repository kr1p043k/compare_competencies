"""AuthZ for profiles/results IDOR fixes.

Покрывает 4 находки аудита:
  1. /profiles/compare - раньше без auth вовсе.
  2. /profiles/{profile} - анонимно, утечка навыков любого студента.
  3. /profiles/{profile}/profession-evaluation - анонимно.
  4. /results/recommendations/{profile} и /results/report/{profile} -
     student допускался к любому профилю (нет проверки владения).

Контракт каждого роута: аноним -> 401, student на чужой профиль -> 403,
student на свой -> 200, admin -> 200.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

# shap/cv2 не установлены в CI, но подтягиваются при импорте app.
sys.modules["shap"] = MagicMock()
sys.modules["cv2"] = MagicMock()

import src.analyzers.skills.profession_taxonomy as prof_tax_mod  # noqa: E402
from src import Ok  # noqa: E402
from src.api_pkg import create_app, deps  # noqa: E402
from src.models.student import StudentProfile  # noqa: E402

SELF_EMAIL = "test.student@example.com"
SELF_NAME = "self_test_student_example_com"
OTHER = "base"


def _profile(name: str) -> StudentProfile:
    return StudentProfile(
        profile_name=name,
        competencies=["UK-1"],
        skills=["python", "sql"],
        target_level="middle",
    )


class _FakeTaxonomy:
    def __init__(self, *args, **kwargs):
        pass

    def get_profile_target(self, profile_name: str) -> dict:
        return {"target_profession": "Data Scientist", "target_domains": ["data"]}

    def get_profession_info(self, profession_name: str) -> dict | None:
        return {"domains": ["data"]}


def _make_client(monkeypatch, *, role: str | None, email: str | None = None) -> TestClient:
    """Client с подменённым пользователем и заглушками тяжёлых зависимостей.

    role=None - анонимный запрос (get_current_user -> None).
    """
    from src.api_pkg.routers import auth as _auth

    evaluator = MagicMock()
    evaluator.evaluate_profile.return_value = Ok(
        {
            "market_coverage_score": 70,
            "skill_coverage": 60,
            "domain_coverage_score": 50,
            "readiness_score": 65,
            "market_skill_coverage": 40,
            "profession_coverage": 80,
            "krm_coverage": {},
        }
    )
    monkeypatch.setattr(deps, "evaluator", evaluator)
    monkeypatch.setattr(deps, "recommendation_engine", MagicMock())
    monkeypatch.setattr(deps, "student_profiles", {OTHER: _profile(OTHER), SELF_NAME: _profile(SELF_NAME)})
    monkeypatch.setattr(prof_tax_mod, "ProfessionTaxonomy", _FakeTaxonomy)

    if role is None:
        async def _no_user(_request):
            return None

        monkeypatch.setattr(_auth, "get_current_user", _no_user)
    else:
        async def _fake_user(_request):
            return {"uid": "u-1", "u": email or f"{role}@example.com", "r": role, "t": None}

        monkeypatch.setattr(_auth, "get_current_user", _fake_user)

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------- FINDING 1
def test_compare_anon_401(monkeypatch):
    c = _make_client(monkeypatch, role=None)
    assert c.get("/api/profiles/compare").status_code == 401


def test_compare_student_200(monkeypatch):
    c = _make_client(monkeypatch, role="student")
    assert c.get("/api/profiles/compare").status_code == 200


def test_compare_admin_200(monkeypatch):
    c = _make_client(monkeypatch, role="admin")
    assert c.get("/api/profiles/compare").status_code == 200


# ---------------------------------------------------------------- FINDING 2
def test_get_profile_anon_401(monkeypatch):
    c = _make_client(monkeypatch, role=None)
    assert c.get(f"/api/profiles/{OTHER}").status_code == 401


def test_get_profile_student_other_403(monkeypatch):
    c = _make_client(monkeypatch, role="student", email=SELF_EMAIL)
    assert c.get(f"/api/profiles/{OTHER}").status_code == 403


def test_get_profile_student_self_200(monkeypatch):
    c = _make_client(monkeypatch, role="student", email=SELF_EMAIL)
    assert c.get(f"/api/profiles/{SELF_NAME}").status_code == 200


def test_get_profile_admin_200(monkeypatch):
    c = _make_client(monkeypatch, role="admin")
    assert c.get(f"/api/profiles/{OTHER}").status_code == 200


# ---------------------------------------------------------------- FINDING 3
def test_profession_eval_anon_401(monkeypatch):
    c = _make_client(monkeypatch, role=None)
    assert c.get(f"/api/profiles/{OTHER}/profession-evaluation").status_code == 401


def test_profession_eval_student_other_403(monkeypatch):
    c = _make_client(monkeypatch, role="student", email=SELF_EMAIL)
    assert c.get(f"/api/profiles/{OTHER}/profession-evaluation").status_code == 403


def test_profession_eval_student_self_200(monkeypatch):
    c = _make_client(monkeypatch, role="student", email=SELF_EMAIL)
    assert c.get(f"/api/profiles/{SELF_NAME}/profession-evaluation").status_code == 200


def test_profession_eval_admin_200(monkeypatch):
    c = _make_client(monkeypatch, role="admin")
    assert c.get(f"/api/profiles/{OTHER}/profession-evaluation").status_code == 200


# ---------------------------------------------------------------- FINDING 4
def test_recommendations_anon_401(monkeypatch):
    c = _make_client(monkeypatch, role=None)
    assert c.get(f"/api/results/recommendations/{OTHER}").status_code == 401


def test_recommendations_student_other_403(monkeypatch):
    c = _make_client(monkeypatch, role="student", email=SELF_EMAIL)
    assert c.get(f"/api/results/recommendations/{OTHER}").status_code == 403


def test_recommendations_student_self_200(monkeypatch):
    c = _make_client(monkeypatch, role="student", email=SELF_EMAIL)
    assert c.get(f"/api/results/recommendations/{SELF_NAME}").status_code == 200


def test_recommendations_admin_200(monkeypatch):
    c = _make_client(monkeypatch, role="admin")
    assert c.get(f"/api/results/recommendations/{OTHER}").status_code == 200


def test_report_anon_401(monkeypatch):
    c = _make_client(monkeypatch, role=None)
    assert c.get(f"/api/results/report/{OTHER}").status_code == 401


def test_report_student_other_403(monkeypatch):
    c = _make_client(monkeypatch, role="student", email=SELF_EMAIL)
    assert c.get(f"/api/results/report/{OTHER}").status_code == 403


def _prepare_report(monkeypatch, tmp_path: Path, profile: str) -> None:
    from src import config

    result_dir = Path(config.DATA_DIR) / "result" / profile
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"full_recommendations_{profile}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "src.reports.pdf_report.build_profile_pdf",
        lambda *args, **kwargs: b"%PDF-1.4-authz-test",
    )


def test_report_student_self_200(monkeypatch, tmp_path):
    _prepare_report(monkeypatch, tmp_path, SELF_NAME)
    c = _make_client(monkeypatch, role="student", email=SELF_EMAIL)
    r = c.get(f"/api/results/report/{SELF_NAME}")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/pdf")


def test_report_admin_200(monkeypatch, tmp_path):
    _prepare_report(monkeypatch, tmp_path, OTHER)
    c = _make_client(monkeypatch, role="admin")
    r = c.get(f"/api/results/report/{OTHER}")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/pdf")
