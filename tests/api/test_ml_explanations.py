"""ML #78: `/results/ml-explanations/{profile}` - per-skill SHAP-объяснения.

Контракт эндпоинта:
  - гейт require_any_role("admin", "teacher", "rop"): ��ноним -> 401,
    student -> 403 (вкладка «Визуализация» - админская);
  - неизвестный профиль -> 404;
  - unfit-модель -> 200 + message + пустые explanations (graceful);
  - ошибка инференса -> 200 + message + пустые explanations;
  - fitted: топ-10 навыков, вклады отсортированы по |SHAP| по убыванию,
    нулевые вклады отброшены, direction = sign(value), top clamp 1..8,
    строка X выровнена с навыком (feature_value из его же строки).

Реальный LTR-движок (xgboost/shap) не грузится: module-global
results._explanation_engine подменяется impostor-объектом, поэтому тесты
идут без тяжёлых артефактов.
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


# shap/cv2 не установлены в CI, но подтягиваются при импорте app.
class _StubModule(types.ModuleType):
    """Модуль-заглушка: атрибуты - MagicMock, сам объект - ModuleType."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock()


sys.modules.setdefault("shap", _StubModule("shap"))
sys.modules.setdefault("cv2", _StubModule("cv2"))

from src import Err, Ok  # noqa: E402
from src.api_pkg import create_app, deps  # noqa: E402
from src.api_pkg.routers import results  # noqa: E402
from src.models.student import StudentProfile  # noqa: E402
from tests.conftest import route_gate_roles  # noqa: E402

PROFILE = "base"
URL = f"/api/results/ml-explanations/{PROFILE}"
FEATURES = [
    "level_encoded",
    "category_encoded",
    "cosine_sim",
    "in_student_profile",
    "skill_freq_normalized",
    "co_occurrence",
    "category_avg_weight",
    "student_skills_count",
]

# impacts отсортированы по score: kubernetes (30) > docker (20).
_IMPACTS = [
    ("kubernetes", 30.0, "kubernetes: важность 30.0%"),
    ("docker", 20.0, "docker: важность 20.0%"),
]
# Строки SHAP выровнены с impacts: строка 0 - kubernetes, 1 - docker.
_SHAP = np.array(
    [
        [0.15, 0.0, 0.05, 0.0, 0.0, -0.2, 0.4, 0.0],
        [0.3, 0.0, 0.1, 0.0, 0.0, 0.1, 0.2, 0.0],
    ]
)
_X = pd.DataFrame(
    {
        "level_encoded": [2.0, 3.0],
        "category_encoded": [0.0, 0.0],
        "cosine_sim": [0.91, 0.42],
        "in_student_profile": [0.0, 0.0],
        "skill_freq_normalized": [0.8, 0.6],
        "co_occurrence": [0.33, 0.66],
        "category_avg_weight": [0.7, 0.5],
        "student_skills_count": [12.0, 12.0],
    },
    columns=FEATURES,
)


def _fake_engine(*, fitted: bool = True, predict=None, load=None) -> MagicMock:
    engine = MagicMock()
    engine.is_fitted = fitted
    engine.skill_metadata = {
        "docker": {"freq_normalized": 0.6},
        "kubernetes": {"freq_normalized": 0.8},
    }
    if predict is not None:
        engine.predict_skill_impact_with_shap = lambda *a, **kw: predict
    if load is not None:
        engine.load_model = lambda *a, **kw: load
    return engine


def _make_client(monkeypatch, *, role: str | None, engine: MagicMock) -> TestClient:
    from src.api_pkg.routers import auth as _auth

    monkeypatch.setattr(
        deps, "student_profiles", {PROFILE: _student(PROFILE)}
    )
    monkeypatch.setattr(results, "_explanation_engine", engine)

    if role is None:
        async def _no_user(_request):
            return None

        monkeypatch.setattr(_auth, "get_current_user", _no_user)
    else:
        async def _fake_user(_request):
            return {"uid": "u-1", "u": f"{role}@example.com", "r": role, "t": None}

        monkeypatch.setattr(_auth, "get_current_user", _fake_user)

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def _student(name: str) -> StudentProfile:
    return StudentProfile(
        profile_name=name,
        competencies=["UK-1"],
        skills=["python"],
        target_level="middle",
    )


def test_route_gated_admin_teacher_rop() -> None:
    """Статический контракт: только admin/teacher/rop, без student."""
    roles = route_gate_roles(create_app(), "/api/results/ml-explanations/{profile}", "GET")
    assert roles == {("admin", "teacher", "rop")}


def test_anonymous_gets_401(monkeypatch) -> None:
    c = _make_client(monkeypatch, role=None, engine=_fake_engine())
    assert c.get(URL).status_code == 401


def test_student_gets_403(monkeypatch) -> None:
    c = _make_client(
        monkeypatch, role="student", engine=_fake_engine()
    )
    assert c.get(URL).status_code == 403


def test_admin_gets_structured_explanations(monkeypatch) -> None:
    engine = _fake_engine(predict=Ok((_IMPACTS, _SHAP, _X)))
    c = _make_client(monkeypatch, role="admin", engine=engine)

    r = c.get(URL)
    assert r.status_code == 200
    body = r.json()
    assert body["profile"] == PROFILE
    assert body["model"] == "ltr_xgboost_shap"
    assert "message" not in body

    exps = body["explanations"]
    assert [e["skill"] for e in exps] == ["kubernetes", "docker"]
    assert exps[0]["score"] == 30.0
    assert exps[0]["explanation"].startswith("kubernetes")

    # Kubernetes, строка 0: вклады по |SHAP| убыванию, нули отброшены
    # (4 ненулевых при дефолтном top=5).
    contribs = exps[0]["contributions"]
    assert [c_["feature"] for c_ in contribs] == [
        "category_avg_weight",
        "co_occurrence",
        "level_encoded",
        "cosine_sim",
    ]
    magnitudes = [abs(c_["value"]) for c_ in contribs]
    assert magnitudes == sorted(magnitudes, reverse=True)
    # direction = sign(value) и точность округления.
    by_feature = {c_["feature"]: c_ for c_ in contribs}
    assert by_feature["co_occurrence"]["direction"] == "negative"
    assert by_feature["co_occurrence"]["value"] == -0.2
    assert by_feature["category_avg_weight"]["direction"] == "positive"
    # feature_value взят из ТОЙ ЖЕ строки X (выравнив��ние impacts[i] - X.iloc[i]).
    assert by_feature["co_occurrence"]["feature_value"] == 0.33


def test_top_query_param_limits_contributions(monkeypatch) -> None:
    engine = _fake_engine(predict=Ok((_IMPACTS, _SHAP, _X)))
    c = _make_client(monkeypatch, role="admin", engine=engine)

    r = c.get(f"{URL}?top=1")
    assert r.status_code == 200
    contribs = r.json()["explanations"][0]["contributions"]
    assert len(contribs) == 1
    assert contribs[0]["feature"] == "category_avg_weight"
    assert contribs[0]["value"] == 0.4


def test_top_param_out_of_range_is_422(monkeypatch) -> None:
    engine = _fake_engine(predict=Ok((_IMPACTS, _SHAP, _X)))
    c = _make_client(monkeypatch, role="admin", engine=engine)
    assert c.get(f"{URL}?top=9").status_code == 422


def test_unknown_profile_404(monkeypatch) -> None:
    c = _make_client(monkeypatch, role="admin", engine=_fake_engine())
    assert c.get("/api/results/ml-explanations/nobody").status_code == 404


def test_unfitted_model_returns_graceful_message(monkeypatch) -> None:
    engine = _fake_engine(fitted=False, load=Err(MagicMock(message="no file")))
    c = _make_client(monkeypatch, role="admin", engine=engine)

    r = c.get(URL)
    assert r.status_code == 200
    body = r.json()
    assert body["explanations"] == []
    assert "message" in body


def test_full_coverage_profile_returns_empty(monkeypatch) -> None:
    """Профиль со всеми навыками рынка: missing пуст, объяснений нет."""
    engine = _fake_engine()
    cp = _client_with_skills(
        monkeypatch, role="teacher", engine=engine, skills=["python", "docker", "kubernetes"]
    )
    r = cp.get(URL)
    assert r.status_code == 200
    body = r.json()
    assert body["explanations"] == []
    assert "покрывает" in body["message"]


def _client_with_skills(monkeypatch, *, role: str, engine: MagicMock, skills: list[str]) -> TestClient:
    from src.api_pkg.routers import auth as _auth

    monkeypatch.setattr(deps, "student_profiles", {PROFILE: StudentProfile(
        profile_name=PROFILE, competencies=[], skills=skills, target_level="middle",
    )})
    monkeypatch.setattr(results, "_explanation_engine", engine)

    async def _fake_user(_request):
        return {"uid": "u-2", "u": f"{role}@example.com", "r": role, "t": None}

    monkeypatch.setattr(_auth, "get_current_user", _fake_user)
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def test_predict_error_returns_graceful_message(monkeypatch) -> None:
    engine = _fake_engine(predict=Err(MagicMock(message="boom")))
    c = _make_client(monkeypatch, role="teacher", engine=engine)

    r = c.get(URL)
    assert r.status_code == 200
    body = r.json()
    assert body["explanations"] == []
    assert "message" in body
