"""Покрытие роутера results: сводка, рекомендации, ML-объяснения, отчёт, PNG.

Ни один из 8 GET-эндпоинтов роутера не имел HTTP-тестов (backlog #49):
tests/api/test_results_cache.py гоняет только кэш _read_json. Данные не
дёргаем: load/build-слои подменяем на res.*-атрибутах (имена из пространства
results), профили - через dependency_overrides[deps.get_student_profiles],
REPORTS_DIR (его автофикстура conftest не подменяет) - monkeypatch-ем
в тестах картинок.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.results as res
import src.reports.pdf_report as pdf_mod
from src import Err, Ok, config
from src.api_pkg import create_app, deps
from src.api_pkg.user_keys import self_profile_keys
from src.models.student import StudentProfile
from tests.conftest import open_all_gates

PNG_BYTES = b"\x89PNG\x0d\x0a\x1a\x0a-fake-bytes"


@pytest.fixture(autouse=True)
def _reset_json_cache():
    """Кэш _read_json в results - модульный: чистим до и после каждого теста."""
    res._reset_results_cache()
    yield
    res._reset_results_cache()


def _sp(name, skills=("Python",)) -> StudentProfile:
    return StudentProfile(profile_name=name, skills=list(skills))


def _client(monkeypatch, *, role="teacher", profiles=None) -> TestClient:
    """Клиент с открытыми гейтами: role уходит в fake-юзер каждого гейта."""
    app = create_app()
    open_all_gates(app, role=role)
    if profiles is not None:
        app.dependency_overrides[deps.get_student_profiles] = lambda: profiles
    for lim in (getattr(app.state, "limiter", None), getattr(res, "limiter", None)):
        if lim is not None:
            lim.reset()
    return TestClient(app, raise_server_exceptions=False)


def _student_client(monkeypatch, *, username="student@example.com", profiles=None) -> TestClient:
    """Реальные гейты + get_current_user -> студент.

    Гейт результатов пускает студента, поэтому open_all_gates не нужен:
    проверяем именно студенческие ветки. Патч достаёт и до
    _require_student_profile - он импортирует get_current_user внутри тела
    и видит подмену.
    """
    app = create_app()
    if profiles is not None:
        app.dependency_overrides[deps.get_student_profiles] = lambda: profiles

    async def _student(_request):
        return {"uid": "u-2", "u": username, "r": "student", "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _student)
    for lim in (getattr(app.state, "limiter", None), getattr(res, "limiter", None)):
        if lim is not None:
            lim.reset()
    return TestClient(app, raise_server_exceptions=False)


def _write_rec(profile: str) -> Path:
    """Файл рекомендаций профиля во tmp-каталоге (фикстура подменяет DATA_DIR)."""
    path = config.DATA_DIR / "result" / profile / f"full_recommendations_{profile}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profile": profile, "recommendations": [{"skill": "sql"}]}), encoding="utf-8")
    return path


class _FakeEngine:
    """Что нужно эндпоинту: is_fitted, словарь skill_metadata, predict -> Status."""

    def __init__(self, *, metadata=None, predict_result=None):
        self.is_fitted = True
        self.skill_metadata = metadata or {}
        self._predict_result = predict_result
        self.calls: list[tuple] = []

    def predict_skill_impact_with_shap(self, student_skills, missing, *_):
        self.calls.append((tuple(student_skills), tuple(missing)))
        return self._predict_result


# ------------------------------------------------------------- гейты (401)

@pytest.mark.parametrize(
    "url",
    [
        "/api/results/summary",
        "/api/results/recommendations/alpha",
        "/api/results/ml-explanations/alpha",
        "/api/results/report/alpha",
        "/api/results/images/alpha/radar",
        "/api/results/images/coverage-comparison",
    ],
)
def test_results_endpoints_require_auth(url):
    """Без токена каждый эндпоинт результатов - 401, до тела ручки не доходим."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(url)
    assert r.status_code == 401, (url, r.text)


# ------------------------------------------------------------- /results/summary


def test_summary_teacher_gets_all_profiles(monkeypatch):
    monkeypatch.setattr(res, "load_recommendations_from_disk", lambda: "LOADED-REC")
    seen = []

    def _build(recs):
        seen.append(recs)
        return {"base": {"score": 1}, "self_test": {"score": 2}}

    monkeypatch.setattr(res, "build_summary_payload", _build)
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/results/summary")
    assert r.status_code == 200, r.text
    body = r.json()
    # build получает именно результат load_recommendations_from_disk
    assert seen == ["LOADED-REC"]
    assert body["profiles"] == ["base", "self_test"]
    assert set(body["evaluations"]) == {"base", "self_test"}
    assert body["self_profile"] is None


def test_summary_student_gets_only_self(monkeypatch):
    monkeypatch.setattr(res, "load_recommendations_from_disk", lambda: None)
    monkeypatch.setattr(
        res, "build_summary_payload", lambda recs: {"base": {"score": 1}, "self_test": {"score": 2}}
    )
    client = _student_client(monkeypatch, username="test")
    r = client.get("/api/results/summary")
    assert r.status_code == 200, r.text
    body = r.json()
    assert self_profile_keys("test") == {"self_test"}
    assert body["profiles"] == ["self_test"]
    assert set(body["evaluations"]) == {"self_test"}
    assert body["self_profile"] == "self_test"


def test_summary_student_without_self_key_gets_message(monkeypatch):
    """Своего профиля нет: архетипы студенту не светят, evaluations пусты."""
    monkeypatch.setattr(res, "load_recommendations_from_disk", lambda: None)
    monkeypatch.setattr(res, "build_summary_payload", lambda recs: {"base": {"score": 1}})
    client = _student_client(monkeypatch, username="test")
    r = client.get("/api/results/summary")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "message" in body
    assert body["profiles"] == []
    assert "evaluations" not in body


def test_summary_fallback_to_stored_file(monkeypatch):
    monkeypatch.setattr(res, "load_recommendations_from_disk", lambda: [])
    monkeypatch.setattr(res, "build_summary_payload", lambda recs: {})
    stored = {"evaluations": {"base": {"score": 1}}, "profiles": ["base"], "version": 2}
    (config.DATA_RESULT_DIR / "profiles_comparison_summary.json").write_text(json.dumps(stored), encoding="utf-8")
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/results/summary")
    assert r.status_code == 200, r.text
    assert r.json() == stored


def test_summary_corrupt_stored_file_500(monkeypatch):
    monkeypatch.setattr(res, "load_recommendations_from_disk", lambda: [])
    monkeypatch.setattr(res, "build_summary_payload", lambda recs: {})
    (config.DATA_RESULT_DIR / "profiles_comparison_summary.json").write_text("{not-json", encoding="utf-8")
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/results/summary")
    assert r.status_code == 500
    # не-админ получает безопасное сообщение, а не техдетали об ошибке
    assert "Попробуйте позже" in r.json()["detail"]


# ---------------------------------------------- /results/recommendations/{p}


def test_recommendations_returns_saved_payload(monkeypatch):
    path = _write_rec("alpha")
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/recommendations/alpha")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recommendations"] == [{"skill": "sql"}]
    # generated_at перезаписывается mtime файла, «old»-заглушек не осталось
    assert "generated_at" in body
    assert path.exists()  # данные на месте, кэш их не мутировал


def test_recommendations_unknown_profile_404(monkeypatch):
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/recommendations/zeta")
    assert r.status_code == 404
    assert r.json()["detail"] == "Профиль не найден"


def test_recommendations_missing_file_returns_stub(monkeypatch):
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/recommendations/alpha")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["profile"] == "alpha"
    assert body["recommendations"] == []
    assert "message" in body


def test_recommendations_student_foreign_profile_403(monkeypatch):
    client = _student_client(monkeypatch, profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/recommendations/alpha")
    assert r.status_code == 403, r.text
    assert r.json()["detail"] == "Доступ к чужому профилю запрещён"


def test_recommendations_student_own_key_passes_guard(monkeypatch):
    """Свой self_<username>-ключ гейт пропускает: студент получает 404
    «профиля нет», а не 403 «чужой профиль»."""
    own = next(iter(self_profile_keys("student@example.com")))
    assert own == "self_student_example_com"
    client = _student_client(monkeypatch, profiles={})
    r = client.get(f"/api/results/recommendations/{own}")
    assert r.status_code == 404
    assert r.json()["detail"] == "Профиль не найден"


# --------------------------------------------- /results/ml-explanations/{p}


def test_ml_explanations_unknown_profile_404(monkeypatch):
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/ml-explanations/zeta")
    assert r.status_code == 404
    assert r.json()["detail"] == "Профиль не найден"


def test_ml_explanations_student_forbidden_403(monkeypatch):
    """Гейт только admin/teacher/rop: студент не проходит роль."""
    client = _student_client(monkeypatch, profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/ml-explanations/alpha")
    assert r.status_code == 403


def test_ml_explanations_unfitted_engine_graceful(monkeypatch):
    monkeypatch.setattr(
        res, "_get_or_load_explanation_engine", lambda: SimpleNamespace(is_fitted=False)
    )
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/ml-explanations/alpha")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["explanations"] == []
    assert "не обучена" in body["message"]


def test_ml_explanations_all_covered(monkeypatch):
    engine = _FakeEngine(metadata={"Python": {"freq_normalized": 2.0}, "SQL": {"freq_normalized": 1.0}})
    monkeypatch.setattr(res, "_get_or_load_explanation_engine", lambda: engine)
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha", skills=["Python", "SQL"])})
    r = client.get("/api/results/ml-explanations/alpha")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["message"] == "Профиль уже покрывает все навыки рынка."
    assert body["explanations"] == []
    assert engine.calls == []  # инференс не дёргаем, если недостающих нет


def test_ml_explanations_shap_contributions(monkeypatch):
    x = SimpleNamespace(columns=["f1", "f2"], iloc=[{"f1": 1.5, "f2": "not-a-number"}])
    engine = _FakeEngine(
        metadata={"Python": {"freq_normalized": 2.0}, "rust": {"freq_normalized": 0.5}},
        predict_result=Ok(([("rust", 0.42, "кандидат из топа рынка")], [[0.5, -0.25]], x)),
    )
    monkeypatch.setattr(res, "_get_or_load_explanation_engine", lambda: engine)
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha", skills=["Python"])})
    r = client.get("/api/results/ml-explanations/alpha")
    assert r.status_code == 200, r.text
    body = r.json()
    # predict получил только недостающие навыки
    assert engine.calls == [(("Python",), ("rust",))]
    assert [e["skill"] for e in body["explanations"]] == ["rust"]
    top = body["explanations"][0]
    assert top["score"] == 0.42
    assert top["explanation"] == "кандидат из топа рынка"
    contribs = top["contributions"]
    # сортировка по |SHAP|-вкладу: f1 (0.5) -> f2 (0.25)
    assert [c["feature"] for c in contribs] == ["f1", "f2"]
    assert contribs[0]["value"] == 0.5
    assert contribs[0]["direction"] == "positive"
    assert contribs[0]["feature_value"] == 1.5
    # нечисловое значение столбца не роняет объяснение
    assert contribs[1]["value"] == -0.25
    assert contribs[1]["direction"] == "negative"
    assert contribs[1]["feature_value"] is None


def test_ml_explanations_predict_err_graceful(monkeypatch):
    engine = _FakeEngine(
        metadata={"Python": {"freq_normalized": 1.0}, "rust": {"freq_normalized": 0.5}},
        predict_result=Err("predict failed"),
    )
    monkeypatch.setattr(res, "_get_or_load_explanation_engine", lambda: engine)
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha", skills=["Python"])})
    r = client.get("/api/results/ml-explanations/alpha")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["message"] == "Не удалось посчитать объяснения ML-модели."
    assert body["explanations"] == []


@pytest.mark.parametrize("top", [0, 9])
def test_ml_explanations_top_out_of_range_422(monkeypatch, top):
    """Граница: top ограничен ge=1 le=8 - за границей валидация даёт 422."""
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get(f"/api/results/ml-explanations/alpha?top={top}")
    assert r.status_code == 422


# ---------------------------------------------------- /results/report/{p}


def _patch_pdf_builder(monkeypatch, fn):
    """build_profile_pdf импортируется в ручке - патчим его модуль."""
    monkeypatch.setattr(pdf_mod, "build_profile_pdf", fn)


def test_report_pdf_happy(monkeypatch):
    _write_rec("alpha")
    seen = []

    def _fake_pdf(profile, payload, reports_dir):
        seen.append((profile, payload, reports_dir))
        return b"%PDF-fake"

    _patch_pdf_builder(monkeypatch, _fake_pdf)
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/report/alpha")
    assert r.status_code == 200, r.text
    assert r.content == b"%PDF-fake"
    assert r.headers["content-type"].startswith("application/pdf")
    assert "attachment" in r.headers["content-disposition"]
    assert "report_alpha.pdf" in r.headers["content-disposition"]
    profile, payload, reports_dir = seen[0]
    assert profile == "alpha"
    assert payload["profile"] == "alpha"
    assert Path(reports_dir) == config.REPORTS_DIR


def test_report_unknown_profile_404(monkeypatch):
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/report/zeta")
    assert r.status_code == 404
    assert r.json()["detail"] == "Профиль не найден"


def test_report_no_recommendations_404(monkeypatch):
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/report/alpha")
    assert r.status_code == 404
    assert r.json()["detail"].startswith("Рекомендации не найдены")


def test_report_builder_failure_500(monkeypatch):
    _write_rec("alpha")

    def _boom(profile, payload, reports_dir):
        raise RuntimeError("pdf fail")

    _patch_pdf_builder(monkeypatch, _boom)
    client = _client(monkeypatch, role="teacher", profiles={"alpha": _sp("alpha")})
    r = client.get("/api/results/report/alpha")
    assert r.status_code == 500
    assert "Попробуйте позже" in r.json()["detail"]


# --------------------------------------------- /results/images/... (FROZEN v1)


def test_profile_image_happy_200(monkeypatch):
    path = config.DATA_DIR / "result" / "alpha" / "radar_alpha.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_BYTES)
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/results/images/alpha/radar")
    assert r.status_code == 200, r.text
    assert r.content == PNG_BYTES
    assert r.headers["content-type"].startswith("image/png")


def test_profile_image_fallback_to_result_root(monkeypatch):
    """Файла в подкаталоге профиля нет - берём result/<img>_<profile>.png."""
    path = config.DATA_DIR / "result" / "radar_alpha.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_BYTES)
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/results/images/alpha/radar")
    assert r.status_code == 200, r.text
    assert r.content == PNG_BYTES


def test_profile_image_invalid_name_400(monkeypatch):
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/results/images/bad.name/radar")
    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid profile name"


def test_profile_image_invalid_type_400(monkeypatch):
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/results/images/alpha/bogus")
    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid image type"


def test_profile_image_missing_404(monkeypatch):
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/results/images/alpha/radar")
    assert r.status_code == 404
    assert "radar_alpha.png" in r.json()["detail"]


def test_profile_image_outside_result_dir_400(monkeypatch, tmp_path):
    """Нашлось в REPORTS_DIR вне data/result - защита от выхода из каталога: 400."""
    reports = tmp_path / "reports"
    image = reports / "alpha" / "radar_alpha.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(PNG_BYTES)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    client = _client(monkeypatch, role="teacher")
    r = client.get("/api/results/images/alpha/radar")
    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid image path"


@pytest.mark.parametrize(
    ("url", "filename"),
    [
        ("coverage-comparison", "coverage_comparison.png"),
        ("skills-heatmap", "skills_heatmap.png"),
        ("skill-correlation", "skill_correlation_heatmap.png"),
    ],
)
def test_public_images_missing_404_then_200(monkeypatch, tmp_path, url, filename):
    """Публичные PNG из REPORTS_DIR: 404, пока файла нет; 200, когда появился."""
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    client = _client(monkeypatch, role="teacher")
    assert client.get(f"/api/results/images/{url}").status_code == 404
    (reports / filename).write_bytes(PNG_BYTES)
    r = client.get(f"/api/results/images/{url}")
    assert r.status_code == 200, r.text
    assert r.content == PNG_BYTES
    assert r.headers["content-type"].startswith("image/png")
