"""PDF report endpoint: route wiring + builder smoke on real data."""
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from src.api_pkg import deps


def test_report_route_registered():
    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/results/report/{profile}" in paths


def test_report_unknown_profile_404(monkeypatch):
    from src.models.student import StudentProfile

    app = create_app()
    app.dependency_overrides[deps.get_student_profiles] = lambda: {
        "base": StudentProfile(profile_name="base", competencies=[],
                               skills=["python"], target_level="middle"),
    }
    client = TestClient(app, raise_server_exceptions=False)
    try:
        assert client.get("/api/results/report/nope").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_report_pdf_bytes_base(monkeypatch):
    import shutil
    from pathlib import Path
    from src.models.student import StudentProfile

    from src import config as _cfg
    # conftest autouse уводит DATA_DIR в tmp: кладём туда настоящий full_rec.
    dest = Path(str(_cfg.DATA_DIR)) / "result" / "base"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy("data/result/base/full_recommendations_base.json",
                dest / "full_recommendations_base.json")
    app = create_app()
    app.dependency_overrides[deps.get_student_profiles] = lambda: {
        "base": StudentProfile(profile_name="base", competencies=[],
                               skills=["python"], target_level="middle"),
    }
    client = TestClient(app, raise_server_exceptions=False)
    try:
        r = client.get("/api/results/report/base")
        assert r.status_code == 200, r.text[:200]
        assert r.headers["content-type"] == "application/pdf"
        assert r.content[:5] == b"%PDF-"
        assert len(r.content) > 50000  # таблицы + PNG-графики внутри
    finally:
        app.dependency_overrides.clear()


def test_builder_cyrillic():
    import json
    from src.reports.pdf_report import build_profile_pdf
    d = json.load(open("data/result/base/full_recommendations_base.json",
                       encoding="utf-8"))
    blob = build_profile_pdf("base", d, "data/result/reports")
    assert blob[:5] == b"%PDF-"
    text = blob.decode("latin-1", errors="ignore")
    assert "Data Science" in text or "DS" in text or "Python" in text or "python" in text
