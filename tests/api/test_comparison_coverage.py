"""Покрытие роутера comparison: сравнение наборов двух лет (backlog #49).

GET /teacher/intake-comparison не имел ни одного теста: grep по tests/ не находит
ни эндпоинта, ни импорта модуля. Скрипт scripts/compare_intakes.py грузится в
ручке через importlib, реальные данные не читаем - модуль скрипта подменяем
фейком через comparison._compare_module (это кэш _load_compare: пока он не
None, загрузчик не трогает файловую систему). Сам _load_compare проверяем на
настоящем скрипте: его импорт без побочных эффектов (guard
``if __name__ == "__main__"`` в scripts/compare_intakes.py:344).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.comparison as cmp_mod
from src.api_pkg import create_app
from tests.conftest import open_all_gates

REPORT = {
    "direction": "09.03.02",
    "old_year": 2024,
    "new_year": 2025,
    "summary": {"added": 1, "removed": 0, "kept": 2},
    "added": [{"name": "Машинное обучение"}],
}


class _FakeCompare:
    """Модуль compare_intakes: build_intake_report + IntakeNotFoundError."""

    def __init__(self, *, result=None, raise_not_found: bool = False):
        self.IntakeNotFoundError = type("IntakeNotFoundError", (Exception,), {})
        self.calls: list[tuple] = []
        self._result = result
        self._raise_not_found = raise_not_found

    def build_intake_report(self, direction, old_year, new_year, base=None):
        self.calls.append((direction, old_year, new_year))
        if self._raise_not_found:
            raise self.IntakeNotFoundError("старый", old_year, direction)
        return self._result


def _client(monkeypatch, fake) -> TestClient:
    """Приложение с открытыми гейтами и подсунутым модулем compare_intakes."""
    app = create_app()
    open_all_gates(app, role="admin")
    monkeypatch.setattr(cmp_mod, "_compare_module", fake)
    return TestClient(app, raise_server_exceptions=False)


# ----------------------------------------------------- happy path


def test_intake_comparison_returns_report(monkeypatch):
    fake = _FakeCompare(result=REPORT)
    client = _client(monkeypatch, fake)
    r = client.get("/api/teacher/intake-comparison?dir_code=09.03.02&old=2024&new=2025")
    assert r.status_code == 200, r.text
    assert r.json() == REPORT
    assert fake.calls == [("09.03.02", 2024, 2025)]


def test_intake_comparison_defaults_boundary(monkeypatch):
    """Граница: без query-параметров ручка идёт к дефолтам 09.03.02 / 2024 / 2025."""
    fake = _FakeCompare(result={"summary": {"added": 0}})
    client = _client(monkeypatch, fake)
    r = client.get("/api/teacher/intake-comparison")
    assert r.status_code == 200, r.text
    assert fake.calls == [("09.03.02", 2024, 2025)]


# ----------------------------------------------------- ошибки (точный код)


def test_intake_comparison_missing_intake_404(monkeypatch):
    """Нет сводки за год: 404, и текст исключения наружу не отдаётся."""
    fake = _FakeCompare(raise_not_found=True)
    client = _client(monkeypatch, fake)
    r = client.get("/api/teacher/intake-comparison?dir_code=09.03.02&old=2024&new=2025")
    assert r.status_code == 404, r.text
    detail = r.json()["detail"]
    assert "09.03.02" in detail and "2024" in detail and "2025" in detail, detail


# ----------------------------------------------------- гейты (real gates)


def test_intake_comparison_requires_auth_401():
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get("/api/teacher/intake-comparison")
    assert r.status_code == 401, r.text


def test_intake_comparison_student_forbidden_403(monkeypatch):
    """Гейт admin/teacher/rop: student получает 403, а не отчёт."""

    async def _student(_request):
        return {"uid": "u-1", "u": "student@example.com", "r": "student", "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _student)
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get("/api/teacher/intake-comparison")
    assert r.status_code == 403, r.text


# ----------------------------------------------------- _load_compare


def test_load_compare_returns_and_caches_real_script(monkeypatch):
    """Реальная загрузка scripts/compare_intakes.py + кэш на второй вызов."""
    monkeypatch.setattr(cmp_mod, "_compare_module", None)
    first = cmp_mod._load_compare()
    assert callable(first.build_intake_report)
    assert issubclass(first.IntakeNotFoundError, Exception)
    assert cmp_mod._load_compare() is first
