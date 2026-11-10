"""Покрытие роутера krm_teacher: реальные загрузчики KRM и gap-анализ (backlog #49).

test_coverage_krmteacher_history.py закрыл authz трёх ручек, но СТАБАМИ
_list_all_dirs / _load_krm_direction - тела загрузчиков не исполнялись, а
disciplines / disciplines/{name} / gap не имели тестов вообще. Этот файл добивает:

- _list_all_dirs и _load_krm_direction по-настоящему: справочник - реальные
  JSON-файлы в tmp_path (config.REFERENCE_DIR -> tmp, настоящий data/reference
  не читаем и не трогаем), включая ветки _clean / мусорный JSON / не-словарь /
  пустой файл / пропавший каталог;
- привязки user_directions - fake-пул (в БД не ходим): rop/teacher с кодами
  привязок. Pool-код не обязан существовать в справочнике - так HTTP-ом
  достижимы 404 отсутствующего файла и 404 _clean-файла;
- GET .../disciplines и .../disciplines/{name}: счётчики, exact-match,
  fuzzy (регистр/пробелы), 404;
- POST .../gap: анализатор - фейковый модуль в sys.modules (ленивый импорт в
  ручке подхватывает его первым), LLM-ветка - так же через фейковый
  src.services.llm_recommend и флаги config.LLM_ENABLED / LLM_ENHANCE_TEACHER;
  403 по роли (гейт) и 403 по чужой привязке.

Подменяется ТОЛЬКО get_current_user (гейты и authz-хелперы остаются живыми).
"""

from __future__ import annotations

import contextlib
import json
import sys
import types

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.krm_teacher as krm_mod
from src import config
from src.api_pkg import create_app

GOOD = "09.03.02"
OTHER = "01.03.02"

GOOD_ENTRY = {
    "dir_code": GOOD,
    "name": "Информационные системы и технологии",
    "profile": "Разработка ИС",
    "disciplines_count": 2,
}
OTHER_ENTRY = {
    "dir_code": OTHER,
    "name": "Прикладная информатика",
    "profile": "Профильный",
    "disciplines_count": 1,
}

# содержимое krm_disciplines_09.03.02.json: первый value и есть данные направления
GOOD_DATA = {
    GOOD: {
        "direction_name": GOOD_ENTRY["name"],
        "profile": GOOD_ENTRY["profile"],
        "disciplines": {
            "Архитектура ПО": {
                "competencies": ["ОПК-3"],
                "skills": {"ОПК-3": ["диаграммы"]},
            },
            "Базы данных": {
                "competencies": ["ОПК-1", "ОПК-2"],
                "skills": {"ОПК-1": ["SQL", "join"], "ОПК-2": ["index"]},
            },
        },
    }
}
OTHER_DATA = {
    OTHER: {
        "direction_name": OTHER_ENTRY["name"],
        "profile": OTHER_ENTRY["profile"],
        "disciplines": {
            "Аналитика данных": {
                "competencies": ["ОПК-9"],
                "skills": {"ОПК-9": ["BI"]},
            },
        },
    }
}

DB_SKILLS = [
    {"code": "ОПК-1", "skills": ["SQL", "join"]},
    {"code": "ОПК-2", "skills": ["index"]},
]

GAP_RESULT = {
    "summary": "Разрыв по теме SQL",
    "detailed_analysis": [{"competency": "ОПК-1", "gap": 0.3}],
    "recommendations": ["закрыть ОПК-1"],
    "total_gaps": 1,
}


@pytest.fixture(autouse=True)
def _reset_limiter():
    with contextlib.suppress(Exception):
        krm_mod.limiter._storage.reset()
    yield


@pytest.fixture(autouse=True)
def ref_dir(tmp_path, monkeypatch):
    """Справочник KRM каждого теста - пустой tmp: data/reference не читаем."""
    monkeypatch.setattr(config, "REFERENCE_DIR", tmp_path)
    return tmp_path


class _BoundPool:
    """asyncpg-подобный пул: fetch всегда возвращает фиксированные привязки."""

    def __init__(self, codes):
        self._rows = [{"dir_code": code} for code in codes]

    async def fetch(self, _sql, *_args):
        return self._rows


def _write_krm(ref_dir, code, payload):
    (ref_dir / f"krm_disciplines_{code}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _write_raw(ref_dir, code, text):
    (ref_dir / f"krm_disciplines_{code}.json").write_text(text, encoding="utf-8")


def _client(monkeypatch, role="admin", pool=None):
    """Живое приложение; authz-хелперы krm_teacher работают по-настоящему."""
    user = None if role is None else {
        "uid": "u-1", "u": f"{role}@example.com", "r": role, "t": None,
    }

    async def _fake_user(_request):
        return user

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    monkeypatch.setattr(krm_mod, "get_pool", lambda: pool)
    return TestClient(create_app(), raise_server_exceptions=False)


def _install_gap_module(monkeypatch, calls, *, result=None, error=None):
    """Фейковый src.analyzers.academic_gap: ленивый импорт ручки берёт его из sys.modules."""
    class _Analyzer:
        def __init__(self, dir_code=None):
            self.dir_code = dir_code

        def analyze(self, topic):
            calls.append((self.dir_code, topic))
            if error is not None:
                raise error
            return result

    fake_mod = types.ModuleType("src.analyzers.academic_gap")
    fake_mod.AcademicGapAnalyzer = _Analyzer
    monkeypatch.setitem(sys.modules, "src.analyzers.academic_gap", fake_mod)


def _install_llm_module(monkeypatch, calls, *, enhanced=None, error=None):
    """Фейковый src.services.llm_recommend.enhance_teacher_recs."""
    def _enhance(discipline, gaps, base_recs):
        calls.append((discipline, gaps))
        if error is not None:
            raise error
        return enhanced

    fake_mod = types.ModuleType("src.services.llm_recommend")
    fake_mod.enhance_teacher_recs = _enhance
    monkeypatch.setitem(sys.modules, "src.services.llm_recommend", fake_mod)


# ----------------------------------------------------- _list_all_dirs (real)


def test_my_directions_admin_reads_real_reference_files(monkeypatch, ref_dir):
    """Админ видит справочник из файлов; мусор (_clean/код/JSON) отфильтрован."""
    _write_krm(ref_dir, GOOD, GOOD_DATA)
    _write_krm(ref_dir, OTHER, OTHER_DATA)
    _write_raw(ref_dir, f"{GOOD}_clean", "{}")       # _clean-файл пропускается
    _write_raw(ref_dir, "noise", "{}")              # код не матчится на regex
    _write_raw(ref_dir, "", "{}")                   # пустое имя направления
    _write_raw(ref_dir, "09.03.06", "not json at all")  # битый JSON пропускается

    r = _client(monkeypatch, role="admin").get("/api/krm/teacher/my-directions")
    assert r.status_code == 200, r.text
    assert r.json() == [OTHER_ENTRY, GOOD_ENTRY], "сортировка по dir_code"


def test_my_directions_admin_keeps_non_dict_json_with_empty_meta(monkeypatch, ref_dir):
    """JSON-массив вместо словаря: направление остаётся, метаданные пустые."""
    _write_krm(ref_dir, GOOD, GOOD_DATA)
    _write_raw(ref_dir, "09.03.04", "[1, 2]")

    r = _client(monkeypatch, role="admin").get("/api/krm/teacher/my-directions")
    assert r.status_code == 200, r.text
    assert r.json() == [
        GOOD_ENTRY,
        {"dir_code": "09.03.04", "name": "", "profile": "", "disciplines_count": 0},
    ]


def test_list_all_dirs_missing_reference_dir_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REFERENCE_DIR", tmp_path / "no-such-dir")
    assert krm_mod._list_all_dirs() == []


def test_my_directions_rop_sees_only_bound_directions(monkeypatch, ref_dir):
    """rop: только привязки из пула, а не весь справочник."""
    _write_krm(ref_dir, GOOD, GOOD_DATA)
    _write_krm(ref_dir, OTHER, OTHER_DATA)

    r = _client(monkeypatch, role="rop", pool=_BoundPool([GOOD])).get(
        "/api/krm/teacher/my-directions"
    )
    assert r.status_code == 200, r.text
    assert r.json() == [GOOD_ENTRY]


def test_check_access_true_for_bound_code(monkeypatch):
    r = _client(monkeypatch, role="rop", pool=_BoundPool([GOOD])).get(
        f"/api/krm/teacher/check-access?dir_code={GOOD}"
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"access": True, "dir_code": GOOD}


# ------------------------------------------- _load_krm_direction (real)


def test_direction_data_from_real_file(monkeypatch, ref_dir):
    _write_krm(ref_dir, GOOD, GOOD_DATA)

    r = _client(monkeypatch, role="admin").get(f"/api/krm/teacher/directions/{GOOD}")
    assert r.status_code == 200, r.text
    assert r.json() == GOOD_DATA[GOOD]


def test_direction_missing_file_404_for_pool_binding(monkeypatch):
    """Привязка есть, файла нет: 404, а не 403/500 (доступ проверен пулом)."""
    client = _client(monkeypatch, role="teacher", pool=_BoundPool(["09.99.99"]))

    r = client.get("/api/krm/teacher/directions/09.99.99")
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "Direction '09.99.99' not found"


def test_direction_clean_file_404_for_pool_binding(monkeypatch, ref_dir):
    """_clean-файл не отдаётся, даже если код прописан в привязках."""
    _write_raw(ref_dir, f"{GOOD}_clean", "{}")
    client = _client(monkeypatch, role="teacher", pool=_BoundPool([f"{GOOD}_clean"]))

    r = client.get(f"/api/krm/teacher/directions/{GOOD}_clean")
    assert r.status_code == 404, r.text


def test_direction_broken_json_503(monkeypatch, ref_dir):
    """Битый KRM-файл: 503 'данные недоступны', а не падение ручки."""
    _write_raw(ref_dir, GOOD, "{oops")
    client = _client(monkeypatch, role="teacher", pool=_BoundPool([GOOD]))

    r = client.get(f"/api/krm/teacher/directions/{GOOD}")
    assert r.status_code == 503, r.text
    assert r.json()["detail"] == "KRM data unavailable"


@pytest.mark.parametrize("raw", ["{}", "[1, 2]"], ids=["empty-dict", "non-dict"])
def test_direction_empty_shape_returns_empty_dict(monkeypatch, ref_dir, raw):
    """Пустой словарь и не-словарь на входе -> {} без 500."""
    _write_raw(ref_dir, "09.03.04", raw)
    client = _client(monkeypatch, role="teacher", pool=_BoundPool(["09.03.04"]))

    r = client.get("/api/krm/teacher/directions/09.03.04")
    assert r.status_code == 200, r.text
    assert r.json() == {}


# ----------------------------------------------------- disciplines / detail


def test_disciplines_list_from_real_file(monkeypatch, ref_dir):
    _write_krm(ref_dir, GOOD, GOOD_DATA)

    r = _client(monkeypatch, role="admin").get(
        f"/api/krm/teacher/directions/{GOOD}/disciplines"
    )
    assert r.status_code == 200, r.text
    assert r.json() == [
        {"name": "Архитектура ПО", "competencies_count": 1, "skills_count": 1},
        {"name": "Базы данных", "competencies_count": 2, "skills_count": 3},
    ]


def test_disciplines_list_empty_boundary(monkeypatch, ref_dir):
    """disciplines: null в данных не роняет ручку - пустой список."""
    _write_krm(ref_dir, "09.03.04", {"x": {"direction_name": "Пустое", "disciplines": None}})
    client = _client(monkeypatch, role="teacher", pool=_BoundPool(["09.03.04"]))

    r = client.get("/api/krm/teacher/directions/09.03.04/disciplines")
    assert r.status_code == 200, r.text
    assert r.json() == []


def test_discipline_detail_exact(monkeypatch, ref_dir):
    _write_krm(ref_dir, GOOD, GOOD_DATA)

    r = _client(monkeypatch, role="admin").get(
        f"/api/krm/teacher/directions/{GOOD}/disciplines/Базы данных"
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"name": "Базы данных", "dir_code": GOOD, "competencies": DB_SKILLS}


def test_discipline_detail_case_and_space_insensitive(monkeypatch, ref_dir):
    """Fuzzy: другой регистр/пробелы находят ту же дисциплину, имя - как запросили."""
    _write_krm(ref_dir, GOOD, GOOD_DATA)

    r = _client(monkeypatch, role="admin").get(
        f"/api/krm/teacher/directions/{GOOD}/disciplines/базы данных"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "базы данных"
    assert body["dir_code"] == GOOD
    assert body["competencies"] == DB_SKILLS


def test_discipline_detail_not_found_404(monkeypatch, ref_dir):
    _write_krm(ref_dir, GOOD, GOOD_DATA)

    r = _client(monkeypatch, role="admin").get(
        f"/api/krm/teacher/directions/{GOOD}/disciplines/Физика"
    )
    assert r.status_code == 404, r.text
    assert r.json()["detail"] == "Discipline 'Физика' not found"


# ----------------------------------------------------- auth (новые ручки)


def test_new_endpoints_require_token(monkeypatch):
    client = _client(monkeypatch, role=None)
    for path in (
        f"/api/krm/teacher/directions/{GOOD}/disciplines",
        f"/api/krm/teacher/directions/{GOOD}/disciplines/Базы данных",
    ):
        assert client.get(path).status_code == 401, path
    r = client.post(f"/api/krm/teacher/directions/{GOOD}/gap", json={"topic": "sql"})
    assert r.status_code == 401, r.text


# ----------------------------------------------------- gap


def test_gap_returns_analyzer_result(monkeypatch):
    """Happy path: анализатор сработал; LLM-улучшение выключено флагом."""
    monkeypatch.setattr(config, "LLM_ENABLED", True)
    monkeypatch.setattr(config, "LLM_ENHANCE_TEACHER", False)
    calls: list = []
    _install_gap_module(monkeypatch, calls, result=GAP_RESULT)
    client = _client(monkeypatch, role="teacher", pool=_BoundPool([GOOD]))

    r = client.post(f"/api/krm/teacher/directions/{GOOD}/gap", json={"topic": "sql"})
    assert r.status_code == 200, r.text
    assert r.json() == {"dir_code": GOOD, **GAP_RESULT}
    assert calls == [(GOOD, "sql")], "анализатор получил код и тему"


def test_gap_llm_enhanced_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENABLED", True)
    monkeypatch.setattr(config, "LLM_ENHANCE_TEACHER", True)
    _install_gap_module(monkeypatch, [], result=GAP_RESULT)
    llm_calls: list = []
    enhanced = {"recommendations": ["улучшено LLM"], "total_gaps": 0, "llm": True}
    _install_llm_module(monkeypatch, llm_calls, enhanced=enhanced)
    client = _client(monkeypatch, role="teacher", pool=_BoundPool([GOOD]))

    r = client.post(f"/api/krm/teacher/directions/{GOOD}/gap", json={"topic": "sql"})
    assert r.status_code == 200, r.text
    assert r.json() == {"dir_code": GOOD, **enhanced}
    assert llm_calls == [("sql", GAP_RESULT["detailed_analysis"])]


def test_gap_llm_non_dict_result_ignored(monkeypatch):
    """enhance вернул не словарь: остаётся базовый результат анализатора."""
    monkeypatch.setattr(config, "LLM_ENABLED", True)
    monkeypatch.setattr(config, "LLM_ENHANCE_TEACHER", True)
    _install_gap_module(monkeypatch, [], result=GAP_RESULT)
    _install_llm_module(monkeypatch, [], enhanced="not a dict")
    client = _client(monkeypatch, role="teacher", pool=_BoundPool([GOOD]))

    r = client.post(f"/api/krm/teacher/directions/{GOOD}/gap", json={"topic": "sql"})
    assert r.status_code == 200, r.text
    assert r.json() == {"dir_code": GOOD, **GAP_RESULT}


def test_gap_llm_error_swallowed(monkeypatch):
    """LLM упал - ответ всё равно 200 с базовыми рекомендациями."""
    monkeypatch.setattr(config, "LLM_ENABLED", True)
    monkeypatch.setattr(config, "LLM_ENHANCE_TEACHER", True)
    _install_gap_module(monkeypatch, [], result=GAP_RESULT)
    _install_llm_module(monkeypatch, [], error=RuntimeError("llm down"))
    client = _client(monkeypatch, role="teacher", pool=_BoundPool([GOOD]))

    r = client.post(f"/api/krm/teacher/directions/{GOOD}/gap", json={"topic": "sql"})
    assert r.status_code == 200, r.text
    assert r.json() == {"dir_code": GOOD, **GAP_RESULT}


def test_gap_llm_skipped_when_disabled(monkeypatch):
    """LLM_ENABLED=False: короткое замыкание до импорта llm-модуля."""
    monkeypatch.setattr(config, "LLM_ENABLED", False)
    monkeypatch.setattr(config, "LLM_ENHANCE_TEACHER", True)
    calls: list = []
    _install_gap_module(monkeypatch, [], result=GAP_RESULT)
    _install_llm_module(monkeypatch, calls, enhanced={"llm": "must-not-appear"})
    client = _client(monkeypatch, role="teacher", pool=_BoundPool([GOOD]))

    r = client.post(f"/api/krm/teacher/directions/{GOOD}/gap", json={"topic": "sql"})
    assert r.status_code == 200, r.text
    assert r.json() == {"dir_code": GOOD, **GAP_RESULT}
    assert calls == [], "LLM не должен вызываться при выключенном LLM_ENABLED"


def test_gap_analyzer_error_503(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENABLED", True)
    monkeypatch.setattr(config, "LLM_ENHANCE_TEACHER", False)
    _install_gap_module(monkeypatch, [], error=RuntimeError("analyzer boom"))
    client = _client(monkeypatch, role="teacher", pool=_BoundPool([GOOD]))

    r = client.post(f"/api/krm/teacher/directions/{GOOD}/gap", json={"topic": "sql"})
    assert r.status_code == 503, r.text
    assert r.json()["detail"] == "Не удалось выполнить анализ разрыва"


def test_gap_student_forbidden_403_gate(monkeypatch):
    """Гейт admin/teacher/rop: student не проходит до анализатора."""
    client = _client(monkeypatch, role="student")

    r = client.post(f"/api/krm/teacher/directions/{GOOD}/gap", json={"topic": "sql"})
    assert r.status_code == 403, r.text


def test_gap_teacher_foreign_direction_403(monkeypatch, ref_dir):
    """Роль проходит гейт, но привязка на чужое направление: 403 от allow-list."""
    _write_krm(ref_dir, GOOD, GOOD_DATA)
    client = _client(monkeypatch, role="teacher", pool=_BoundPool([OTHER]))

    r = client.post(f"/api/krm/teacher/directions/{GOOD}/gap", json={"topic": "sql"})
    assert r.status_code == 403, r.text
    assert "запрещ" in r.json()["detail"]
