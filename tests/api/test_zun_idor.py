"""IDOR-guard в ZUN: преподаватель не правит ЗУН чужого направления.

Отдельно от логических тестов (test_zun_links_v33 и др.): там guard
подменяется на пропуск, здесь проверяется именно он, через реальный
`zun._require_dir_access`.

Контракт:
  * competency/discipline/KSA другого направления -> 403;
  * своё направление -> операция проходит;
  * admin -> проходит всегда;
  * неизвестный объект -> 404 (от эндпоинта, не от guard).
"""
import uuid

from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.zun as zun_mod
from src.api_pkg import create_app

COMP_ID = str(uuid.uuid4())
DISC_ID = str(uuid.uuid4())
KSA_ID = str(uuid.uuid4())

OWN = "09.03.02"
FOREIGN = "01.03.01"


class FakePool:
    """Пул, который отвечает, что объект принадлежит FOREIGN."""

    def __init__(self, dir_code: str = FOREIGN):
        self.dir_code = dir_code
        self.executed: list[tuple[str, tuple]] = []

    async def fetchrow(self, sql, *args):
        # запрос направления для guard — первым: он ищет d.code
        if "d.code" in sql:
            return {"code": self.dir_code}
        # проверка существования компетенции/дисциплины
        if "FROM competencies c" in sql and "WHERE c.id = $1" in sql:
            return {"id": COMP_ID}
        if "FROM disciplines WHERE id = $1" in sql:
            return {"id": DISC_ID}
        if "SELECT id FROM competencies WHERE id = $1" in sql:
            return {"id": COMP_ID}
        # дубликаты KSA / компетенций — нет
        if "FROM ksa_entries" in sql and "original_text" in sql:
            return None
        if "FROM competencies WHERE discipline_id = $1 AND code = $2" in sql:
            return None
        if "FROM skills WHERE LOWER(name)" in sql:
            return {"id": str(uuid.uuid4()), "source": "rpd_skills"}
        if "FROM competency_skills cs" in sql:
            return None
        if "UPDATE ksa_entries" in sql:
            return {"id": KSA_ID, "ksa_type": "skills", "competency_id": COMP_ID}
        if "DELETE FROM ksa_entries" in sql:
            return {"competency_id": COMP_ID}
        return None

    async def fetchval(self, sql, *args):
        if "SELECT d.code FROM ksa_entries k" in sql:
            return self.dir_code
        if "SELECT COALESCE(MAX(sort_order)" in sql:
            return 1
        if "INSERT INTO" in sql and "RETURNING id" in sql:
            return str(uuid.uuid4())
        if "FROM disciplines d2" in sql:
            return None
        return None

    async def fetch(self, sql, *args):
        self.executed.append((sql, args))
        if "FROM disciplines d2" in sql:
            return [{"name": "Базы данных и СУБД"}]
        if "FROM discipline_scope" in sql:
            return []
        return []

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "OK"


def _make_client(monkeypatch, pool, role="teacher", dirs=(OWN,)):
    """Клиент с подменённым пользователем и списком доступных направлений."""
    app = create_app()

    async def _fake_user(_request):
        return {"uid": "u-1", "u": f"{role}@example.com", "r": role, "t": None}

    async def _fake_dirs(_user):
        return list(dirs)

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    monkeypatch.setattr(zun_mod, "_user_dir_codes", _fake_dirs)
    monkeypatch.setattr(zun_mod, "get_pool", lambda: pool)
    return TestClient(app)


# --------------------------------------------------------------- access guard

def test_add_entry_foreign_direction_403(monkeypatch):
    client = _make_client(monkeypatch, FakePool(FOREIGN))
    r = client.post(
        f"/api/teacher/zun/competencies/{COMP_ID}/entries",
        json={"ksa_type": "skills", "text": "docker"},
    )
    assert r.status_code == 403, r.text
    assert "запрещён" in r.json()["detail"]


def test_add_entry_own_direction_201(monkeypatch):
    client = _make_client(monkeypatch, FakePool(OWN))
    r = client.post(
        f"/api/teacher/zun/competencies/{COMP_ID}/entries",
        json={"ksa_type": "skills", "text": "docker"},
    )
    assert r.status_code == 201, r.text


def test_add_entry_admin_bypasses(monkeypatch):
    """Админ видит все направления, поэтому чужое для teacher ему доступно."""
    client = _make_client(
        monkeypatch, FakePool(FOREIGN), role="admin", dirs=(OWN, FOREIGN)
    )
    r = client.post(
        f"/api/teacher/zun/competencies/{COMP_ID}/entries",
        json={"ksa_type": "skills", "text": "docker"},
    )
    assert r.status_code == 201, r.text


def test_add_competency_foreign_direction_403(monkeypatch):
    client = _make_client(monkeypatch, FakePool(FOREIGN))
    r = client.post(
        f"/api/teacher/zun/disciplines/{DISC_ID}/competencies",
        json={"code": "ПК-9.9", "name": "Чужая"},
    )
    assert r.status_code == 403, r.text


def test_link_skill_foreign_direction_403(monkeypatch):
    client = _make_client(monkeypatch, FakePool(FOREIGN))
    r = client.post(
        f"/api/teacher/zun/competencies/{COMP_ID}/skills",
        json={"skill_name": "docker", "ksa_type": "skills"},
    )
    assert r.status_code == 403, r.text


def test_patch_entry_foreign_direction_403(monkeypatch):
    client = _make_client(monkeypatch, FakePool(FOREIGN))
    r = client.patch(f"/api/teacher/zun/entries/{KSA_ID}", json={"text": "взлом"})
    assert r.status_code == 403, r.text


def test_delete_entry_foreign_direction_403(monkeypatch):
    client = _make_client(monkeypatch, FakePool(FOREIGN))
    r = client.delete(f"/api/teacher/zun/entries/{KSA_ID}")
    assert r.status_code == 403, r.text


def test_put_scope_foreign_direction_403(monkeypatch):
    client = _make_client(monkeypatch, FakePool(FOREIGN))
    r = client.put(
        "/api/teacher/zun/scope",
        json={"dir_code": FOREIGN, "changes": [{"discipline_name": "Базы данных и СУБД",
                                               "included": False}]},
    )
    assert r.status_code == 403, r.text


def test_import_foreign_direction_403(monkeypatch):
    client = _make_client(monkeypatch, FakePool(FOREIGN))
    r = client.post(f"/api/teacher/zun/import/{FOREIGN}?dry_run=true")
    assert r.status_code == 403, r.text


class _FakeProc:
    """Заглушка фонового подпроцесса анализа, чтобы тест не запускал src.cli."""
    returncode = 0

    async def communicate(self):
        return b"", b""


def _patch_analyze_subprocess(monkeypatch):
    async def _fake_subprocess(*_args, **_kwargs):
        return _FakeProc()

    monkeypatch.setattr(zun_mod.asyncio, "create_subprocess_exec", _fake_subprocess)


def test_analyze_foreign_direction_403(monkeypatch):
    """Teacher не может запустить анализ чужого направления."""
    _patch_analyze_subprocess(monkeypatch)
    client = _make_client(monkeypatch, FakePool(), dirs=(OWN,))
    r = client.post("/api/teacher/zun/analyze", params={"dir_code": FOREIGN})
    assert r.status_code == 403, r.text
    assert "запрещён" in r.json()["detail"]


def test_analyze_admin_bypasses(monkeypatch):
    """Админ запускает анализ любого направления (не 403)."""
    _patch_analyze_subprocess(monkeypatch)
    client = _make_client(
        monkeypatch, FakePool(), role="admin", dirs=(OWN, FOREIGN)
    )
    r = client.post("/api/teacher/zun/analyze", params={"dir_code": FOREIGN})
    assert r.status_code == 200, r.text


def test_analyze_results_foreign_direction_403(monkeypatch, tmp_path):
    """Результаты анализа чужого направления не читаются с диска."""
    monkeypatch.setattr(zun_mod, "RESULT_TEACHER", tmp_path)
    client = _make_client(monkeypatch, FakePool(), dirs=(OWN,))
    r = client.get("/api/teacher/zun/analyze/results", params={"dir_code": FOREIGN})
    assert r.status_code == 403, r.text


# ------------------------------------------------------------ auth / identity

def test_patch_entry_unknown_ksa_is_not_403(monkeypatch):
    """Несуществующая запись не должна маскироваться под отказ в доступе."""

    class NoKsaPool(FakePool):
        async def fetchval(self, sql, *args):
            if "SELECT d.code FROM ksa_entries k" in sql:
                return None
            return await super().fetchval(sql, *args)

        async def fetchrow(self, sql, *args):
            if "UPDATE ksa_entries" in sql:
                return None
            return await super().fetchrow(sql, *args)

    client = _make_client(monkeypatch, NoKsaPool())
    r = client.patch(f"/api/teacher/zun/entries/{KSA_ID}", json={"text": "x"})
    assert r.status_code == 404, r.text


def test_unauthorized_when_no_session(monkeypatch):
    """Без пользователя guard отдаёт 401, а не 403/500."""
    app = create_app()

    async def _no_user(_request):
        return None

    async def _dirs(_user):
        return [OWN]

    monkeypatch.setattr(auth_mod, "get_current_user", _no_user)
    monkeypatch.setattr(zun_mod, "_user_dir_codes", _dirs)
    monkeypatch.setattr(zun_mod, "get_pool", lambda: FakePool(FOREIGN))
    client = TestClient(app)

    r = client.post(
        f"/api/teacher/zun/competencies/{COMP_ID}/entries",
        json={"ksa_type": "skills", "text": "docker"},
    )
    assert r.status_code == 401, r.text


# ------------------------------------------------------- анализ направления

def test_analyze_own_direction_starts(monkeypatch):
    """Своё направление - запуск разрешён.

    Реальный CLI-прогон не нужен и опасен в тесте (идёт минуты): подменяем
    создание подпроцесса заглушкой.
    """
    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def _fake_exec(*_a, **_kw):
        return _Proc()

    # /analyze ограничен 2/мин; несколько тестов в файле исчерпывают лимит.
    # Счётчик slowapi живёт на процесс, поэтому сбрасываем перед вызовом.
    import contextlib as _ctx
    with _ctx.suppress(Exception):
        zun_mod.limiter.reset()

    monkeypatch.setattr(zun_mod.asyncio, "create_subprocess_exec", _fake_exec)
    client = _make_client(monkeypatch, FakePool(OWN))
    r = client.post(f"/api/teacher/zun/analyze?dir_code={OWN}")
    assert r.status_code == 200, r.text
    assert r.json().get("status") == "started"
