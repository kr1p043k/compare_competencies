"""Покрытие тела сбора background_collector: регионы/страницы, пустые ответы
hh.ru, сетевые ошибки/таймауты, отмена, JSON-слияние и лимиты, обогащение,
парсинг навыков, запись в БД (мок), снапшоты рынка и служебные ветки планировщика.

Все внешние вызовы (hh.ru API, PostgreSQL, LLM, снапшот-CLI) подменяются;
файлы пишутся только в tmp-директории из conftest. Боевые таблицы не трогаем.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.pipeline.background_collector as sched
from src import Err, Ok

# ---------------------------------------------------------------------------
# Фикстуры и заглушки
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_sched(tmp_path, monkeypatch):
    """Изолированные настройки планировщика + чистые глобальные кэши."""
    monkeypatch.setattr(sched.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sched.config.settings, "BACKGROUND_COLLECTOR_ENABLED", True)
    monkeypatch.setattr(sched.config.settings, "SCHEDULER_COLLECT_INTERVAL_HOURS", 12)
    monkeypatch.setattr(sched.config.settings, "SCHEDULER_DAILY_GAP_ENABLED", False)
    sched._scheduler_cache = None
    sched._scheduler_busy = None
    yield tmp_path
    sched._scheduler_cache = None
    sched._scheduler_busy = None


async def _ok_true():
    return True


async def _ok_false():
    return False


def _vac(vid, **over):
    """Ответ hh.ru в сыром виде: валиден для Vacancy.from_api по умолчанию."""
    base = {
        "id": str(vid),
        "name": f"Developer {vid}",
        "area": {"id": 1, "name": "Москва"},
        "employer": {"id": "e7", "name": "TestCorp"},
        "key_skills": [{"name": "Docker"}],
        "published_at": "2026-02-01T09:00:00+00:00",
    }
    base.update(over)
    return base


class _RecordingAPI:
    """Заглушка HeadHunterAPI: сценарные ответы + журнал вызовов. Полностью офлайн."""

    def __init__(
        self,
        by_area=None,
        details=None,
        raise_roles=(),
        raise_exc=RuntimeError,
        err_roles=(),
        cancel_areas=(),
        fail_detail_ids=(),
        err_detail_ids=(),
    ):
        self.by_area = by_area or {}
        self.details = details or {}
        self.raise_roles = set(raise_roles)
        self.raise_exc = raise_exc
        self.err_roles = set(err_roles)
        self.cancel_areas = set(cancel_areas)
        self.fail_detail_ids = set(fail_detail_ids)
        self.err_detail_ids = set(err_detail_ids)
        self.search_calls: list[dict] = []
        self.detail_calls: list[str] = []

    def search_vacancies(
        self, text="", area=None, period_days=None, max_pages=None, professional_role=None, **kw
    ):
        self.search_calls.append(
            {
                "text": text,
                "area": area,
                "period_days": period_days,
                "max_pages": max_pages,
                "role": professional_role,
            }
        )
        if professional_role in self.raise_roles:
            raise self.raise_exc("hh.ru unavailable")
        if area in self.cancel_areas:
            raise asyncio.CancelledError()
        if professional_role in self.err_roles:
            return Err("hh api 503 service unavailable")
        return Ok(list(self.by_area.get(area, [])))

    def get_vacancy_details(self, vacancy_id):
        self.detail_calls.append(str(vacancy_id))
        if vacancy_id in self.fail_detail_ids:
            raise RuntimeError("detail request failed")
        if vacancy_id in self.err_detail_ids:
            return Err("404 not found")
        return Ok(dict(self.details.get(vacancy_id, {})))


class _FakeConn:
    def __init__(self):
        self.executed: list[tuple] = []

    async def execute(self, sql, *args):
        self.executed.append((sql, args))


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    """Пул asyncpg: подмена fetchval MAX/COUNT и acquire() без реальной БД."""

    def __init__(self, last_run=None, before=0, after=None, fail_count_after=False, fail_max=False):
        self.last_run = last_run
        self.before = before
        self.after = after if after is not None else before
        self.fail_count_after = fail_count_after
        self.fail_max = fail_max
        self.count_calls = 0
        self.conn = _FakeConn()

    async def fetchval(self, sql, *args):
        if "MAX(started_at)" in sql:
            if self.fail_max:
                raise RuntimeError("pg down")
            return self.last_run
        if "COUNT(*)" in sql:
            self.count_calls += 1
            if self.count_calls == 1:
                return self.before
            if self.fail_count_after:
                raise RuntimeError("count query failed")
            return self.after
        return None

    def acquire(self):
        return _FakeAcquire(self.conn)


class _DBWriterStub:
    """Подмена db_writer: пакетная вставка + журнал pipeline_runs без БД."""

    def __init__(self, batch_error=None, create_error=None, complete_error=None):
        self.batch_error = batch_error
        self.create_error = create_error
        self.complete_error = complete_error
        self.saved: list | None = None
        self.run_events: list[tuple] = []

    async def save_vacancies_batch(self, vacancies, run_id=None):
        if self.batch_error:
            raise self.batch_error
        self.saved = vacancies
        return len(vacancies)

    async def create_pipeline_run(self, action, *a, **kw):
        if self.create_error:
            raise self.create_error
        self.run_events.append(("create", action))
        return "run-1"

    async def complete_pipeline_run(self, run_id, status="completed", error=None, stats=None):
        if status == "completed" and self.complete_error:
            raise self.complete_error
        self.run_events.append(("complete", run_id, status, stats, error))


class _FakeVacancyParser:
    """Подмена VacancyParser: заранее заданные исходы parse_vacancy по id."""

    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.calls: list[str] = []
        self.skill_parser = self

    def parse_vacancy(self, vacancy):
        self.calls.append(str(vacancy.id))
        kind, payload = self.outcomes.get(str(vacancy.id), ("ok", []))
        if kind == "ok":
            return Ok(list(payload))
        return Err(payload)


class _FakeSkillNormalizer:
    @staticmethod
    def normalize(skill):
        return Ok("norm-skill")


def _fake_trends_class(sink, result=None):
    """Фабрика подмены TrendAnalyzer с журналированием входов."""

    class _FakeTrendAnalyzer:
        def __init__(self, freq):
            sink["init_freq"] = dict(freq)

        def save_snapshot(self, frequencies, **kw):
            sink["snapshot"] = dict(frequencies)
            sink["kwargs"] = kw
            sink["calls"] = sink.get("calls", 0) + 1
            return result if result is not None else Ok(Path("freq_market_test.json"))

    return _FakeTrendAnalyzer


async def _noop_async(*_a, **_kw):
    return None


def _patch_collect(monkeypatch, api, pool, roles=("r1",), parser=None, dbw=None, trends=None):
    """Единая точка подмены внешних зависимостей _try_collect."""
    import src.parsing.api.hh_api as hh
    import src.parsing.utils as putils

    monkeypatch.setattr(putils, "IT_PROFESSIONAL_ROLES", list(roles))
    monkeypatch.setattr(hh, "HeadHunterAPI", lambda: api)
    monkeypatch.setattr(sched, "get_pool", lambda: pool)
    monkeypatch.setattr(sched, "_refresh_history_snapshots", _noop_async)
    if parser is not None:
        import src.parsing.skills.vacancy_parser as vp

        monkeypatch.setattr(vp, "VacancyParser", lambda: parser)
    if dbw is not None:
        import src.pipeline.db_writer as dbw_mod

        monkeypatch.setattr(dbw_mod, "save_vacancies_batch", dbw.save_vacancies_batch)
        monkeypatch.setattr(dbw_mod, "create_pipeline_run", dbw.create_pipeline_run)
        monkeypatch.setattr(dbw_mod, "complete_pipeline_run", dbw.complete_pipeline_run)
    if trends is not None:
        import src.analyzers.skills.trends as tr

        monkeypatch.setattr(tr, "TrendAnalyzer", trends)


def _seed_detailed(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


# ---------------------------------------------------------------------------
# _try_collect: период, инкремент, регионы, лимиты
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "last_run_age", "expected_period"),
    [
        ({"force_period_days": 99}, None, 30),
        ({"force_period_days": 0}, None, 1),
        ({}, timedelta(days=5, hours=1), 5),
        ({}, timedelta(days=40), 30),
        ({}, None, 30),
    ],
    ids=["force-high-clamp", "force-low-clamp", "incremental", "too-old-stays-30", "no-last-run"],
)
async def test_collect_period_and_limits(tmp_sched, monkeypatch, kwargs, last_run_age, expected_period):
    """Лимиты поиска: force clamp, инкремент по дельте last_run, дефолт 30."""
    api = _RecordingAPI()
    last_run = None
    if last_run_age is not None:
        last_run = datetime.now(UTC) - last_run_age
    pool = _FakePool(last_run=last_run)
    _patch_collect(monkeypatch, api, pool, roles=("r1",))
    status, new = await sched._try_collect(**kwargs)
    assert status == "collected-empty"
    assert new == 0
    assert api.search_calls, "сбор обязан хотя бы раз дернуть API"
    periods = {c["period_days"] for c in api.search_calls}
    assert periods == {expected_period}
    assert {c["max_pages"] for c in api.search_calls} == {5}
    assert {c["area"] for c in api.search_calls} == set(range(1, 201))
    assert {c["role"] for c in api.search_calls} == {"r1"}


async def test_collect_dedup_seen_ids_across_regions_and_roles(tmp_sched, monkeypatch):
    """Дубли по id между регионами/ролями не попадают в выборку дважды;
    вакансия без id отбрасывается на входе."""
    api = _RecordingAPI(
        by_area={
            1: [_vac("dup"), {"id": None, "name": "Безымянная", "area": {}, "employer": {}}],
            2: [_vac("dup"), _vac("unique")],
        }
    )
    pool = _FakePool(before=0, after=1)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, roles=("r1", "r2"), parser=parser, dbw=dbw)
    status, new = await sched._try_collect()
    assert status == "collected" and new == 1
    assert dbw.saved is not None
    saved_ids = [v["id"] for v in dbw.saved]
    assert sorted(saved_ids) == ["dup", "unique"]


async def test_collect_role_timeout_and_err_yield_collected_empty(tmp_sched, monkeypatch):
    """Таймаут/сбой сети на одной роли и Err-ответ hh.ru не роняют сбор:
    остальные роли обходятся, итог - collected-empty при пустой выдаче."""
    api = _RecordingAPI(raise_roles=("r1",), raise_exc=TimeoutError, err_roles=("r2",))
    pool = _FakePool()
    _patch_collect(monkeypatch, api, pool, roles=("r1", "r2"))
    status, new = await sched._try_collect()
    assert (status, new) == ("collected-empty", 0)
    # 200 регионов x 2 роли = пасть должны обе, но продолжить цикл
    assert len(api.search_calls) == 400


async def test_collect_role_connection_error_continues(tmp_sched, monkeypatch):
    """ConnectionError на первой роли не мешает второй собрать данные."""
    api = _RecordingAPI(
        by_area={3: [_vac("survivor")]},
        raise_roles=("r1",),
        raise_exc=ConnectionError,
    )
    pool = _FakePool(before=0, after=1)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, roles=("r1", "r2"), parser=parser, dbw=dbw)
    status, _ = await sched._try_collect()
    assert status == "collected"
    assert [v["id"] for v in dbw.saved] == ["survivor"]


async def test_collect_region_cancelled_is_logged_and_collected(tmp_sched, monkeypatch):
    """Отмена одного региона фиксируется как BaseException в gather,
    сбор остальных региона продолжается (прерывание не роняет цикл сбора)."""
    api = _RecordingAPI(by_area={2: [_vac("kept")]}, cancel_areas=(7,))
    pool = _FakePool(before=0, after=1)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, _ = await sched._try_collect()
    assert status == "collected"
    assert [v["id"] for v in dbw.saved] == ["kept"]


async def test_collect_cancelled_error_propagates_through_run_once(tmp_sched, monkeypatch):
    """CancelledError сквозь тело сбора не глотается сервисной обёрткой:
    _run_collect_once обязан пробросить её и снять флаг busy."""
    async def _cancelled(*_a, **_kw):
        raise asyncio.CancelledError()

    monkeypatch.setattr(sched, "_try_collect", _cancelled)
    with pytest.raises(asyncio.CancelledError):
        await sched._run_collect_once()
    assert sched.is_scheduler_busy() is False


# ---------------------------------------------------------------------------
# _try_collect: подробный JSON-файл (слияние, лимит записей, дедупликация)
# ---------------------------------------------------------------------------


async def test_collect_json_merge_cap_and_undated(tmp_sched, monkeypatch):
    """Слияние с существующим файлом: дубликаты по id не перезаписываются,
    лимит JSON_MAX_RECORDS режет старые датированные записи, без даты - сохраняются."""
    monkeypatch.setattr(sched, "JSON_MAX_RECORDS", 2)
    detailed = sched.config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    _seed_detailed(
        detailed,
        [
            {"id": "999", "name": "SEED-999", "published_at": "2020-01-01T00:00:00+00:00"},
            {"id": "1", "name": "SEED-1", "published_at": "2021-01-01"},
        ],
    )
    api = _RecordingAPI(by_area={1: [_vac("1"), _vac("2", published_at=None), _vac("3", published_at="2026-01-01")]})
    pool = _FakePool(before=0, after=3)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, _ = await sched._try_collect()
    assert status == "collected"
    merged = json.loads(detailed.read_text(encoding="utf-8"))
    by_id = {v["id"]: v for v in merged}
    assert set(by_id) == {"1", "2", "3"}
    assert by_id["1"]["name"] == "SEED-1", "уже существующая вакансия не перезаписывается"
    assert "999" not in by_id, "лимит должен отрезать самые старые датированные записи"
    assert "2" in by_id, "записи без published_at не подвергаются лимиту"
    saved_ids = [v["id"] for v in dbw.saved]
    assert saved_ids == [1, 2, 3], "числовые id приводятся к int перед батчем (боевой код L579-582)"


async def test_collect_json_save_failure_does_not_break_collect(tmp_sched, monkeypatch):
    """Ошибка записи подробного JSON логируется и не ломает сбор: записи идут в БД."""
    def _boom(data, path):
        raise OSError("disk full")

    monkeypatch.setattr(sched, "atomic_write_json", _boom)
    api = _RecordingAPI(by_area={1: [_vac("10")]})
    pool = _FakePool(before=0, after=1)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, new = await sched._try_collect()
    assert (status, new) == ("collected", 1)
    detailed = sched.config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    assert not detailed.exists()


# ---------------------------------------------------------------------------
# _try_collect: обогащение деталями
# ---------------------------------------------------------------------------


async def test_collect_enrich_err_exception_and_prefilled_skip(tmp_sched, monkeypatch):
    """Ветви обогащения: Err и исключение не роняют сбор, карточки
    с готовым описанием не запрашиваются повторно."""
    api = _RecordingAPI(
        by_area={
            1: [
                _vac("1", description="Уже готово"),
                _vac("2"),
                _vac("3"),
                _vac("4"),
            ]
        },
        err_detail_ids={"3"},
        fail_detail_ids={"4"},
        details={
            "1": {"description": "никогда не запрашивается"},
            "2": {"description": "Ответ сервера с описанием вакансии номер 2"},
        },
    )
    pool = _FakePool(before=0, after=4)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, _ = await sched._try_collect()
    assert status == "collected"
    assert "1" not in api.detail_calls, "вакансии с готовым описанием не должны дёргать API"
    assert {"2", "3", "4"} <= set(api.detail_calls)
    by_id = {str(v["id"]): v for v in dbw.saved}
    assert by_id["2"]["description"] == "Ответ сервера с описанием вакансии номер 2"
    assert "description" not in by_id["3"], "Err-ответ не должен менять вакансию"
    assert "description" not in by_id["4"], "упавший запрос деталей не должен менять вакансию"


async def test_collect_enrich_snippet_merge_priority(tmp_sched, monkeypatch):
    """Мерж сниппета: заполненные поля вакансии приоритетнее ответа API,
    пустые - добираются из деталей."""
    api = _RecordingAPI(
        by_area={
            1: [
                _vac("s1", snippet={"requirement": "Mine", "responsibility": None}),
                _vac("s2", snippet={}),
                _vac("s3"),
            ]
        },
        details={
            "s1": {
                "description": "Описание с требованием знания python и docker",
                "snippet": {"requirement": "DetReq1", "responsibility": "DetResp1"},
            },
            "s2": {
                "description": "Описание второй вакансии, тоже про python",
                "snippet": {"requirement": "DetReq2", "responsibility": "DetResp2"},
            },
            "s3": {
                "description": "Третье описание, короткое",
                "snippet": {"requirement": "DetReq3"},
            },
        },
    )
    pool = _FakePool(before=0, after=3)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, _ = await sched._try_collect()
    assert status == "collected"
    by_id = {v["id"]: v for v in dbw.saved}
    assert by_id["s1"]["snippet"] == {"requirement": "Mine", "responsibility": "DetResp1"}
    assert by_id["s2"]["snippet"] == {"requirement": "DetReq2", "responsibility": "DetResp2"}
    assert by_id["s3"]["snippet"] == {"requirement": "DetReq3", "responsibility": ""}


# ---------------------------------------------------------------------------
# _try_collect: парсинг навыков, отсев не-IT, запись parsed_skills
# ---------------------------------------------------------------------------


async def test_collect_parse_skills_cases(tmp_sched, monkeypatch):
    """Все исходы парсинга навыков: Ok с данными -> UPDATE parsed_skills,
    Ok без данных -> отсев по ключам/описанию, Err -> skip, битая вакансия -> skip."""
    import src.parsing.skills.skill_normalizer as sn

    monkeypatch.setattr(sn, "SkillNormalizer", _FakeSkillNormalizer)
    api = _RecordingAPI(
        by_area={
            1: [
                _vac("100"),
                _vac("101"),
                _vac(
                    "102",
                    key_skills=[],
                    description="Требуется уборка столов и мытьё полов на кухне каждый день утром и вечером",
                ),
                _vac("103", key_skills=[], description="Нужен инженер, знает python, docker, kubernetes, git и sql"),
                _vac("104", name=""),
                _vac("105"),
                _vac("abc"),
            ]
        }
    )
    parser = _FakeVacancyParser(
        outcomes={
            "100": ("ok", [SimpleNamespace(text=" Python "), SimpleNamespace(text="SQL"), SimpleNamespace(text="python")]),
            "105": ("err", "parser exploded"),
        }
    )
    # "abc" без исхода -> Ok([]) от заглушки: чистый путь до конца блока,
    # branch отсева и prune отрабатывают (без исключения int падение блока
    # и пропуск prune проверяет test_collect_parse_block_exception_keeps_pipeline)
    sink = {}
    pool = _FakePool(before=0, after=5)
    dbw = _DBWriterStub()
    _patch_collect(
        monkeypatch, api, pool, parser=parser, dbw=dbw, trends=_fake_trends_class(sink)
    )
    status, _ = await sched._try_collect()
    assert status == "collected"
    # UPDATE только для вакансии со скиллами
    assert len(pool.conn.executed) == 1
    sql, args = pool.conn.executed[0]
    assert "parsed_skills" in sql
    assert json.loads(args[0]) == ["python", "sql"]
    assert args[1] == 100
    # отработавшие регулярки
    assert "104" not in parser.calls, "битая вакансия не должна доходить до парсера навыков"
    saved_ids = [v["id"] for v in dbw.saved]
    # порядок = порядок сбора; _vid-фильтр (L561-568) снимает только не-IT
    # вакансии по empty_ids - не-IT "102" отсечена, "abc" остаётся строкой:
    # int-конверсия перед батчем обёрнута в contextlib.suppress (L579-582)
    assert saved_ids == [100, 101, 103, 104, 105, "abc"], (
        "числовые id становятся int, не-IT отсекается, не парсящийся id не убивает батч"
    )


async def test_collect_parse_block_exception_keeps_pipeline(tmp_sched, monkeypatch):
    """Сбой внутри блока парсинга глушится как collect_parse_failed:
    неразобранные вакансии всё равно уходят в батч."""
    api = _RecordingAPI(by_area={1: [_vac("19xyz"), _vac("500")]})
    parser = _FakeVacancyParser(outcomes={"19xyz": ("ok", [SimpleNamespace(text="python")])})
    sink = {}
    pool = _FakePool(before=0, after=2)
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw, trends=_fake_trends_class(sink))
    status, new = await sched._try_collect()
    assert (status, new) == ("collected", 2)
    assert pool.conn.executed == [], "упавший int(id) не должен оставлять UPDATE"
    saved_ids = [v["id"] for v in dbw.saved]
    assert saved_ids == ["19xyz", 500], (
        "сломанный int(id) не выбрасывает вакансию из батча; годные id приводятся к int"
    )
    # Снапшот привязан не к парсингу, а к батчу: counted после сбоя блока.
    # extracted_skills в 19xyz записан write-back'ом (L532) ДО падения int(id),
    # поэтому в частотах остаётся именно python (реальный SkillNormalizer).
    assert sink["calls"] == 1, "сбой парсинга не отменяет снапшот рынка"
    assert sink["init_freq"] == {"python": 1}
    assert sink["snapshot"] == {"python": 1}
    assert sink["kwargs"] == {
        "apply_whitelist": True,
        "source_type": "full_market",
        "vacancy_count": 2,
    }


# ---------------------------------------------------------------------------
# _try_collect: запись в БД и учёт pipeline_runs
# ---------------------------------------------------------------------------


async def test_collect_full_chain_with_real_models(tmp_sched, monkeypatch):
    """Сквозной сбор: регионы -> дедуп -> JSON -> обогащение -> парсинг ->
    UPDATE parsed_skills -> батч в БД -> pipeline_run -> снапшот рынка."""
    api = _RecordingAPI(
        by_area={1: [_vac("1")], 2: [_vac("2"), _vac("3", description="Готовое описание") ] },
        details={
            "1": {
                "description": "Владеет python, docker и sql на продвинутом уровне",
                "snippet": {"requirement": "DetReq"},
            },
            "2": {"description": "Владеет python, docker и sql уверенно"},
        },
        err_detail_ids={"3"},
    )
    parser = _FakeVacancyParser(
        outcomes={
            "1": ("ok", [SimpleNamespace(text="python"), SimpleNamespace(text="sql")]),
            "2": ("ok", [SimpleNamespace(text="python")]),
            "3": ("ok", []),
        }
    )
    pool = _FakePool(before=10, after=13)
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, new = await sched._try_collect()
    assert (status, new) == ("collected", 3)
    assert sorted(v["id"] for v in dbw.saved) == [1, 2, 3], "id должны быть приведены к int"
    assert [a[1] for _, a in pool.conn.executed] == [1, 2]
    detailed = sched.config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    assert {v["id"] for v in json.loads(detailed.read_text(encoding="utf-8"))} == {"1", "2", "3"}
    assert [e[:2] for e in dbw.run_events] == [("create", "data-collection"), ("complete", "run-1")]
    stats = dbw.run_events[1][3]
    assert stats == {"collected": 3, "new": 3}
    # Реальный TrendAnimator пишет снапшот рынка в изолированный HISTORY_DIR
    market = list(Path(sched.config.HISTORY_DIR).glob("freq_market_*.json"))
    assert market, "снапшот рынка обязан появиться после полного сбора"


async def test_collect_db_batch_failure_propagates(tmp_sched, monkeypatch):
    """Сбой пакетной записи в БД не глушится внутри _try_collect - уходит наружу."""
    api = _RecordingAPI(by_area={1: [_vac("600")]})
    pool = _FakePool(before=0, after=1)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub(batch_error=RuntimeError("batch insert failed"))
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    with pytest.raises(RuntimeError, match="batch insert failed"):
        await sched._try_collect()


async def test_collect_run_once_wraps_db_failure(tmp_sched, monkeypatch):
    """Тот же сбой через сервисную обёртку _run_collect_once -> 'failed', busy снят."""
    api = _RecordingAPI(by_area={1: [_vac("601")]})
    pool = _FakePool(before=0, after=1)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub(batch_error=RuntimeError("batch insert failed"))
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    assert await sched._run_collect_once() == ("failed", 0)
    assert sched.is_scheduler_busy() is False


async def test_collect_after_count_failure_keeps_result(tmp_sched, monkeypatch):
    """Неудачный финальный COUNT не роняет сбор: новое число товаров - 0."""
    api = _RecordingAPI(by_area={1: [_vac("700")]})
    pool = _FakePool(before=2, fail_count_after=True)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, new = await sched._try_collect()
    assert (status, new) == ("collected", 0)


async def test_collect_pipeline_run_create_failure(tmp_sched, monkeypatch):
    """Не удалось открыть pipeline_run - предупреждение, сбор всё равно завершён."""
    api = _RecordingAPI(by_area={1: [_vac("800")]})
    pool = _FakePool(before=0, after=1)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub(create_error=RuntimeError("no pipeline_runs table"))
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, _ = await sched._try_collect()
    assert status == "collected"
    assert dbw.run_events == []


async def test_collect_pipeline_run_complete_failure_marks_run_failed(tmp_sched, monkeypatch):
    """Ошибка завершения pipeline_run оставляет явную 'failed'-строку вместо зависшего 'started'."""
    api = _RecordingAPI(by_area={1: [_vac("900")]})
    pool = _FakePool(before=0, after=1)
    parser = _FakeVacancyParser()
    dbw = _DBWriterStub(complete_error=RuntimeError("complete crashed"))
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, _ = await sched._try_collect()
    assert status == "collected"
    kinds = [(e[0], e[2] if e[0] == "complete" else e[1]) for e in dbw.run_events]
    assert kinds == [("create", "data-collection"), ("complete", "failed")]
    assert dbw.run_events[1][4] is not None, "в 'failed'-строке должен остаться текст ошибки"


# ---------------------------------------------------------------------------
# _try_collect: авто-снапшот рынка
# ---------------------------------------------------------------------------


async def test_collect_snapshot_saved_from_extracted_skills(tmp_sched, monkeypatch):
    """extracted_skills уходят в снапшот после нормализации."""
    import src.parsing.skills.skill_normalizer as sn

    monkeypatch.setattr(sn, "SkillNormalizer", _FakeSkillNormalizer)
    api = _RecordingAPI(by_area={1: [_vac("1000")]})
    parser = _FakeVacancyParser(outcomes={"1000": ("ok", [SimpleNamespace(text="python")])})
    sink = {}
    pool = _FakePool(before=0, after=1)
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw, trends=_fake_trends_class(sink))
    status, _ = await sched._try_collect()
    assert status == "collected"
    assert sink["calls"] == 1
    assert sink["snapshot"] == {"norm-skill": 1}
    assert sink["kwargs"]["source_type"] == "full_market"
    assert sink["kwargs"]["vacancy_count"] == 1
    assert sink["kwargs"]["apply_whitelist"] is True


async def test_collect_snapshot_save_failed_logs_warning(tmp_sched, monkeypatch):
    """Err от save_snapshot логируется, но не ломает сбор."""
    import src.parsing.skills.skill_normalizer as sn

    monkeypatch.setattr(sn, "SkillNormalizer", _FakeSkillNormalizer)
    api = _RecordingAPI(by_area={1: [_vac("1100")]})
    parser = _FakeVacancyParser(outcomes={"1100": ("ok", [SimpleNamespace(text="python")])})
    sink = {}
    pool = _FakePool(before=0, after=1)
    dbw = _DBWriterStub()
    _patch_collect(
        monkeypatch, api, pool, parser=parser, dbw=dbw,
        trends=_fake_trends_class(sink, result=Err("history dir read-only")),
    )
    status, _ = await sched._try_collect()
    assert status == "collected"
    assert sink["calls"] == 1


async def test_collect_snapshot_outer_error_swallowed(tmp_sched, monkeypatch):
    """Полный сбой блока снапшотов не меняет статус сбора."""
    import src.parsing.skills.skill_normalizer as sn

    monkeypatch.setattr(sn, "SkillNormalizer", _FakeSkillNormalizer)
    import src.analyzers.skills.trends as tr

    monkeypatch.setattr(tr, "TrendAnalyzer", None)
    api = _RecordingAPI(by_area={1: [_vac("1200")]})
    parser = _FakeVacancyParser(outcomes={"1200": ("ok", [SimpleNamespace(text="python")])})
    pool = _FakePool(before=0, after=1)
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw)
    status, _ = await sched._try_collect()
    assert status == "collected"


async def test_collect_snapshot_parsed_skills_bad_json_skips_analyzer(tmp_sched, monkeypatch):
    """Строковые parsed_skills с битым JSON не ломают снапшот: пустой freq - анализатор не вызывается."""
    import src.parsing.skills.skill_normalizer as sn

    monkeypatch.setattr(sn, "SkillNormalizer", _FakeSkillNormalizer)
    api = _RecordingAPI(by_area={1: [_vac("1300", parsed_skills="[bad json") ]})
    parser = _FakeVacancyParser()
    sink = {}
    pool = _FakePool(before=0, after=1)
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw, trends=_fake_trends_class(sink))
    status, _ = await sched._try_collect()
    assert status == "collected"
    assert sink == {}, "пустой freq не должен доходить до анализатора"


async def test_collect_snapshot_parsed_skills_mixed_types(tmp_sched, monkeypatch):
    """Валидный JSON со смешанными типами: не-строки пропускаются, строки считаются."""
    import src.parsing.skills.skill_normalizer as sn

    monkeypatch.setattr(sn, "SkillNormalizer", _FakeSkillNormalizer)
    api = _RecordingAPI(by_area={1: [_vac("1400", parsed_skills='["golang", 42, null]')]})
    parser = _FakeVacancyParser()
    sink = {}
    pool = _FakePool(before=0, after=1)
    dbw = _DBWriterStub()
    _patch_collect(monkeypatch, api, pool, parser=parser, dbw=dbw, trends=_fake_trends_class(sink))
    status, _ = await sched._try_collect()
    assert status == "collected"
    assert sink["snapshot"] == {"norm-skill": 1}, "только строковые навыки должны попадать в freq"


# ---------------------------------------------------------------------------
# _run_collect_once / _run_scheduled_gap: взаимные блокировки
# ---------------------------------------------------------------------------


async def test_run_collect_once_busy_skips(tmp_sched):
    """Повторный запуск при активном сборе сразу возвращает 'skipped'."""
    sched._scheduler_busy = "collect"
    try:
        assert await sched._run_collect_once() == ("skipped", 0)
    finally:
        sched._scheduler_busy = None


async def test_scheduled_gap_busy_returns_false(tmp_sched):
    """Chained gap при занятом планировщике не стартует и возвращает False."""
    sched._scheduler_busy = "gap"
    try:
        assert await sched._run_scheduled_gap() is False
    finally:
        sched._scheduler_busy = None


# ---------------------------------------------------------------------------
# _refresh_history_snapshots: суточный гейт снимков профессий
# ---------------------------------------------------------------------------


def _patch_snapshot_cli(monkeypatch, prof_calls):
    import src.cli.backfill_market_snapshots as bf
    import src.cli.snapshot_professions as sp

    monkeypatch.setattr(bf, "main", lambda *a, **k: None)

    def _prof_main(force=False, **kw):
        prof_calls.append(1)

    monkeypatch.setattr(sp, "main", _prof_main)


async def test_refresh_prof_snapshots_recent_file_skipped(tmp_sched, monkeypatch):
    """Свежий (меньше суток) снимок профессии не пересоздаётся."""
    prof_calls: list = []
    _patch_snapshot_cli(monkeypatch, prof_calls)
    hist = Path(sched.config.HISTORY_DIR)
    hist.mkdir(parents=True, exist_ok=True)
    (hist / "freq_profession_2026.json").write_text("{}", encoding="utf-8")
    await sched._refresh_history_snapshots()
    assert prof_calls == []


async def test_refresh_prof_snapshots_stale_file_runs(tmp_sched, monkeypatch):
    """Снимок старше суток пересоздаётся через CLI main."""
    prof_calls: list = []
    _patch_snapshot_cli(monkeypatch, prof_calls)
    hist = Path(sched.config.HISTORY_DIR)
    hist.mkdir(parents=True, exist_ok=True)
    f = hist / "freq_profession_2026.json"
    f.write_text("{}", encoding="utf-8")
    old = time.time() - 2 * 24 * 3600
    os.utime(f, (old, old))
    await sched._refresh_history_snapshots()
    assert len(prof_calls) == 1


async def test_refresh_prof_snapshots_no_files_runs(tmp_sched, monkeypatch):
    """При отсутствии готовых файлов снимок создаётся безусловно."""
    prof_calls: list = []
    _patch_snapshot_cli(monkeypatch, prof_calls)
    hist = Path(sched.config.HISTORY_DIR)
    hist.mkdir(parents=True, exist_ok=True)
    await sched._refresh_history_snapshots()
    assert len(prof_calls) == 1


async def test_refresh_prof_snapshots_cli_error_swallowed(tmp_sched, monkeypatch):
    """Ошибка CLI-снимка логируется и не роняет цикл сбора."""
    import src.cli.backfill_market_snapshots as bf
    import src.cli.snapshot_professions as sp

    monkeypatch.setattr(bf, "main", lambda *a, **k: None)
    monkeypatch.setattr(sp, "main", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    await sched._refresh_history_snapshots()


# ---------------------------------------------------------------------------
# scheduler_status / служебные чистые функции
# ---------------------------------------------------------------------------


def test_scheduler_status_next_collect_in_past_clamped(tmp_sched):
    """next_collect_in_s не бывает отрицательным: клампится к нулю.

    last_collect_ts уходит в прошлое дальше интервала - значит nxt = last + interval
    уже позади now, и in_s обязан получить кламп max(0, ...) == 0.
    """
    sched.save_scheduler_settings({"last_collect_ts": time.time() - 13 * 3600})
    sched._scheduler_busy = "collect"
    st = sched.scheduler_status()
    try:
        assert st["busy"] == "collect"
        assert st["pipeline_running"] is False
        assert st["next_collect_ts"] == sched._scheduler_cache["last_collect_ts"] + 12 * 3600
        assert st["next_collect_in_s"] == 0
    finally:
        sched._scheduler_busy = None


def test_scheduler_status_next_collect_in_future_positive(tmp_sched):
    """Будущий next_collect: in_s считается честно как nxt - now без искажений.

    last_collect_ts сдвинут на час вперёд - next_collect уходит за голый интервал.
    """
    ts = time.time() + 3600
    sched.save_scheduler_settings({"last_collect_ts": ts})
    st = sched.scheduler_status()
    assert st["next_collect_ts"] == ts + 12 * 3600
    assert st["next_collect_in_s"] > 12 * 3600, "last_ts в будущем отодвигает next за голый интервал"
    expected = ts + 12 * 3600 - time.time()
    assert abs(st["next_collect_in_s"] - expected) <= 1, "in_s == int(nxt - now) в пределах гонки секунд"


def test_scheduler_status_without_ts_has_no_next(tmp_sched):
    st = sched.scheduler_status()
    assert st["next_collect_ts"] is None
    assert st["next_collect_in_s"] == 0


def test_scheduler_status_error_branch(tmp_sched, monkeypatch):
    """Сбой чтения настроек не роняет статус: возвращается error-текст."""
    def _boom():
        raise ValueError("settings exploded")

    monkeypatch.setattr(sched, "load_scheduler_settings", _boom)
    st = sched.scheduler_status()
    assert "settings exploded" in st["error"]
    assert st["busy"] is None


def test_is_collect_due_cases():
    now = time.time()
    assert sched.is_collect_due(now, None, 12) is True
    assert sched.is_collect_due(now, now - 13 * 3600, 12) is True
    assert sched.is_collect_due(now, now, 12) is False


def test_is_collect_due_broken_inputs_fallback_true():
    assert sched.is_collect_due(time.time(), object(), 12) is True


def test_is_gap_overdue_cases():
    assert sched.is_gap_overdue("2026-11-08", None) is True
    assert sched.is_gap_overdue("2026-11-08", "2026-01-01", max_days=7) is True
    assert sched.is_gap_overdue("2026-11-08", "2026-11-07", max_days=7) is False
    assert sched.is_gap_overdue("2026-11-08", "garbage", max_days=7) is True


def test_pipeline_running_broken_registry_returns_false(tmp_sched, monkeypatch):
    """Сломанный реестр pipeline-задач приравнивается к 'ничего не работает'."""
    import src.api_pkg.routers.pipeline as pl

    class _BoomRegistry:
        def values(self):
            raise RuntimeError("registry corrupted")

    monkeypatch.setattr(pl, "pipeline_tasks", _BoomRegistry())
    assert sched._pipeline_running() is False


async def test_wait_until_ready_broken_flag_times_out(tmp_sched, monkeypatch):
    """Нечитаемый флаг готовности не даёт вечного зависания."""
    from src.api_pkg import deps as d

    class _BadReady:
        def __bool__(self):
            raise RuntimeError("is_ready is broken")

    monkeypatch.setattr(d, "is_ready", _BadReady(), raising=False)
    assert await sched._wait_until_ready(timeout_s=15) is False


# ---------------------------------------------------------------------------
# _scheduler_loop: тики, цепочки, сбои, отмена
# ---------------------------------------------------------------------------


def _loop_patcher(monkeypatch, collect_result=("collected", 1), real_breakers=True, ready=True):
    calls = {"collect": 0, "sleeps": 0, "gap": []}
    breakers: list = []

    async def _ready(*_a, **_kw):
        return ready

    class _RecorderBreaker:
        def __init__(self, fail_threshold=None, cooldown_s=None):
            self.fail_threshold = fail_threshold
            self.cooldown_s = cooldown_s
            self.successes = 0
            self.failures = 0

        def allow(self):
            return True

        def record_success(self):
            self.successes += 1

        def record_failure(self):
            self.failures += 1

    def _breaker_factory(fail_threshold=None, cooldown_s=None):
        b = _RecorderBreaker(fail_threshold, cooldown_s)
        breakers.append(b)
        return b

    async def _collect(*_a, **_kw):
        calls["collect"] += 1
        if isinstance(collect_result, Exception):
            raise collect_result
        return collect_result

    async def _gap(gap_breaker):
        calls["gap"].append(gap_breaker)
        return True

    async def _sleep(_s):
        calls["sleeps"] += 1
        raise KeyboardInterrupt
    monkeypatch.setattr(sched, "_wait_until_ready", _ready)
    monkeypatch.setattr(sched, "_run_collect_once", _collect)
    monkeypatch.setattr(sched, "_run_chained_gap", _gap)
    monkeypatch.setattr(sched.asyncio, "sleep", _sleep)
    monkeypatch.setattr(sched, "_pipeline_running", lambda: False)
    if not real_breakers:
        monkeypatch.setattr(sched, "CircuitBreaker", _breaker_factory)
    return calls, breakers


async def test_loop_not_ready_still_serves_ticks(tmp_sched, monkeypatch):
    """Неготовность API не мешает циклу дышать: тик проходится, сбор не стартует."""
    monkeypatch.setattr(sched.config.settings, "BACKGROUND_COLLECTOR_ENABLED", False)
    calls, _ = _loop_patcher(monkeypatch, ready=False)
    with pytest.raises(KeyboardInterrupt):
        await sched._scheduler_loop()
    assert calls["sleeps"] == 1
    assert calls["collect"] == 0


async def test_loop_collected_chains_gap_via_real_maybe(tmp_sched, monkeypatch):
    """collected-статус прогоняет chained gap через реальный _maybe_chained_gap.

    Ежедневный gap включён, last_gap_date=None (срок вышел), конвейер не занят -
    значит после успешного сбора цикл обязан дотянуть до _run_chained_gap
    ровно один раз и передать ему gap-брейкер.
    """
    monkeypatch.setattr(sched.config.settings, "SCHEDULER_DAILY_GAP_ENABLED", True)
    calls, breakers = _loop_patcher(monkeypatch, ("collected", 4), real_breakers=False)
    with pytest.raises(KeyboardInterrupt):
        await sched._scheduler_loop()
    assert calls["collect"] == 1
    assert len(calls["gap"]) == 1, "collected + включённый просроченный gap обязан дойти до _run_chained_gap"
    assert calls["gap"][0] is breakers[1], "gap обязан использовать gap_breaker"
    assert breakers[0].successes == 1, "collected ставит успех collect-брейкеру"


async def test_loop_collected_with_daily_disabled_skips_gap(tmp_sched, monkeypatch):
    """Собрано, но ежедневный gap выключен: _maybe_chained_gap выходит без запуска."""
    calls, breakers = _loop_patcher(monkeypatch, ("collected", 4), real_breakers=False)
    with pytest.raises(KeyboardInterrupt):
        await sched._scheduler_loop()
    assert calls["collect"] == 1
    assert calls["gap"] == [], "gap при выключенном daily-режиме запускаться не должен"
    assert breakers[0].successes == 1


async def test_loop_collected_empty_chains_gap_when_overdue(tmp_sched, monkeypatch):
    """collected-empty запускает chained gap, если снапшот протух."""
    monkeypatch.setattr(sched.config.settings, "SCHEDULER_DAILY_GAP_ENABLED", True)
    calls, breakers = _loop_patcher(
        monkeypatch, ("collected-empty", 0), real_breakers=False
    )
    with pytest.raises(KeyboardInterrupt):
        await sched._scheduler_loop()
    assert calls["collect"] == 1
    assert len(calls["gap"]) == 1
    assert calls["gap"][0] is breakers[1], "gap обязан использовать gap_breaker"
    assert breakers[0].successes == 1, "collected-empty ставит успех collect-брейкеру"


async def test_loop_collect_failed_records_breaker_failure(tmp_sched, monkeypatch):
    calls, breakers = _loop_patcher(monkeypatch, ("failed", 0), real_breakers=False)
    with pytest.raises(KeyboardInterrupt):
        await sched._scheduler_loop()
    assert breakers[0].failures == 1
    assert calls["gap"] == []


async def test_loop_tick_error_swallowed_and_cycle_continues(tmp_sched, monkeypatch):
    """Исключение тика логируется, цикл продолжает жить."""
    calls, _ = _loop_patcher(monkeypatch, RuntimeError("tick exploded"), real_breakers=False)
    with pytest.raises(KeyboardInterrupt):
        await sched._scheduler_loop()
    assert calls["sleeps"] == 1


async def test_loop_cancelled_error_propagates(tmp_sched, monkeypatch):
    """Отмена цикла остаётся отменой: CancelledError не глотается."""
    calls, _ = _loop_patcher(monkeypatch, real_breakers=False)
    async def _cancelled(*_a, **_kw):
        raise asyncio.CancelledError()

    monkeypatch.setattr(sched, "_run_collect_once", _cancelled)
    monkeypatch.setattr(sched.config.settings, "BACKGROUND_COLLECTOR_ENABLED", True)
    with pytest.raises(asyncio.CancelledError):
        await sched._scheduler_loop()
    assert calls["sleeps"] == 0


# ---------------------------------------------------------------------------
# load/save настроек планировщика: битые значения и отказ ФС
# ---------------------------------------------------------------------------


def test_load_settings_broken_json_falls_back_to_defaults(tmp_sched):
    settings_dir = tmp_sched / "settings"
    settings_dir.mkdir(parents=True)
    (settings_dir / "scheduler.json").write_text("{not json", encoding="utf-8")
    st = sched.load_scheduler_settings()
    assert st["collect_interval_hours"] == 12
    assert st["last_collect_ts"] is None


def test_save_settings_bool_flags_persist(tmp_sched):
    st = sched.save_scheduler_settings({"collector_enabled": True, "daily_gap_enabled": False})
    assert st["collector_enabled"] is True
    assert st["daily_gap_enabled"] is False
    assert sched.load_scheduler_settings()["collector_enabled"] is True


def test_save_settings_interval_invalid_and_clamped(tmp_sched):
    st = sched.save_scheduler_settings({"collect_interval_hours": "not a number"})
    assert st["collect_interval_hours"] == 12, "битое значение не меняет текущий интервал"
    st = sched.save_scheduler_settings({"collect_interval_hours": 500})
    assert st["collect_interval_hours"] == 72
    st = sched.save_scheduler_settings({"collect_interval_hours": 0})
    assert st["collect_interval_hours"] == 1
    st = sched.save_scheduler_settings({"collect_interval_hours": 3.7})
    assert st["collect_interval_hours"] == 3


def test_save_settings_atomic_failure_still_returns_merged(tmp_sched, monkeypatch):
    def _boom(data, path):
        raise OSError("read-only fs")

    monkeypatch.setattr(sched, "atomic_write_json", _boom)
    st = sched.save_scheduler_settings({"collector_enabled": True})
    assert st["collector_enabled"] is True
    assert not (tmp_sched / "settings" / "scheduler.json").exists()
