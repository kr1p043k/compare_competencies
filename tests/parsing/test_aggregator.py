"""Тесты синглтон-агрегатора источников вакансий."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src import Err, Ok
from src.parsing.api.aggregator import VacancyFeed, get_vacancy_feed, make_vacancy_client, reset_vacancy_feed


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_vacancy_feed()
    yield
    reset_vacancy_feed()


def _hh_stub(vacancies, found=0):
    api = MagicMock()
    api.search_vacancies.return_value = Ok(vacancies)
    api.last_response = {"found": found}
    api.supports_async_details = True
    api._token = None
    api._token_expires_at = 0
    return api


def _rabota_stub(vacancies, found=0, enabled=True):
    api = MagicMock()
    api.enabled = enabled
    api.search_vacancies.return_value = Ok(vacancies)
    api.last_response = {"found": found}
    api.supports_async_details = False
    return api


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_vacancy_feed("hh")
        b = get_vacancy_feed("hh")
        assert a is b

    def test_reset_clears_instance(self):
        a = get_vacancy_feed("hh")
        reset_vacancy_feed()
        b = get_vacancy_feed("hh")
        assert a is not b

    def test_explicit_source_rebuilds(self):
        a = get_vacancy_feed("hh")
        b = get_vacancy_feed("rabota")
        assert a is not b
        assert b.source == "rabota"

    def test_unknown_source_falls_back_to_hh(self):
        feed = VacancyFeed("nonsense")
        assert feed.source == "hh"


class TestSourceSelection:
    def test_hh_only_by_default(self):
        feed = VacancyFeed("hh")
        assert feed.enabled_sources() == ["hh"]

    def test_rabota_skipped_without_credentials(self):
        feed = VacancyFeed("both")
        feed._rabota = _rabota_stub([], enabled=False)
        assert feed.enabled_sources() == ["hh"]

    def test_both_with_credentials(self):
        feed = VacancyFeed("both")
        feed._rabota = _rabota_stub([], enabled=True)
        assert feed.enabled_sources() == ["hh", "rabota"]

    def test_auto_adds_rabota_when_ready(self):
        feed = VacancyFeed("auto")
        feed._rabota = _rabota_stub([], enabled=True)
        assert feed.enabled_sources() == ["hh", "rabota"]

    def test_rabota_source_without_creds_has_no_sources(self):
        feed = VacancyFeed("rabota")
        feed._rabota = _rabota_stub([], enabled=False)
        assert feed.enabled_sources() == []


class TestSearch:
    def test_no_sources_returns_err(self):
        feed = VacancyFeed("rabota")
        feed._rabota = _rabota_stub([], enabled=False)
        assert feed.search_vacancies(text="x", area=1).is_err()

    def test_hh_search_passthrough(self):
        feed = VacancyFeed("hh")
        feed._hh = _hh_stub([{"id": "1"}], found=5)
        match feed.search_vacancies(text="python", area=1):
            case Ok(vacs):
                assert vacs == [{"id": "1"}]
            case Err(e):
                pytest.fail(str(e))
        assert feed.last_response["found"] == 5

    def test_both_merges_and_dedupes(self):
        feed = VacancyFeed("both")
        feed._hh = _hh_stub([{"id": "1", "name": "Python dev",
                              "employer": {"name": "ACME"}, "area": {"name": "Москва"}}], found=3)
        feed._rabota = _rabota_stub([
            {"id": "1000000001", "name": "Python dev",
             "employer": {"name": "ACME"}, "area": {"name": "Москва"}},
            {"id": "1000000002", "name": "Java dev",
             "employer": {"name": "ACME"}, "area": {"name": "Москва"}},
        ], enabled=True, found=2)
        match feed.search_vacancies(text="python", area=1):
            case Ok(vacs):
                assert len(vacs) == 2
            case Err(e):
                pytest.fail(str(e))

    def test_one_source_failing_keeps_other(self):
        feed = VacancyFeed("both")
        feed._hh = _hh_stub([{"id": "1"}])
        rabota = _rabota_stub([], enabled=True)
        rabota.search_vacancies.return_value = Err(MagicMock(message="down"))
        feed._rabota = rabota
        match feed.search_vacancies(text="x", area=1):
            case Ok(vacs):
                assert len(vacs) == 1
            case Err(e):
                pytest.fail(str(e))

    def test_all_sources_failing_is_err(self):
        feed = VacancyFeed("both")
        hh = _hh_stub([])
        hh.search_vacancies.return_value = Err(MagicMock(message="hh down"))
        feed._hh = hh
        rabota = _rabota_stub([], enabled=True)
        rabota.search_vacancies.return_value = Err(MagicMock(message="rabota down"))
        feed._rabota = rabota
        assert feed.search_vacancies(text="x", area=1).is_err()

    def test_dedupe_ignores_incomplete_fingerprints(self):
        merged = VacancyFeed._dedupe([
            {"id": "1", "name": "", "employer": {}, "area": {}},
            {"id": "2", "name": "", "employer": {}, "area": {}},
        ])
        assert len(merged) == 2


class TestRouting:
    def test_supports_async_only_for_hh(self):
        assert VacancyFeed("hh").supports_async_details is True
        feed = VacancyFeed("rabota")
        feed._rabota = _rabota_stub([], enabled=True)
        assert feed.supports_async_details is False

    def test_details_route_by_id_offset(self):
        feed = VacancyFeed("both")
        feed._hh = _hh_stub([])
        rabota = _rabota_stub([], enabled=True)
        feed._rabota = rabota
        feed.get_vacancy_details("1500000000")
        rabota.get_vacancy_details.assert_called_once_with("1500000000")
        feed._hh.get_vacancy_details.assert_not_called()

    def test_details_route_hh_for_small_id(self):
        feed = VacancyFeed("both")
        feed._hh = _hh_stub([])
        rabota = _rabota_stub([], enabled=True)
        feed._rabota = rabota
        feed.get_vacancy_details("12345")
        feed._hh.get_vacancy_details.assert_called_once_with("12345")
        rabota.get_vacancy_details.assert_not_called()

    def test_token_proxy_from_hh(self):
        feed = VacancyFeed("hh")
        feed._hh = _hh_stub([])
        feed._hh._token = "abc"
        assert feed._token == "abc"


class TestFactory:
    def test_make_client_hh_returns_factory_result(self):
        sentinel = object()
        assert make_vacancy_client(lambda: sentinel, "hh") is sentinel

    def test_make_client_rabota_returns_feed(self):
        client = make_vacancy_client(lambda: None, "rabota")
        assert isinstance(client, VacancyFeed)
        assert client.source == "rabota"

    def test_make_client_passes_hh_factory_into_feed(self):
        sentinel = object()
        feed = make_vacancy_client(lambda: sentinel, "both")
        assert feed._hh_client() is sentinel
