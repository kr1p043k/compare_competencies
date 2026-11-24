"""Тесты клиента rabota.ru: подпись, маппинг в hh-формат, пагинация, ошибки."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src import Err, Ok
from src.models.vacancy import Vacancy
from src.parsing.api.rabota_api import (
    RabotaAPI,
    _canonical_json,
    _currency_code,
    compute_signature,
    rabota_to_hh,
)


class TestSignature:
    def test_matches_doc_example(self):
        """Эталон из dev.rabota.ru: подпись должна совпасть побайтово."""
        params = {
            "app_id": "3803",
            "time": "1551787641",
            "code": "3MS3zSsNkyBG1gDlLiApdE7KAOQnG1b0",
        }
        secret = "74JbXYMUR306MHTz0VnCiU5prNv3lO7f"
        assert compute_signature(params, secret) == (
            "8d240c87e944740862b3ebd261ceb6db555268f59b421ffdee012f7737bc2855"
        )

    def test_canonical_json_sorts_keys_recursively(self):
        value = {"b": 1, "a": {"z": [1, {"y": 2, "x": 3}], "c": "строка"}}
        assert _canonical_json(value) == '{"a":{"c":"\\u0441\\u0442\\u0440\\u043e\\u043a\\u0430","z":[1,{"x":3,"y":2}]},"b":1}'

    def test_signature_depends_on_values(self):
        a = compute_signature({"app_id": 1, "time": 10, "request": {"query": "a"}}, "s")
        b = compute_signature({"app_id": 1, "time": 10, "request": {"query": "b"}}, "s")
        assert a != b and len(a) == 64


class TestMapping:
    RAW = {
        "id": 54468063,
        "title": "Стажер Python разработчик",
        "salary": {"from": 40000, "to": 60000, "currency": "руб./мес.", "pay_type": "net"},
        "description": "<p>Текст</p>",
        "short_description": "Кратко",
        "places": [
            {"id": 1, "name": "г Москва", "region": {"id": 3, "name": "Москва"}},
        ],
        "company": {"id": 5281270, "name": "ACME", "is_verified": True, "logo": None},
        "skills": [{"id": 1088, "name": "Python"}, {"id": 1089, "name": "SQL"}],
        "experience": {"id": 5, "name": "от 3 лет"},
        "publish_start_at": "2026-11-11T15:47:03+03:00",
        "professional_areas": [{"id": 2, "name": "Программист"}],
    }

    def test_maps_to_hh_shape(self):
        out = rabota_to_hh(self.RAW, id_offset=1_000_000_000)
        assert out["id"] == "1054468063"
        assert out["rabota_id"] == 54468063
        assert out["name"] == "Стажер Python разработчик"
        assert out["area"] == {"id": 3, "name": "Москва"}
        assert out["employer"]["name"] == "ACME"
        assert out["employer"]["trusted"] is True
        assert [s["name"] for s in out["key_skills"]] == ["Python", "SQL"]
        assert out["salary"] == {"from": 40000, "to": 60000, "currency": "RUB", "gross": False}
        assert out["experience"] == {"id": "between3And6", "name": "от 3 лет"}
        assert out["published_at"] == "2026-11-11T15:47:03+03:00"
        assert out["alternate_url"] == "https://www.rabota.ru/vacancy/54468063/"
        assert out["source"] == "rabota"
        assert out["snippet"]["requirement"] == "Кратко"

    def test_vacancy_from_api_accepts_mapped_dict(self):
        vac = Vacancy.from_api(rabota_to_hh(self.RAW))
        assert vac.name == "Стажер Python разработчик"
        assert vac.area.name == "Москва"
        assert vac.get_skill_names() == ["Python", "SQL"]
        assert vac.salary.from_amount == 40000

    def test_id_offset_avoids_hh_collision(self):
        a = rabota_to_hh({**self.RAW, "id": 1})
        b = rabota_to_hh({**self.RAW, "id": 1}, id_offset=1_000_000_000)
        assert a["id"] != b["id"]

    def test_requires_id_and_title(self):
        with pytest.raises(ValueError):
            rabota_to_hh({"title": "x"})
        with pytest.raises(ValueError):
            rabota_to_hh({"id": 1, "title": "  "})

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("руб./мес.", "RUB"),
            ("₽", "RUB"),
            ("USD", "USD"),
            ("", "RUB"),
            (None, "RUB"),
        ],
    )
    def test_currency(self, raw, expected):
        assert _currency_code(raw) == expected


class TestAreaMapping:
    def test_hh_area_1_is_moscow(self):
        api = RabotaAPI(app_id=1, secret="s", request_delay=0)
        assert api.map_area(1) == 3

    def test_hh_area_2_is_spb(self):
        api = RabotaAPI(app_id=1, secret="s", request_delay=0)
        assert api.map_area(2) == 4

    def test_unknown_area_passes_through(self):
        api = RabotaAPI(app_id=1, secret="s", request_delay=0)
        assert api.map_area(481) == 481


class TestSearch:
    def _api(self, **kw):
        return RabotaAPI(app_id=983, secret="s", request_delay=0, **kw)

    def test_disabled_without_credentials(self):
        api = RabotaAPI(app_id=None, secret=None, request_delay=0)
        api._app_id = None
        api._secret = None
        result = api.search_vacancies(text="python", area=1)
        assert result.is_err()

    def test_search_maps_and_caches_details(self):
        api = self._api()
        payload = {
            "response": {
                "total": 1,
                "vacancies": [
                    {
                        "id": 10,
                        "title": "Python dev",
                        "places": [{"region": {"id": 3, "name": "Москва"}}],
                        "company": {"id": 1, "name": "ACME"},
                        "skills": [{"id": 1, "name": "Python"}],
                    }
                ],
            }
        }
        with patch.object(api, "_post", return_value=Ok(payload)):
            match api.search_vacancies(text="python", area=1, max_pages=1):
                case Ok(vacancies):
                    assert len(vacancies) == 1
                case Err(e):
                    pytest.fail(f"unexpected error: {e}")
        mapped_id = str(10 + 1_000_000_000)
        assert api.get_vacancy_details(mapped_id).is_ok()
        assert api.last_response["found"] == 1

    def test_pagination_accumulates_pages(self):
        api = self._api()
        page1 = {"response": {"total": 150, "vacancies": [
            {"id": i, "title": f"v{i}", "places": [], "company": {}, "skills": []} for i in range(100)
        ]}}
        page2 = {"response": {"total": 150, "vacancies": [
            {"id": i, "title": f"v{i}", "places": [], "company": {}, "skills": []} for i in range(100, 150)
        ]}}
        with patch.object(api, "_post", side_effect=[Ok(page1), Ok(page2)]) as mock_post:
            match api.search_vacancies(text="", area=1, max_pages=5, per_page=100, it_only=False):
                case Ok(vacancies):
                    assert len(vacancies) == 150
                case Err(e):
                    pytest.fail(f"unexpected error: {e}")
        assert mock_post.call_count == 2
        second_offset = mock_post.call_args_list[1].args[1]["offset"]
        assert second_offset == 100

    def test_partial_results_kept_when_later_page_fails(self):
        api = self._api()
        page1 = {"response": {"total": 200, "vacancies": [
            {"id": i, "title": f"v{i}", "places": [], "company": {}, "skills": []} for i in range(100)
        ]}}
        with patch.object(api, "_post", side_effect=[Ok(page1), Err(MagicMock(message="boom"))]):
            match api.search_vacancies(text="", area=1, max_pages=5, it_only=False):
                case Ok(vacancies):
                    assert len(vacancies) == 100
                case Err(e):
                    pytest.fail(f"should keep partial results: {e}")

    def test_first_page_error_returns_err(self):
        api = self._api()
        with patch.object(api, "_post", return_value=Err(MagicMock(message="bad"))):
            assert api.search_vacancies(text="x", area=1).is_err()

    def test_per_page_clamped_to_100(self):
        api = self._api()
        captured = {}

        def fake_post(path, request, *, endpoint):
            captured["limit"] = request["limit"]
            return Ok({"response": {"total": 0, "vacancies": []}})

        with patch.object(api, "_post", side_effect=fake_post):
            api.search_vacancies(text="x", area=1, per_page=5000)
        assert captured["limit"] == 100

    def test_post_error_paths(self):
        api = self._api()
        resp = MagicMock(status_code=200, headers={"content-type": "text/html"}, text="<html>WAF</html>")
        with patch.object(api.session, "post", return_value=resp):
            result = api._post("/outer_clients/vacancies/search.json", {}, endpoint="search")
        assert result.is_err()
        assert "HTML" in result.err().message

    def test_post_success_returns_json(self):
        api = self._api()
        resp = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"response": {"total": 0, "vacancies": []}}),
        )
        with patch.object(api.session, "post", return_value=resp) as mock_post:
            result = api._post("/outer_clients/vacancies/search.json", {"query": "x"}, endpoint="search")
        assert result.is_ok()
        sent = json.loads(mock_post.call_args.kwargs["data"].decode("utf-8"))
        assert sent["application_id"] == 983
        assert len(sent["signature"]) == 64

    def test_request_timeout_is_err(self):
        import requests

        api = self._api()
        with patch.object(api.session, "post", side_effect=requests.exceptions.Timeout()):
            assert api._post("/x", {}, endpoint="x").is_err()


class TestRetryBackoff:
    """6.4.2: сбор устойчив к 429/5xx - повторы с Retry-After и джиттером."""

    @staticmethod
    def _api(**kwargs) -> RabotaAPI:
        return RabotaAPI(app_id=983, secret="s", request_delay=0, **kwargs)

    @staticmethod
    def _json_ok(payload=None):
        return MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value=payload or {"response": {"total": 0, "vacancies": []}}),
        )

    @staticmethod
    def _err_resp(status, headers=None, text="err"):
        return MagicMock(status_code=status, headers=headers or {}, text=text)

    def test_429_honours_retry_after_and_retries(self):
        api = self._api()
        responses = [self._err_resp(429, {"Retry-After": "7"}), self._json_ok()]
        with patch.object(api.session, "post", side_effect=responses) as mock_post, \
                patch("src.parsing.api.rabota_api.time.sleep") as mock_sleep:
            result = api._post("/x", {}, endpoint="x")
        assert result.is_ok()
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(7.0)

    def test_429_exhausted_returns_rate_limit_error(self):
        from src.errors import RateLimitError

        api = self._api(max_attempts=3)
        with patch.object(api.session, "post", return_value=self._err_resp(429, {"Retry-After": "2"})) as mock_post, \
                patch("src.parsing.api.rabota_api.time.sleep"):
            result = api._post("/x", {}, endpoint="x")
        assert result.is_err()
        err = result.err()
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 2.0
        assert mock_post.call_count == 3

    def test_503_retried_then_succeeds(self):
        api = self._api()
        responses = [self._err_resp(503), self._err_resp(502), self._json_ok()]
        with patch.object(api.session, "post", side_effect=responses) as mock_post, \
                patch("src.parsing.api.rabota_api.time.sleep"):
            assert api._post("/x", {}, endpoint="x").is_ok()
        assert mock_post.call_count == 3

    def test_backoff_grows_exponentially_without_retry_after(self):
        api = self._api(retry_base_delay=1.0, retry_max_delay=30.0)
        delays: list[float] = []
        responses = [self._err_resp(500), self._err_resp(500), self._json_ok()]
        with patch.object(api.session, "post", side_effect=responses), \
                patch("src.parsing.api.rabota_api.time.sleep", side_effect=delays.append), \
                patch("src.parsing.api.rabota_api.random.uniform", return_value=0.0):
            assert api._post("/x", {}, endpoint="x").is_ok()
        assert delays == [1.0, 2.0]

    def test_retry_after_capped_by_max_delay(self):
        api = self._api(retry_max_delay=5.0)
        responses = [self._err_resp(429, {"Retry-After": "9999"}), self._json_ok()]
        with patch.object(api.session, "post", side_effect=responses), \
                patch("src.parsing.api.rabota_api.time.sleep") as mock_sleep:
            assert api._post("/x", {}, endpoint="x").is_ok()
        mock_sleep.assert_called_once_with(5.0)

    def test_invalid_retry_after_falls_back_to_backoff(self):
        api = self._api(retry_base_delay=2.0)
        responses = [
            self._err_resp(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
            self._json_ok(),
        ]
        with patch.object(api.session, "post", side_effect=responses), \
                patch("src.parsing.api.rabota_api.time.sleep") as mock_sleep, \
                patch("src.parsing.api.rabota_api.random.uniform", return_value=0.0):
            assert api._post("/x", {}, endpoint="x").is_ok()
        mock_sleep.assert_called_once_with(2.0)

    def test_single_attempt_disables_retry(self):
        from src.errors import RateLimitError

        api = self._api(max_attempts=1)
        with patch.object(api.session, "post", return_value=self._err_resp(429)) as mock_post, \
                patch("src.parsing.api.rabota_api.time.sleep") as mock_sleep:
            result = api._post("/x", {}, endpoint="x")
        assert isinstance(result.err(), RateLimitError)
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    def test_client_error_is_not_retried(self):
        api = self._api()
        with patch.object(api.session, "post", return_value=self._err_resp(400, text="bad request")) as mock_post, \
                patch("src.parsing.api.rabota_api.time.sleep") as mock_sleep:
            result = api._post("/x", {}, endpoint="x")
        assert result.is_err()
        assert result.err().status_code == 400
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    def test_408_is_retried(self):
        api = self._api()
        responses = [self._err_resp(408), self._json_ok()]
        with patch.object(api.session, "post", side_effect=responses) as mock_post, \
                patch("src.parsing.api.rabota_api.time.sleep"):
            assert api._post("/x", {}, endpoint="x").is_ok()
        assert mock_post.call_count == 2

    def test_status_retries_disabled_in_urllib3_adapter(self):
        """Повторы статусов делает _post: адаптер не должен дублировать их молча."""
        api = self._api()
        adapter = api.session.get_adapter("https://api.rabota.ru/v5")
        assert list(adapter.max_retries.status_forcelist) == []


class TestDetails:
    def test_not_cached_returns_vacancy_not_found(self):
        from src.errors import VacancyNotFoundError

        api = RabotaAPI(app_id=1, secret="s", request_delay=0)
        result = api.get_vacancy_details("42")
        assert result.is_err()
        assert isinstance(result.err(), VacancyNotFoundError)

    def test_details_as_object(self):
        api = RabotaAPI(app_id=1, secret="s", request_delay=0)
        api._details_cache["7"] = rabota_to_hh(
            {"id": 7, "title": "QA", "places": [{"region": {"id": 3, "name": "Москва"}}], "company": {}, "skills": []}
        )
        result = api.get_vacancy_details_as_object("7")
        assert result.is_ok()
        assert result.ok().name == "QA"


class TestItFilter:
    """Работа.ру не фильтрует по профессии, поэтому IT режем локально."""

    def test_tree_01_is_it(self):
        from src.parsing.api.rabota_api import _is_it_vacancy

        assert _is_it_vacancy({
            "title": "Специалист",
            "professional_areas": [{"id": 2, "name": "Программист", "tree": "01.01.01.01"}],
        })

    def test_non_it_tree_is_rejected(self):
        from src.parsing.api.rabota_api import _is_it_vacancy

        assert not _is_it_vacancy({
            "title": "Менеджер на выкладку товара",
            "professional_areas": [{"id": 7, "name": "Продавец-консультант", "tree": "07.01.01.04"}],
            "top_position": {"id": 1, "name": "товаровед"},
        })

    @pytest.mark.parametrize("title", [
        "Python-разработчик",
        "Ведущий Java разработчик",
        "DevOps инженер",
        "Системный аналитик",
        "Тестировщик (QA)",
        "1С программист",
        "Data Scientist",
        "ML инженер",
        "Сетевой инженер",
        "Специалист по информационной безопасности",
    ])
    def test_it_titles_pass_without_areas(self, title):
        from src.parsing.api.rabota_api import _is_it_vacancy

        assert _is_it_vacancy({"title": title, "professional_areas": []})

    @pytest.mark.parametrize("title", [
        "Педагог дополнительного образования",
        "Менеджер по продажам",
        "Водитель категории E",
        "Бухгалтер",
        "Повар",
    ])
    def test_non_it_titles_rejected_without_areas(self, title):
        from src.parsing.api.rabota_api import _is_it_vacancy

        assert not _is_it_vacancy({"title": title, "professional_areas": []})

    def _api(self):
        return RabotaAPI(app_id=983, secret="s", request_delay=0)

    def _payload(self):
        return {"response": {"total": 3, "vacancies": [
            {"id": 1, "title": "Python разработчик", "places": [], "company": {},
             "skills": [], "professional_areas": [{"id": 2, "tree": "01.01.01.01"}]},
            {"id": 2, "title": "Педагог дополнительного образования", "places": [],
             "company": {}, "skills": [], "professional_areas": [{"id": 99, "tree": "12.01.01.01"}]},
            {"id": 3, "title": "Менеджер по продажам", "places": [], "company": {},
             "skills": [], "professional_areas": [{"id": 98, "tree": "07.01.01.02"}]},
        ]}}

    def test_search_drops_non_it_by_default(self):
        api = self._api()
        with patch.object(api, "_post", return_value=Ok(self._payload())):
            match api.search_vacancies(text="", area=1, max_pages=1, it_only=True):
                case Ok(vacancies):
                    assert [v["name"] for v in vacancies] == ["Python разработчик"]
                case Err(e):
                    pytest.fail(f"unexpected error: {e}")
        assert api.last_response["filtered_out"] == 2

    def test_search_keeps_non_it_when_disabled(self):
        api = self._api()
        with patch.object(api, "_post", return_value=Ok(self._payload())):
            match api.search_vacancies(text="", area=1, max_pages=1, it_only=False):
                case Ok(vacancies):
                    assert len(vacancies) == 3
                case Err(e):
                    pytest.fail(f"unexpected error: {e}")

    def test_it_only_defaults_to_config(self, monkeypatch):
        import src.parsing.api.rabota_api as mod

        monkeypatch.setattr(mod.config, "RABOTA_IT_ONLY", True)
        api = self._api()
        with patch.object(api, "_post", return_value=Ok(self._payload())):
            result = api.search_vacancies(text="", area=1, max_pages=1)
        assert len(result.ok()) == 1

    @pytest.mark.parametrize("tree", [
        "01.01.01.01",  # программист
        "01.01.02.04",  # 1С
        "01.01.03.09",  # системный аналитик
        "01.01.04.12",  # информационная безопасность
    ])
    def test_it_subtrees_accepted(self, tree):
        from src.parsing.api.rabota_api import _is_it_vacancy

        assert _is_it_vacancy({"title": "Специалист", "professional_areas": [{"tree": tree}]})

    @pytest.mark.parametrize("tree", [
        "01.01.06.18",  # продажи, работа с клиентами (внутри 01.01!)
        "01.01.07.21",  # интернет-маркетинг, SMM
        "01.01.08.26",  # контент, модерация
        "01.01.09.27",  # call-центр
        "01.01.11.34",  # другое
        "14.01.01.01",  # отрасль "IT / Интернет / Телеком" - слишком широко
        "07.01.01.02",  # менеджер по продажам
    ])
    def test_non_it_subtrees_rejected(self, tree):
        from src.parsing.api.rabota_api import _is_it_vacancy

        assert not _is_it_vacancy({"title": "Специалист", "professional_areas": [{"tree": tree}]})

    @pytest.mark.parametrize("title", [
        "Преподаватель по программированию",
        "Преподаватель программирования/робототехники",
        "Учитель информатики",
        "Методист по IT",
    ])
    def test_education_roles_excluded_even_with_it_tree(self, title):
        from src.parsing.api.rabota_api import _is_it_vacancy

        assert not _is_it_vacancy({
            "title": title,
            "professional_areas": [{"tree": "01.01.01.01"}],
        })

    def test_verified_city_pairs(self):
        api = RabotaAPI(app_id=1, secret="s", request_delay=0)
        assert api.map_area(1) == 3     # Москва
        assert api.map_area(2) == 4     # Санкт-Петербург
        assert api.map_area(3) == 328   # Екатеринбург (у hh 3 - Екатеринбург!)
        assert api.map_area(88) == 481  # Казань
        assert api.map_area(66) == 231  # Нижний Новгород
        assert api.map_area(76) == 284  # Ростов-на-Дону
        assert api.map_area(78) == 299  # Самара
        assert api.map_area(99) == 529  # Уфа
        assert api.map_area(54) == 25   # Красноярск

    def test_native_rabota_id_passes_through(self, monkeypatch):
        import src.parsing.api.rabota_api as mod

        monkeypatch.setattr(mod.config, "RABOTA_REGION_IDS", [3, 4, 481])
        api = RabotaAPI(app_id=1, secret="s", request_delay=0)
        assert api.map_area(481) == 481

    def test_unknown_area_falls_back_to_default(self, monkeypatch):
        import src.parsing.api.rabota_api as mod

        monkeypatch.setattr(mod.config, "RABOTA_REGION_IDS", [3, 4])
        monkeypatch.setattr(mod.config, "RABOTA_DEFAULT_REGION_ID", 3)
        api = RabotaAPI(app_id=1, secret="s", request_delay=0)
        assert api.map_area(999999) == 3

    def test_get_areas_returns_all_known_pairs(self):
        from src.parsing.api.rabota_api import HH_AREA_TO_RABOTA_REGION

        api = RabotaAPI(app_id=1, secret="s", request_delay=0)
        areas = api.get_areas().ok()
        assert len(areas) == len(HH_AREA_TO_RABOTA_REGION)
        names = {a["name"] for a in areas}
        assert {"Москва", "Санкт-Петербург", "Казань", "Екатеринбург"} <= names
