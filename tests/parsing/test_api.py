# tests/parsing/test_api.py
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import asyncio
import aiohttp
import pytest
import requests
from aioresponses import aioresponses

from src.models.vacancy import Vacancy
from src.parsing.hh_api import HeadHunterAPI
from src.parsing.hh_api_async import HeadHunterAPIAsync


class TestHeadHunterAPISync:
    """Тесты синхронного клиента HeadHunterAPI"""

    def test_init_creates_session_with_retries(self):
        api = HeadHunterAPI()
        assert api.session is not None
        assert "User-Agent" in api.session.headers
        api.close()

    @patch("src.parsing.hh_api.HeadHunterAPI._get")
    def test_search_vacancies_success(self, mock_get):
        api = HeadHunterAPI()
        mock_get.return_value = {"items": [{"id": "1"}, {"id": "2"}], "pages": 1, "found": 2}
        result = api.search_vacancies(text="Python", area=1)
        assert len(result) == 2
        mock_get.assert_called_once()

    @patch("src.parsing.hh_api.HeadHunterAPI._get")
    def test_search_vacancies_pagination(self, mock_get):
        api = HeadHunterAPI()
        mock_get.side_effect = [
            {"items": [{"id": "1"}, {"id": "2"}], "pages": 2, "found": 3},
            {"items": [{"id": "3"}], "pages": 2, "found": 3},
        ]
        result = api.search_vacancies(text="Java", area=2, max_pages=5)
        assert len(result) == 3
        assert mock_get.call_count == 2

    @patch("src.parsing.hh_api.HeadHunterAPI._get")
    def test_search_vacancies_stops_on_empty_page(self, mock_get):
        api = HeadHunterAPI()
        mock_get.return_value = {"items": [], "pages": 5}
        result = api.search_vacancies(text="C++", area=1)
        assert result == []

    @patch("src.parsing.hh_api.HeadHunterAPI._get")
    def test_get_vacancy_details_success(self, mock_get):
        api = HeadHunterAPI()
        mock_get.return_value = {"id": "123", "name": "Test"}
        details = api.get_vacancy_details("123")
        assert details["id"] == "123"

    def test_get_vacancy_details_returns_none_on_error(self):
        api = HeadHunterAPI()
        with patch.object(api, "_get", return_value=None):
            details = api.get_vacancy_details("999")
            assert details is None

    def test_get_vacancy_details_as_object_valid(self):
        api = HeadHunterAPI()
        raw = {
            "id": "456",
            "name": "Python Dev",
            "area": {"id": 1, "name": "Москва"},
            "employer": {"id": "10", "name": "Company"},
            "key_skills": [{"name": "Python"}, {"name": "Django"}],
        }
        with patch.object(api, "_get", return_value=raw):
            vacancy = api.get_vacancy_details_as_object("456")
            assert isinstance(vacancy, Vacancy)
            assert vacancy.id == "456"
            assert len(vacancy.key_skills) == 2

    def test_get_vacancy_details_as_object_invalid(self):
        api = HeadHunterAPI()
        with patch.object(api, "_get", return_value={"id": "no_name"}):
            vacancy = api.get_vacancy_details_as_object("bad")
            assert vacancy is None

    def test_get_handles_429_retry(self):
        api = HeadHunterAPI()
        with patch.object(api.session, "get") as mock_session_get:
            rate_limit_response = Mock(status_code=429, headers={"Retry-After": "1"})
            success_response = Mock(status_code=200)
            success_response.json.return_value = {"result": "ok"}
            mock_session_get.side_effect = [rate_limit_response, success_response]
            result = api._get("https://test.url")
            assert result == {"result": "ok"}
            assert mock_session_get.call_count == 2

    def test_get_handles_403_forbidden(self):
        api = HeadHunterAPI()
        with patch.object(api.session, "get") as mock_session_get:
            mock_session_get.return_value = Mock(status_code=403)
            result = api._get("https://blocked.url")
            assert result is None

    def test_get_handles_timeout(self):
        api = HeadHunterAPI()
        with patch.object(api.session, "get", side_effect=requests.exceptions.Timeout):
            result = api._get("https://timeout.url")
            assert result is None

    def test_context_manager_closes_session(self):
        with patch.object(HeadHunterAPI, "close") as mock_close:
            with HeadHunterAPI():
                pass
            mock_close.assert_called_once()

    def test_search_vacancies_with_max_pages_one(self):
        api = HeadHunterAPI()
        with patch.object(api, "_get") as mock_get:
            mock_get.return_value = {"items": [{"id": "1"}], "pages": 1}
            result = api.search_vacancies(text="Python", area=1, max_pages=1)
            assert len(result) == 1
            mock_get.assert_called_once()

    def test_get_handles_304_not_modified(self):
        api = HeadHunterAPI()
        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value = Mock(status_code=304)
            result = api._get("https://test.url")
            assert result is None

    def test_get_handles_unexpected_status(self):
        api = HeadHunterAPI()
        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value = Mock(status_code=418)
            result = api._get("https://test.url")
            assert result is None

    def test_get_handles_general_exception(self):
        api = HeadHunterAPI()
        with patch.object(api.session, "get", side_effect=Exception("Boom")):
            result = api._get("https://test.url")
            assert result is None

    def test_get_app_token_success(self, monkeypatch):
        """Строка 51: успешное получение токена"""
        monkeypatch.setattr("src.parsing.hh_api.config.HH_CLIENT_ID", "test_id")
        monkeypatch.setattr("src.parsing.hh_api.config.HH_CLIENT_SECRET", "test_secret")

        api = HeadHunterAPI()
        with patch.object(api.session, "post") as mock_post:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = {
                "access_token": "app_token_123",
                "expires_in": 86400,
            }
            mock_post.return_value = mock_response

            api._get_app_token()
            assert api._token == "app_token_123"

    def test_get_app_token_failure(self, monkeypatch):
        """Строки 75-76: ошибка получения токена"""
        monkeypatch.setattr("src.parsing.hh_api.config.HH_CLIENT_ID", "test_id")
        monkeypatch.setattr("src.parsing.hh_api.config.HH_CLIENT_SECRET", "test_secret")

        api = HeadHunterAPI()
        with patch.object(api.session, "post") as mock_post:
            mock_response = Mock(status_code=400)
            mock_response.text = "Bad request"
            mock_post.return_value = mock_response

            api._get_app_token()
            assert api._token is None

    def test_get_app_token_exception(self, monkeypatch):
        """Строки 75-76: исключение при получении токена"""
        monkeypatch.setattr("src.parsing.hh_api.config.HH_CLIENT_ID", "test_id")
        monkeypatch.setattr("src.parsing.hh_api.config.HH_CLIENT_SECRET", "test_secret")

        api = HeadHunterAPI()
        with patch.object(api.session, "post", side_effect=Exception("Network error")):
            api._get_app_token()
            assert api._token is None

    def test_ensure_token_expired(self):
        """Строки 99-100: проверка истечения токена"""
        api = HeadHunterAPI()
        api._token = "old_token"
        api._token_expires_at = time.time() - 100  # истёк

        with patch.object(api, "_get_app_token") as mock_get_token:
            api._ensure_token()
            mock_get_token.assert_called_once()

    def test_ensure_token_missing(self):
        """Строки 99-100: токен отсутствует"""
        api = HeadHunterAPI()
        api._token = None

        with patch.object(api, "_get_app_token") as mock_get_token:
            api._ensure_token()
            mock_get_token.assert_called_once()

    def test_ensure_token_valid(self):
        """Строки 99-100: токен валиден — не обновляется"""
        api = HeadHunterAPI()
        api._token = "valid_token"
        api._token_expires_at = time.time() + 3600

        with patch.object(api, "_get_app_token") as mock_get_token:
            api._ensure_token()
            mock_get_token.assert_not_called()

    def test_search_vacancies_with_date_range(self):
        """Строки 112-113: поиск с date_from/date_to"""
        api = HeadHunterAPI()
        with patch.object(api, "_get") as mock_get:
            mock_get.return_value = {"items": [{"id": "1"}], "pages": 1, "found": 1}
            result = api.search_vacancies(
                text="Python", area=1, date_from="2024-01-01", date_to="2024-01-31"
            )
            assert len(result) == 1
            call_args = mock_get.call_args[1]["params"]
            assert "date_from" in call_args
            assert "date_to" in call_args

    def test_get_vacancy_details_returns_none(self):
        """Строка 138: детали не получены"""
        api = HeadHunterAPI()
        with patch.object(api, "_get", return_value=None):
            result = api.get_vacancy_details("999")
            assert result is None

    def test_get_429_exceeded_retries(self):
        """Строки 161-162: превышены попытки 429"""
        api = HeadHunterAPI()
        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value = Mock(status_code=429, headers={"Retry-After": "0"})
            with patch("time.sleep", return_value=None):
                result = api._get("https://test.url", retry_count=3)
                assert result is None

    def test_get_401_retry(self):
        """Строки 164-170: 401 → обновление токена и повтор"""
        api = HeadHunterAPI()
        with patch.object(api.session, "get") as mock_get:
            response_401 = Mock(status_code=401)
            response_200 = Mock(status_code=200)
            response_200.json.return_value = {"result": "ok"}
            mock_get.side_effect = [response_401, response_200]

            with patch.object(api, "_get_app_token"):
                result = api._get("https://test.url")
                assert result == {"result": "ok"}

    def test_get_401_double_fail(self):
        """Строки 181-182: повторный 401 → None"""
        api = HeadHunterAPI()
        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value = Mock(status_code=401)
            with patch.object(api, "_get_app_token"):
                result = api._get("https://test.url", _is_retry_after_401=True)
                assert result is None

    def test_get_connection_error(self):
        """Строки 190-191: ошибка соединения"""
        api = HeadHunterAPI()
        with patch.object(api.session, "get", side_effect=requests.exceptions.ConnectionError):
            result = api._get("https://test.url")
            assert result is None

class TestHeadHunterAPIAsync:
    """Тесты асинхронного клиента HeadHunterAPIAsync"""

    @pytest.mark.asyncio
    async def test_throttle_respects_delay(self):
        api = HeadHunterAPIAsync(request_delay=0.1)
        start = time.time()
        await api._throttle()
        await api._throttle()
        elapsed = time.time() - start
        assert elapsed >= 0.095

    @pytest.mark.asyncio
    async def test_request_success(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url", payload={"data": "value"})
            async with aiohttp.ClientSession() as session:
                result = await api._request(session, "https://test.url")
                assert result == {"data": "value"}
                assert api.stats["success"] == 1

    @pytest.mark.asyncio
    async def test_request_handles_429_retry(self):
        api = HeadHunterAPIAsync(request_delay=0)
        with aioresponses() as m:
            m.get("https://test.url", status=429, headers={"Retry-After": "0"})
            m.get("https://test.url", payload={"ok": True})
            async with aiohttp.ClientSession() as session:
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await api._request(session, "https://test.url", retries=0, max_retries=2)
                    assert result == {"ok": True}
                    assert api.stats["429_errors"] == 1

    @pytest.mark.asyncio
    async def test_request_handles_403_as_missing(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url/vacancies/123", status=403)
            async with aiohttp.ClientSession() as session:
                result = await api._request(session, "https://test.url/vacancies/123")
                assert result is None
                assert api.stats["403_errors"] == 1

    @pytest.mark.asyncio
    async def test_request_handles_404(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url", status=404)
            async with aiohttp.ClientSession() as session:
                result = await api._request(session, "https://test.url")
                assert result is None
                assert api.stats["404_errors"] == 1

    @pytest.mark.asyncio
    async def test_request_timeout_retry(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url", exception=TimeoutError())
            m.get("https://test.url", payload={"ok": True})
            async with aiohttp.ClientSession() as session:
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await api._request(session, "https://test.url", max_retries=1)
                    assert result == {"ok": True}
                    assert api.stats["timeouts"] >= 1

    @pytest.mark.asyncio
    async def test_get_vacancy_details_async(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://api.hh.ru/vacancies/abc", payload={"id": "abc", "name": "Job"})
            async with aiohttp.ClientSession() as session:
                details = await api.get_vacancy_details_async(session, "abc")
                assert details["id"] == "abc"

    @pytest.mark.asyncio
    async def test_get_vacancies_details_batch_success(self):
        api = HeadHunterAPIAsync(max_concurrent=2, batch_size=2)
        vacancy_ids = ["1", "2", "3", "4"]
        with aioresponses() as m:
            for vid in vacancy_ids:
                m.get(f"https://api.hh.ru/vacancies/{vid}", payload={"id": vid, "details": f"data_{vid}"})
            results = await api.get_vacancies_details_batch(vacancy_ids)
            assert len(results) == 4

    @pytest.mark.asyncio
    async def test_get_vacancies_details_batch_filters_none(self):
        api = HeadHunterAPIAsync(batch_size=2)
        vacancy_ids = ["1", "bad", "2"]
        with aioresponses() as m:
            m.get("https://api.hh.ru/vacancies/1", payload={"id": "1"})
            m.get("https://api.hh.ru/vacancies/bad", status=404)
            m.get("https://api.hh.ru/vacancies/2", payload={"id": "2"})
            results = await api.get_vacancies_details_batch(vacancy_ids)
            assert len(results) == 2
            assert results[0]["id"] == "1"
            assert results[1]["id"] == "2"

    def test_get_vacancies_details_sync_wrapper(self):
        api = HeadHunterAPIAsync()
        vacancy_ids = ["1", "2"]
        with patch.object(api, "get_vacancies_details_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [{"id": "1"}, {"id": "2"}]
            results = api.get_vacancies_details_sync(vacancy_ids)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_throttle_skips_sleep_when_elapsed_greater(self):
        api = HeadHunterAPIAsync(request_delay=0.05)
        api.last_request_time = time.time() - 1.0
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await api._throttle()
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_request_handles_other_status(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url", status=500)
            async with aiohttp.ClientSession() as session:
                result = await api._request(session, "https://test.url")
                assert result is None
                assert api.stats["other_errors"] == 1

    @pytest.mark.asyncio
    async def test_request_handles_client_error(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url", exception=aiohttp.ClientError())
            async with aiohttp.ClientSession() as session:
                result = await api._request(session, "https://test.url")
                assert result is None
                assert api.stats["other_errors"] == 1

    @pytest.mark.asyncio
    async def test_get_vacancies_details_batch_empty_list(self):
        api = HeadHunterAPIAsync()
        results = await api.get_vacancies_details_batch([])
        assert results == []


class TestHeadHunterAPIMockedRequests:
    def test_search_vacancies_uses_industry_param(self):
        api = HeadHunterAPI()
        with patch.object(api, "_get") as mock_get:
            mock_get.return_value = {"items": [], "pages": 0}
            api.search_vacancies(text="DevOps", area=1, industry=7)
            call_params = mock_get.call_args[1]["params"]
            assert call_params["industry"] == 7
# Добавить в класс TestHeadHunterAPIAsync в tests/parsing/test_api.py

class TestHeadHunterAPIAsyncToken:
    """Тесты токенов (строки 41, 46, 49-50, 57-60)"""

    @pytest.mark.asyncio
    async def test_ensure_token_expired_gets_new(self):
        """Строки 41-50: токен истёк → получение нового"""
        api = HeadHunterAPIAsync(token="old", token_expires_at=time.time() - 100)

        with patch.object(api, "_get_app_token", new_callable=AsyncMock) as mock_get_token:
            await api._ensure_token()
            mock_get_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_token_no_credentials(self):
        """Строка 46: нет CLIENT_ID/SECRET"""
        api = HeadHunterAPIAsync()
        with patch("src.parsing.hh_api_async.config.HH_CLIENT_ID", None):
            await api._ensure_token()
            assert api._token is None

    @pytest.mark.asyncio
    async def test_ensure_token_from_sync_api(self, monkeypatch):
        """Строки 49-50: использование токена из синхронного API"""
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_ID", "test_id")
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_SECRET", "test_secret")

        api = HeadHunterAPIAsync()

        mock_sync = MagicMock()
        mock_sync._token = "sync_token"
        mock_sync._token_expires_at = time.time() + 3600

        # Патчим класс HeadHunterAPI в том месте, где он импортируется
        with patch("src.parsing.hh_api.HeadHunterAPI") as mock_hh_class:
            mock_hh_class.return_value = mock_sync
            await api._ensure_token()
            assert api._token == "sync_token"

    @pytest.mark.asyncio
    async def test_get_app_token_success(self, monkeypatch):
        """Строки 57-60: успешное получение токена"""
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_ID", "test_id")
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_SECRET", "test_secret")

        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.post("https://api.hh.ru/token", payload={
                "access_token": "new_token",
                "expires_in": 86400,
            })
            await api._get_app_token()
            assert api._token == "new_token"

    @pytest.mark.asyncio
    async def test_get_app_token_403_fallback(self, monkeypatch):
        """Строки 87-91: 403 при получении токена — без авторизации"""
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_ID", "test_id")
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_SECRET", "test_secret")

        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.post("https://api.hh.ru/token", status=403)
            await api._get_app_token()
            assert api._token is None

    @pytest.mark.asyncio
    async def test_get_app_token_exception(self, monkeypatch):
        """Строка 99: исключение при получении токена"""
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_ID", "test_id")
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_SECRET", "test_secret")

        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.post("https://api.hh.ru/token", exception=Exception("Network error"))
            await api._get_app_token()
            assert api._token is None


class TestHeadHunterAPIAsyncRequest:
    """Тесты запросов (строки 137-138, 142-148, 156-157, 178, 183-186)"""

    @pytest.mark.asyncio
    async def test_request_401_retry_with_token_refresh(self):
        """Строки 137-138: 401 → обновление токена и повтор"""
        api = HeadHunterAPIAsync(request_delay=0)
        with aioresponses() as m:
            m.get("https://test.url", status=401)
            m.get("https://test.url", payload={"ok": True})
            async with aiohttp.ClientSession() as session:
                with patch.object(api, "_ensure_token", new_callable=AsyncMock):
                    result = await api._request(session, "https://test.url", retries=0, max_retries=2)
                    assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_401_double_fail(self):
        """Строки 142-148: повторный 401 → None"""
        api = HeadHunterAPIAsync(request_delay=0)
        with aioresponses() as m:
            m.get("https://test.url", status=401)
            m.get("https://test.url", status=401)
            async with aiohttp.ClientSession() as session:
                with patch.object(api, "_ensure_token", new_callable=AsyncMock):
                    result = await api._request(session, "https://test.url", retries=0, max_retries=2)
                    assert result is None

    @pytest.mark.asyncio
    async def test_request_429_max_retries_exceeded(self):
        """Строки 156-157: превышены попытки 429"""
        api = HeadHunterAPIAsync(request_delay=0)
        with aioresponses() as m:
            m.get("https://test.url", status=429, headers={"Retry-After": "0"})
            m.get("https://test.url", status=429, headers={"Retry-After": "0"})
            async with aiohttp.ClientSession() as session:
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await api._request(session, "https://test.url", retries=1, max_retries=1)
                    assert result is None

    @pytest.mark.asyncio
    async def test_request_timeout_max_retries(self):
        """Строки 178, 183-186: таймаут с исчерпанием попыток"""
        api = HeadHunterAPIAsync(request_delay=0)
        with aioresponses() as m:
            m.get("https://test.url", exception=TimeoutError())
            m.get("https://test.url", exception=TimeoutError())
            async with aiohttp.ClientSession() as session:
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await api._request(session, "https://test.url", max_retries=1)
                    assert result is None

    def test_get_headers_with_token(self):
        """Строки 100-105: заголовки с токеном"""
        api = HeadHunterAPIAsync(token="my_token")
        headers = api._get_headers()
        assert headers["Authorization"] == "Bearer my_token"

    def test_get_headers_without_token(self):
        """Строки 100-105: заголовки без токена"""
        api = HeadHunterAPIAsync()
        headers = api._get_headers()
        assert "Authorization" not in headers


class TestHeadHunterAPIAsyncSyncWrapper:
    """Тесты синхронной обёртки (строки 250-253)"""

    def test_get_vacancies_details_sync_closed_loop(self):
        """Строки 250-253: закрытый event loop → создаётся новый"""
        api = HeadHunterAPIAsync()

        with patch.object(asyncio, "get_event_loop", side_effect=RuntimeError("No event loop")):
            with patch.object(api, "get_vacancies_details_batch", new_callable=AsyncMock) as mock_batch:
                mock_batch.return_value = [{"id": "1"}]
                results = api.get_vacancies_details_sync(["1"])
                assert len(results) == 1

    def test_get_vacancies_details_sync_existing_loop(self):
        """Строки 250-253: существующий event loop"""
        api = HeadHunterAPIAsync()

        # Создаём новый loop для теста
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            with patch.object(api, "get_vacancies_details_batch", new_callable=AsyncMock) as mock_batch:
                mock_batch.return_value = [{"id": "1"}, {"id": "2"}]
                results = api.get_vacancies_details_sync(["1", "2"])
                assert len(results) == 2
        finally:
            loop.close()


class TestHeadHunterAPIAsyncBatch:
    """Тесты пакетной загрузки (строка 223)"""

    @pytest.mark.asyncio
    async def test_get_vacancies_details_batch_empty(self):
        """Строка 223: пустой список"""
        api = HeadHunterAPIAsync()
        results = await api.get_vacancies_details_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_vacancies_details_batch_with_exceptions(self):
        """Строки 210-220: обработка исключений в gather"""
        api = HeadHunterAPIAsync(max_concurrent=2, batch_size=10)
        with aioresponses() as m:
            m.get("https://api.hh.ru/vacancies/1", payload={"id": "1"})
            m.get("https://api.hh.ru/vacancies/2", exception=Exception("Boom"))
            m.get("https://api.hh.ru/vacancies/3", payload={"id": "3"})

            results = await api.get_vacancies_details_batch(["1", "2", "3"])
            assert len(results) == 2
            ids = [r["id"] for r in results]
            assert "1" in ids
            assert "3" in ids

    def test_init_with_custom_batch_size(self):
        """Строки 25-30: кастомный batch_size"""
        api = HeadHunterAPIAsync(batch_size=25)
        assert api.batch_size == 25


    @pytest.mark.asyncio
    async def test_ensure_token_no_credentials(self):
        """Строка 46: нет CLIENT_ID/SECRET"""
        api = HeadHunterAPIAsync()
        with patch("src.parsing.hh_api_async.config.HH_CLIENT_ID", None):
            with patch("src.parsing.hh_api_async.config.HH_CLIENT_SECRET", None):
                await api._ensure_token()
                assert api._token is None

    @pytest.mark.asyncio
    async def test_get_app_token_success(self, monkeypatch):
        """Строки 57-60: успешное получение токена"""
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_ID", "test_id")
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_SECRET", "test_secret")

        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.post("https://api.hh.ru/token", payload={
                "access_token": "async_token",
                "expires_in": 86400,
            })
            await api._get_app_token()
            assert api._token == "async_token"

    @pytest.mark.asyncio
    async def test_get_app_token_other_error(self, monkeypatch):
        """Строка 91: другая ошибка при получении токена"""
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_ID", "test_id")
        monkeypatch.setattr("src.parsing.hh_api_async.config.HH_CLIENT_SECRET", "test_secret")

        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.post("https://api.hh.ru/token", status=500, body="Internal error")
            await api._get_app_token()
            assert api._token is None

    @pytest.mark.asyncio
    async def test_request_429_max_retries(self):
        """Строки 156-157: превышены попытки 429"""
        api = HeadHunterAPIAsync(request_delay=0)
        with aioresponses() as m:
            m.get("https://test.url", status=429, headers={"Retry-After": "0"})
            async with aiohttp.ClientSession() as session:
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await api._request(session, "https://test.url", retries=5, max_retries=5)
                    assert result is None

    @pytest.mark.asyncio
    async def test_get_vacancies_details_batch_empty(self):
        """Строка 223: пустой список ID"""
        api = HeadHunterAPIAsync()
        results = await api.get_vacancies_details_batch([])
        assert results == []

    def test_get_vacancies_details_sync_new_loop(self):
        """Строка 250: создание нового event loop"""
        api = HeadHunterAPIAsync()
        mock_batch = AsyncMock()
        mock_batch.return_value = [{"id": "1"}, {"id": "2"}]

        # Закрываем текущий loop
        try:
            loop = asyncio.get_event_loop()
            loop.close()
        except RuntimeError:
            pass

        with patch.object(api, "get_vacancies_details_batch", mock_batch):
            results = api.get_vacancies_details_sync(["1", "2"])
            assert len(results) == 2
