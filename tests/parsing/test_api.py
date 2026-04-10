import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from src.parsing.hh_api import HeadHunterAPI
from src.parsing.hh_api_async import HeadHunterAPIAsync
from src.models.vacancy import Vacancy
import requests
import aiohttp
from aioresponses import aioresponses
class TestHeadHunterAPISync:
    """Тесты синхронного клиента HeadHunterAPI"""

class TestHeadHunterAPISync:
    def test_init_creates_session_with_retries(self):
        api = HeadHunterAPI()
        assert api.session is not None
        assert 'User-Agent' in api.session.headers
        api.close()

    @patch('src.parsing.hh_api.HeadHunterAPI._get')
    def test_search_vacancies_success(self, mock_get):
        api = HeadHunterAPI()
        mock_get.return_value = {
            "items": [{"id": "1"}, {"id": "2"}],
            "pages": 1,
            "found": 2
        }
        result = api.search_vacancies(text="Python", area=1)
        assert len(result) == 2
        mock_get.assert_called_once()

    @patch('src.parsing.hh_api.HeadHunterAPI._get')
    def test_search_vacancies_pagination(self, mock_get):
        api = HeadHunterAPI()
        mock_get.side_effect = [
            {"items": [{"id": "1"}, {"id": "2"}], "pages": 2, "found": 3},
            {"items": [{"id": "3"}], "pages": 2, "found": 3},
        ]
        result = api.search_vacancies(text="Java", area=2, max_pages=5)
        assert len(result) == 3
        assert mock_get.call_count == 2

    @patch('src.parsing.hh_api.HeadHunterAPI._get')
    def test_search_vacancies_stops_on_empty_page(self, mock_get):
        api = HeadHunterAPI()
        mock_get.return_value = {"items": [], "pages": 5}
        result = api.search_vacancies(text="C++", area=1)
        assert result == []

    @patch('src.parsing.hh_api.HeadHunterAPI._get')
    def test_get_vacancy_details_success(self, mock_get):
        api = HeadHunterAPI()
        mock_get.return_value = {"id": "123", "name": "Test"}
        details = api.get_vacancy_details("123")
        assert details["id"] == "123"

    def test_get_vacancy_details_returns_none_on_error(self):
        api = HeadHunterAPI()
        with patch.object(api, '_get', return_value=None):
            details = api.get_vacancy_details("999")
            assert details is None

    def test_get_vacancy_details_as_object_valid(self):
        api = HeadHunterAPI()
        raw = {
            "id": "456",
            "name": "Python Dev",
            "area": {"id": 1, "name": "Москва"},
            "employer": {"id": "10", "name": "Company"},
            "key_skills": [{"name": "Python"}, {"name": "Django"}]
        }
        with patch.object(api, '_get', return_value=raw):
            vacancy = api.get_vacancy_details_as_object("456")
            assert isinstance(vacancy, Vacancy)
            assert vacancy.id == "456"
            assert len(vacancy.key_skills) == 2

    def test_get_vacancy_details_as_object_invalid(self):
        api = HeadHunterAPI()
        with patch.object(api, '_get', return_value={"id": "no_name"}):
            vacancy = api.get_vacancy_details_as_object("bad")
            assert vacancy is None

    def test_get_handles_429_retry(self):
        api = HeadHunterAPI()
        with patch.object(api.session, 'get') as mock_session_get:
            rate_limit_response = Mock(status_code=429, headers={"Retry-After": "1"})
            success_response = Mock(status_code=200)
            success_response.json.return_value = {"result": "ok"}
            mock_session_get.side_effect = [rate_limit_response, success_response]

            result = api._get("https://test.url")
            assert result == {"result": "ok"}
            assert mock_session_get.call_count == 2

    def test_get_handles_403_forbidden(self):
        api = HeadHunterAPI()
        with patch.object(api.session, 'get') as mock_session_get:
            mock_session_get.return_value = Mock(status_code=403)
            result = api._get("https://blocked.url")
            assert result is None

    def test_get_handles_timeout(self):
        api = HeadHunterAPI()
        with patch.object(api.session, 'get', side_effect=requests.exceptions.Timeout):
            result = api._get("https://timeout.url")
            assert result is None

    def test_context_manager_closes_session(self):
        with HeadHunterAPI() as api:
            assert api.session is not None
        # после выхода сессия должна быть закрыта
        # проверить сложно, но можно убедиться, что метод close вызван
        with patch.object(HeadHunterAPI, 'close') as mock_close:
            with HeadHunterAPI():
                pass
            mock_close.assert_called_once()

    def test_search_vacancies_with_max_pages_one(self):
        api = HeadHunterAPI()
        with patch.object(api, '_get') as mock_get:
            mock_get.return_value = {"items": [{"id": "1"}], "pages": 1}
            result = api.search_vacancies(text="Python", area=1, max_pages=1)
            assert len(result) == 1
            mock_get.assert_called_once()

    def test_get_vacancy_details_as_object_success_existing(self):
        api = HeadHunterAPI()
        raw = {"id": "123", "name": "Test", "area": {"id":1,"name":"MSK"}, "employer":{"id":"10","name":"Corp"}}
        with patch.object(api, '_get', return_value=raw):
            vac = api.get_vacancy_details_as_object("123")
            assert vac.id == "123"

    def test_get_handles_304_not_modified(self):
        api = HeadHunterAPI()
        with patch.object(api.session, 'get') as mock_get:
            mock_get.return_value = Mock(status_code=304)
            result = api._get("https://test.url")
            assert result is None

    def test_get_handles_unexpected_status(self):
        api = HeadHunterAPI()
        with patch.object(api.session, 'get') as mock_get:
            mock_get.return_value = Mock(status_code=418)
            result = api._get("https://test.url")
            assert result is None

    def test_get_handles_general_exception(self):
        api = HeadHunterAPI()
        with patch.object(api.session, 'get', side_effect=Exception("Boom")):
            result = api._get("https://test.url")
            assert result is None
class TestHeadHunterAPIAsync:
    @pytest.mark.asyncio
    async def test_throttle_respects_delay(self):
        api = HeadHunterAPIAsync(request_delay=0.1)
        import time
        start = time.time()
        await api._throttle()
        await api._throttle()
        elapsed = time.time() - start
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_request_success(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url", payload={"data": "value"})
            async with aiohttp.ClientSession() as session:
                result = await api._request(session, "https://test.url")
                assert result == {"data": "value"}
                assert api.stats['success'] == 1

    @pytest.mark.asyncio
    async def test_request_handles_429_retry(self):
        api = HeadHunterAPIAsync(request_delay=0)
        with aioresponses() as m:
            m.get("https://test.url", status=429, headers={"Retry-After": "0"})
            m.get("https://test.url", payload={"ok": True})
            async with aiohttp.ClientSession() as session:
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    result = await api._request(session, "https://test.url", retries=0, max_retries=2)
                    assert result == {"ok": True}
                    assert api.stats['429_errors'] == 1

    @pytest.mark.asyncio
    async def test_request_handles_403_as_missing(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url/vacancies/123", status=403)
            async with aiohttp.ClientSession() as session:
                result = await api._request(session, "https://test.url/vacancies/123")
                assert result is None
                assert api.stats['403_errors'] == 1

    @pytest.mark.asyncio
    async def test_request_handles_404(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url", status=404)
            async with aiohttp.ClientSession() as session:
                result = await api._request(session, "https://test.url")
                assert result is None
                assert api.stats['404_errors'] == 1

    @pytest.mark.asyncio
    async def test_request_timeout_retry(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url", exception=asyncio.TimeoutError())
            m.get("https://test.url", payload={"ok": True})   # second try success
            async with aiohttp.ClientSession() as session:
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    result = await api._request(session, "https://test.url", max_retries=1)
                    assert result == {"ok": True}
                    # stats могут быть обновлены дважды: первый timeout, потом success
                    assert api.stats['timeouts'] >= 1

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
        with patch.object(api, 'get_vacancies_details_batch', new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [{"id": "1"}, {"id": "2"}]
            results = api.get_vacancies_details_sync(vacancy_ids)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_stats_accumulation(self):
        api = HeadHunterAPIAsync()
        api.stats['success'] = 5
        api.stats['403_errors'] = 2
        api.stats = {k: 0 for k in api.stats}
        assert api.stats['success'] == 0
        
    @pytest.mark.asyncio
    async def test_throttle_skips_sleep_when_elapsed_greater(self):
        api = HeadHunterAPIAsync(request_delay=0.05)
        # Устанавливаем last_request_time в прошлом, чтобы не спать
        api.last_request_time = time.time() - 1.0
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
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
                assert api.stats['other_errors'] == 1

    @pytest.mark.asyncio
    async def test_request_handles_client_error(self):
        api = HeadHunterAPIAsync()
        with aioresponses() as m:
            m.get("https://test.url", exception=aiohttp.ClientError())
            async with aiohttp.ClientSession() as session:
                result = await api._request(session, "https://test.url")
                assert result is None
                assert api.stats['other_errors'] == 1

    @pytest.mark.asyncio
    async def test_get_vacancies_details_batch_empty_list(self):
        api = HeadHunterAPIAsync()
        results = await api.get_vacancies_details_batch([])
        assert results == []

    def test_get_vacancies_details_sync_closed_loop(self):
        api = HeadHunterAPIAsync()
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.close()
        with patch.object(api, 'get_vacancies_details_batch', new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = [{"id": "1"}]
            # После закрытия loop метод должен создать новый
            results = api.get_vacancies_details_sync(["1"])
            assert results == [{"id": "1"}]
    
        
# Дополнительно: тест для HeadHunterAPI с реальным (замоканным) requests.get
class TestHeadHunterAPIMockedRequests:
    def test_search_vacancies_uses_industry_param(self):
        api = HeadHunterAPI()
        with patch.object(api, '_get') as mock_get:
            mock_get.return_value = {"items": [], "pages": 0}
            api.search_vacancies(text="DevOps", area=1, industry=7)
            call_params = mock_get.call_args[1]['params']
            assert call_params['industry'] == 7