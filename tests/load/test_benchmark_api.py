import pytest
import requests
import random
"""Тут короче bechmark hard-тест для ML, вроде как работает, грузит сильно долго правда - это нормально"""
BASE_URL = "http://localhost:8000"

pytest.importorskip("pytest_benchmark")


def _server_up():
    import socket
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", 8000))
        return True
    except OSError:
        return False
    finally:
        s.close()


pytestmark = pytest.mark.skipif(not _server_up(), reason="needs live backend on :8000")


def _admin_headers():
    """Teacher endpoints are role-gated; benchmark logs in once (dev DB seeds)."""
    r = requests.post(f"{BASE_URL}/api/auth/login", json={
        "email": "teacher@compare-competencies.local", "password": "teacher123"})
    r.raise_for_status()
    return {"Authorization": "Bearer " + r.json()["token"]}

class TestAPIBenchmark:
    
    def test_vacancies_endpoint(self, benchmark):
        """Бенчмарк для /api/vacancies"""
        def get_vacancies():
            response = requests.get(
                f"{BASE_URL}/api/vacancies",
                params={"query": "python", "limit": 50}
            )
            assert response.status_code == 200
            return response.json()
        
        result = benchmark(get_vacancies)
        assert len(result.get("items", [])) <= 50
    
    def test_gap_analysis_benchmark(self, benchmark):
        """Бенчмарк для gap-анализа"""
        headers = _admin_headers()

        def run_gap():
            response = requests.get(
                f"{BASE_URL}/api/teacher/krm/coverage",
                params={"direction": "09.03.02"},
                headers=headers,
            )
            assert response.status_code == 200
            return response.json()

        result = benchmark(run_gap)
        assert isinstance(result, (dict, list))
    
    def test_ltr_prediction_benchmark(self, benchmark):
        """Бенчмарк для LTR-предсказания"""
        skills = ["python", "sql", "docker", "kubernetes", "pandas"]
        
        def predict():
            response = requests.get(
                f"{BASE_URL}/api/forecast/top",
                params={"limit": 5}
            )
            assert response.status_code == 200
            return response.json()

        result = benchmark(predict)
        assert isinstance(result, (dict, list))
    
    @pytest.mark.parametrize("concurrent", [10, 50, 100])
    def test_concurrent_vacancies(self, concurrent):
        """Тест конкурентных запросов"""
        import concurrent.futures as _cf

        def fetch():
            return requests.get(
                f"{BASE_URL}/api/vacancies",
                params={"query": random.choice(["python", "java"]), "limit": 20}
            )

        with _cf.ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(fetch) for _ in range(concurrent)]
            results = [f.result() for f in futures]
        
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count / concurrent >= 0.95  # 95% успешных