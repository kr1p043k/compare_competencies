import pytest
import requests
import random
"""Тут короче bechmark hard-тест для ML, вроде как работает, грузит сильно долго правда - это нормально"""
BASE_URL = "http://localhost:8000"

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
        payload = {
            "student_profile": "dc",
            "region_id": 1,
            "top_n": 20
        }
        
        def run_gap():
            response = requests.post(
                f"{BASE_URL}/api/gap-analysis",
                json=payload
            )
            assert response.status_code == 200
            return response.json()
        
        result = benchmark(run_gap)
        assert "gaps" in result
    
    def test_ltr_prediction_benchmark(self, benchmark):
        """Бенчмарк для LTR-предсказания"""
        skills = ["python", "sql", "docker", "kubernetes", "pandas"]
        
        def predict():
            response = requests.post(
                f"{BASE_URL}/api/predict/skill-importance",
                json={"skills": skills}
            )
            assert response.status_code == 200
            return response.json()
        
        result = benchmark(predict)
        assert len(result.get("predictions", [])) == len(skills)
    
    @pytest.mark.parametrize("concurrent", [10, 50, 100])
    def test_concurrent_vacancies(self, concurrent):
        """Тест конкурентных запросов"""
        import concurrent.futures
        
        def fetch():
            return requests.get(
                f"{BASE_URL}/api/vacancies",
                params={"query": random.choice(["python", "java"]), "limit": 20}
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(fetch) for _ in range(concurrent)]
            results = [f.result() for f in futures]
        
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count / concurrent >= 0.95  # 95% успешных