"""Тут происходят темки с hard-тестом"""
from locust import HttpUser, task, between, events 
import json
import random

class CompetencyUser(HttpUser):
    wait_time = between(0.5, 2)
    host = "http://localhost:8000"
    
    def on_start(self):
        """Логин и подготовка данных"""
        self.profile_ids = ["base", "dc", "top_dc"]
        self.regions = [1, 2, 3, 4, 5, 66, 113, 114]
        
    @task(3)
    def get_vacancies(self):
        """Получение вакансий (самый частый запрос)"""
        params = {
            "query": random.choice(["python", "java", "data scientist", "devops"]),
            "area_id": random.choice(self.regions),
            "limit": 20
        }
        self.client.get("/api/vacancies", params=params, name="/api/vacancies")
    
    @task(2)
    def get_profiles(self):
        """Получение профилей студентов"""
        profile = random.choice(self.profile_ids)
        self.client.get(f"/api/profiles/{profile}", name="/api/profiles/{profile}")
    
    @task(2)
    def get_gap_analysis(self):
        """Gap-анализ (тяжёлый запрос)"""
        payload = {
            "student_profile": random.choice(self.profile_ids),
            "region_id": random.choice(self.regions),
            "top_n": 10
        }
        self.client.post("/api/gap-analysis", json=payload, name="/api/gap-analysis")
    
    @task(1)
    def get_trends(self):
        """Получение трендов"""
        self.client.get("/api/trends?days=30", name="/api/trends")
    
    @task(1)
    def get_clusters(self):
        """Получение кластеров вакансий"""
        self.client.get("/api/clusters", name="/api/clusters")
    
    @task(1)
    def get_market_metrics(self):
        """Рыночные метрики"""
        params = {"skill": random.choice(["python", "sql", "machine learning"])}
        self.client.get("/api/market/metrics", params=params, name="/api/market/metrics")
    
    @task(1)
    def trigger_pipeline(self):
        """Запуск пайплайна (редко)"""
        self.client.post("/api/pipeline/run", name="/api/pipeline/run", timeout=300)

class HeavyLoadUser(HttpUser):
    """Тяжёлая нагрузка — ML и аналитика"""
    wait_time = between(1, 5)
    host = "http://localhost:8000"
    
    @task
    def full_gap_analysis(self):
        """Полный gap-анализ со всеми опциями"""
        payload = {
            "student_profile": "dc",
            "region_id": 1,
            "levels": ["junior", "middle", "senior"],
            "include_shap": True,
            "include_recommendations": True
        }
        self.client.post("/api/gap-analysis/full", json=payload, timeout=60)
    
    @task
    def ltr_predict(self):
        """LTR-предсказание"""
        skills = ["python", "django", "postgresql", "docker"]
        self.client.post("/api/predict/skill-importance", 
                        json={"skills": skills}, timeout=30)

class PipelineUser(HttpUser):
    """Пользователи, запускающие пайплайн"""
    wait_time = between(60, 300)
    host = "http://localhost:8000"
    
    @task
    def run_nightly_pipeline(self):
        """Ежедневный пайплайн"""
        self.client.post("/api/pipeline/nightly", timeout=600)
    
    @task
    def check_progress(self):
        """Проверка прогресса"""
        self.client.get("/api/pipeline/progress")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Load test started")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test finished")