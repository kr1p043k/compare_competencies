"""Метрики времени и конверсии для gap-анализа."""

import time
from typing import Dict, Optional, Any
from prometheus_client import Histogram, Counter, Gauge

# Время gap-анализа
gap_duration = Histogram(
    'gap_analysis_duration_seconds',
    'Duration of gap analysis',
    ['profile_type', 'region'],
    buckets=[0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60]
)

# Количество найденных гэпов
gaps_found = Histogram(
    'gap_analysis_gaps_found',
    'Number of gaps found per analysis',
    ['profile_type', 'severity'],
    buckets=[0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
)

# Успешность gap-анализа
gap_success_rate = Gauge(
    'gap_analysis_success_rate_percent',
    'Success rate of gap analysis',
    ['profile_type']
)

# Время SHAP объяснений (если используется)
shap_duration = Histogram(
    'gap_shap_duration_seconds',
    'Duration of SHAP explanation generation',
    ['profile_type'],
    buckets=[0.5, 1, 2, 3, 5, 10, 15, 20, 30]
)

# Количество рекомендаций
recommendations_count = Counter(
    'gap_recommendations_total',
    'Total recommendations generated',
    ['profile_type', 'priority']  # priority: critical, high, medium, low
)

# Время загрузки профиля студента
profile_load_duration = Histogram(
    'gap_profile_load_duration_seconds',
    'Duration of student profile loading',
    ['profile_type'],
    buckets=[0.1, 0.2, 0.5, 1, 2, 5]
)

# Кэш hit rate для gap-анализа
gap_cache_hit_rate = Gauge(
    'gap_cache_hit_rate_percent',
    'Cache hit rate for gap analysis',
    ['profile_type']
)


class GapMetricsTracker:
    """Трекер метрик для gap-анализа."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Сбросить текущие метрики."""
        self.start_time: Optional[float] = None
        self.shap_start_time: Optional[float] = None
        self.profile_type: str = ""
        self.region: int = 0
        self.cache_hit: bool = False
        self.gaps: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        self.recommendations_by_priority: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    
    def start(self, profile_type: str, region: int, cache_hit: bool = False) -> None:
        """Начать gap-анализ."""
        self.profile_type = profile_type
        self.region = region
        self.cache_hit = cache_hit
        self.start_time = time.time()
        self.gaps = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        self.recommendations_by_priority = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        # Обновляем cache hit rate
        current = gap_cache_hit_rate.labels(profile_type=profile_type)._value.get()
        if current is None:
            current = 0.5
        new_rate = current * 0.95 + (1.0 if cache_hit else 0.0) * 0.05
        gap_cache_hit_rate.labels(profile_type=profile_type).set(new_rate)
    
    def start_shap(self) -> None:
        """Начать вычисление SHAP объяснений."""
        self.shap_start_time = time.time()
    
    def end_shap(self) -> None:
        """Завершить вычисление SHAP объяснений."""
        if self.shap_start_time:
            duration = time.time() - self.shap_start_time
            shap_duration.labels(profile_type=self.profile_type).observe(duration)
            self.shap_start_time = None
    
    def add_gaps(self, gaps: Dict[str, int]) -> None:
        """Добавить найденные гэпы."""
        for severity, count in gaps.items():
            if severity in self.gaps:
                self.gaps[severity] += count
    
    def add_recommendations(self, recommendations: Dict[str, int]) -> None:
        """Добавить сгенерированные рекомендации."""
        for priority, count in recommendations.items():
            if priority in self.recommendations_by_priority:
                self.recommendations_by_priority[priority] += count
    
    def record_profile_load(self, duration: float) -> None:
        """Записать время загрузки профиля."""
        profile_load_duration.labels(profile_type=self.profile_type).observe(duration)
    
    def end(self, success: bool = True) -> Dict[str, Any]:
        """Завершить gap-анализ и записать метрики."""
        if self.start_time:
            duration = time.time() - self.start_time
            
            # Записываем общее время
            gap_duration.labels(
                profile_type=self.profile_type,
                region=str(self.region)
            ).observe(duration)
            
            # Записываем найденные гэпы
            for severity, count in self.gaps.items():
                if count > 0:
                    for _ in range(min(count, 100)):  # Ограничиваем количество наблюдений
                        gaps_found.labels(
                            profile_type=self.profile_type,
                            severity=severity
                        ).observe(count)
            
            # Записываем рекомендации
            for priority, count in self.recommendations_by_priority.items():
                if count > 0:
                    recommendations_count.labels(
                        profile_type=self.profile_type,
                        priority=priority
                    ).inc(count)
            
            # Обновляем успешность (скользящее среднее)
            current = gap_success_rate.labels(profile_type=self.profile_type)._value.get()
            current_rate = current if current is not None else 100.0
            new_rate = current_rate * 0.95 + (100.0 if success else 0.0) * 0.05
            gap_success_rate.labels(profile_type=self.profile_type).set(new_rate)
            
            result = {
                "profile_type": self.profile_type,
                "region": self.region,
                "duration": duration,
                "gaps_found": self.gaps.copy(),
                "recommendations": self.recommendations_by_priority.copy(),
                "cache_hit": self.cache_hit,
                "success": success
            }
            
            self.reset()
            return result
        
        return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Полуаем тут короче текущую статистику по gap-анализу."""
        stats = {}
        for profile in ["base", "dc", "top_dc"]:
            success_rate = gap_success_rate.labels(profile_type=profile)._value.get()
            cache_rate = gap_cache_hit_rate.labels(profile_type=profile)._value.get()
            stats[profile] = {
                "success_rate": success_rate if success_rate is not None else 100.0,
                "cache_hit_rate": cache_rate if cache_rate is not None else 0.0
            }
        return stats

gap_metrics = GapMetricsTracker()