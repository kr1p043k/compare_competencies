"""Метрики конверсии и времени выполнения pipeline."""

import time
from typing import Dict, Optional, Any
from prometheus_client import Counter, Histogram, Gauge

# Счётчики конверсии между шагами
step_conversion_counter = Counter(
    'pipeline_step_conversion_total',
    'Number of conversions between pipeline steps',
    ['from_step', 'to_step', 'status']
)

# Время выполнения каждого шага
step_duration_histogram = Histogram(
    'pipeline_step_duration_seconds',
    'Duration of each pipeline step',
    ['step'],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300]
)

# Текущий прогресс пайплайна
pipeline_progress = Gauge(
    'pipeline_progress_percent',
    'Current pipeline progress (0-100)',
    ['pipeline_id']
)

# Количество обработанных объектов на шаге
step_items_processed = Counter(
    'pipeline_step_items_processed_total',
    'Items processed per step',
    ['step', 'item_type']
)

# Успешность шага (для расчёта конверсии)
step_success_rate = Gauge(
    'pipeline_step_success_rate_percent',
    'Success rate per pipeline step',
    ['step']
)

# Общая успешность пайплайна
pipeline_success_rate = Gauge(
    'pipeline_success_rate_percent',
    'Overall pipeline success rate'
)

# Общее время выполнения пайплайна
pipeline_total_duration = Histogram(
    'pipeline_total_duration_seconds',
    'Total pipeline execution time',
    buckets=[60, 120, 180, 300, 600, 900, 1200, 1800, 3600]
)


class ConversionTracker:
    """Трекер конверсии и времени для pipeline."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.pipeline_id: Optional[str] = None
        self.step_times: Dict[str, float] = {}
        self.step_status: Dict[str, str] = {}
        self.step_items: Dict[str, int] = {}
        self.conversion_counts: Dict[str, int] = {}
        self.step_order = [
            "vacancy_fetch",
            "spam_filter", 
            "skill_parse",
            "weight_normalize",
            "level_assign",
            "cluster_train",
            "ltr_train",
            "gap_compute"
        ]
    
    def start_pipeline(self, pipeline_id: str) -> None:
        """Начать выполнение пайплайна."""
        self.pipeline_id = pipeline_id
        self.start_time = time.time()
        self.step_times = {}
        self.step_status = {}
        self.step_items = {}
        self.conversion_counts = {}
        pipeline_progress.labels(pipeline_id=pipeline_id).set(0)
    
    def record_step_start(self, step: str) -> None:
        """Записать начало выполнения шага."""
        self.step_times[step] = time.time()
        self.step_status[step] = "running"
    
    def record_step_end(self, step: str, status: str = "success", items_count: int = 0) -> None:
        """Записать окончание выполнения шага."""
        if step in self.step_times:
            duration = time.time() - self.step_times[step]
            step_duration_histogram.labels(step=step).observe(duration)
        
        self.step_status[step] = status
        
        if items_count > 0:
            step_items_processed.labels(step=step, item_type="items").inc(items_count)
            self.step_items[step] = items_count
        
        # Обновляем успешность шага (скользящее среднее)
        current_value = step_success_rate.labels(step=step)._value.get()
        current_rate = current_value if current_value is not None else 1.0
        new_rate = current_rate * 0.95 + (1.0 if status == "success" else 0.0) * 0.05
        step_success_rate.labels(step=step).set(new_rate)
    
    def record_conversion(self, from_step: str, to_step: str) -> None:
        """Запись перехода между шагами."""
        status = "success" if self.step_status.get(from_step) == "success" else "failed"
        step_conversion_counter.labels(
            from_step=from_step,
            to_step=to_step,
            status=status
        ).inc()
        
        key = f"{from_step}->{to_step}"
        self.conversion_counts[key] = self.conversion_counts.get(key, 0) + 1
    
    def end_pipeline(self, pipeline_id: str = None) -> Dict[str, Any]:
        """Завершить выполнение пайплайна."""
        pid = pipeline_id or self.pipeline_id
        if self.start_time:
            total_duration = time.time() - self.start_time
            pipeline_total_duration.observe(total_duration)
            
            # Общая успешность
            total_steps = len(self.step_status)
            if total_steps > 0:
                success_steps = sum(1 for s in self.step_status.values() if s == "success")
                success_rate = (success_steps / total_steps * 100)
                pipeline_success_rate.set(success_rate)
            else:
                success_rate = 0
            
            if pid:
                pipeline_progress.labels(pipeline_id=pid).set(100)
            
            result = {
                "pipeline_id": pid,
                "total_duration": total_duration,
                "success_rate": success_rate,
                "steps": self.step_status.copy(),
                "items_processed": self.step_items.copy(),
                "conversions": self.conversion_counts.copy()
            }
            return result
        
        return {}
    
    def get_conversion_funnel(self) -> Dict[str, Dict[str, float]]:
        """Получить воронку конверсии."""
        funnel = {}
        for step in self.step_order:
            success_rate_value = step_success_rate.labels(step=step)._value.get()
            funnel[step] = {
                "success_rate": success_rate_value if success_rate_value is not None else 100.0,
                "status": self.step_status.get(step, "unknown")
            }
        return funnel
    
    def get_step_duration_stats(self) -> Dict[str, Dict[str, float]]:
        """Получить статистику по времени шагов."""
        stats = {}
        for step in self.step_order:
            metric = step_duration_histogram.labels(step=step)
            count = metric._count.get()
            if count > 0:
                stats[step] = {
                    "avg_duration": metric._sum.get() / count,
                    "count": count
                }
            else:
                stats[step] = {"avg_duration": 0, "count": 0}
        return stats
    
    def reset(self) -> None:
        """Сбросить текущий трекер."""
        self.start_time = None
        self.pipeline_id = None
        self.step_times = {}
        self.step_status = {}
        self.step_items = {}
        self.conversion_counts = {}

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Получить latest pipeline metrics without mutating state."""
        if not self.start_time:
            return None
        return {
            "pipeline_id": self.pipeline_id,
            "total_duration": time.time() - self.start_time,
            "success_rate": pipeline_success_rate._value.get() or 0.0,
            "steps": self.step_status.copy(),
            "items_processed": self.step_items.copy(),
            "conversions": self.conversion_counts.copy(),
        }


pipeline_metrics = ConversionTracker()