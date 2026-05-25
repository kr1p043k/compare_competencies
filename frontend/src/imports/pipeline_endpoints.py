"""
Эндпоинты FastAPI для запуска пайплайна с прогресс-баром.
Используют Server-Sent Events (SSE) для real-time обновлений.
"""

from pathlib import Path

from fastapi import Query
from fastapi.responses import StreamingResponse

from .pipeline_runner import PipelineRunner

# Инициализация runner (путь к корню проекта)
# ВАЖНО: Измените этот путь на актуальный путь к вашему проекту
BASE_DIR = Path(__file__).parent.parent.parent.parent
pipeline_runner = PipelineRunner(BASE_DIR)


def register_pipeline_endpoints(app):
    """Регистрирует эндпоинты пайплайна в FastAPI приложении."""

    @app.post("/api/pipeline/full-cycle")
    async def run_full_cycle_stream(
        regions: str = Query("0", description="ID регионов через запятую (0 = вся Россия)"),
    ):
        """
        Запускает полный цикл анализа через pipeline.bat
        Возвращает Server-Sent Events с прогрессом.

        События в формате:
        data: {"step": 1, "total": 4, "status": "running", "message": "...", "progress": 25}
        """
        return StreamingResponse(
            pipeline_runner.run_full_cycle(regions=regions),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Отключить буферизацию nginx
            },
        )


    @app.get("/api/pipeline/status")
    async def get_pipeline_status():
        """Проверяет доступность пайплайна."""
        # Проверяем наличие main.py
        main_py = BASE_DIR / "main.py"
        scripts_dir = BASE_DIR / "scripts"

        return {
            "available": main_py.exists() and scripts_dir.exists(),
            "base_dir": str(BASE_DIR),
            "main_py_exists": main_py.exists(),
            "scripts_dir_exists": scripts_dir.exists(),
        }
