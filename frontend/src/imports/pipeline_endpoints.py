"""
FastAPI endpoints for the analysis pipeline with SSE progress.
"""

from pathlib import Path

from fastapi import Query
from fastapi.responses import StreamingResponse

from .pipeline_runner import PipelineRunner

BASE_DIR = Path(__file__).parent.parent.parent.parent
pipeline_runner = PipelineRunner(BASE_DIR)


def register_pipeline_endpoints(app):
    @app.post("/api/pipeline/full-cycle")
    async def run_full_cycle_stream(
        regions: str = Query("0", description="ID регионов через запятую (0 = вся Россия)"),
    ):
        return StreamingResponse(
            pipeline_runner.run_full_cycle(regions=regions),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
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
