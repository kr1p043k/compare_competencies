import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator

import structlog

logger = structlog.get_logger("pipeline")

PYTHON = sys.executable

class PipelineRunner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    async def run_full_cycle(
        self,
        regions: str = "0",
    ) -> AsyncGenerator[str, None]:
        steps = [
            ("Сбор IT-вакансий с hh.ru", ["main.py", "--it-sector", "--regions", regions, "--excel"]),
            ("Обучение кластеров вакансий", ["scripts/train_clusters.py", "--level", "all"]),
            ("Обучение ML-модели ранжирования", ["main.py", "--train-model"]),
            ("GAP-анализ компетенций", ["main.py", "--skip-collection", "--run-gap-analysis"]),
        ]

        total_steps = len(steps)

        for i, (label, cmd) in enumerate(steps, 1):
            yield self._fmt(i, total_steps, "running", f"Шаг {i}/{total_steps}: {label}")
            try:
                proc = await asyncio.create_subprocess_exec(
                    PYTHON, *cmd,
                    cwd=self.base_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    err = stderr.decode("utf-8", errors="ignore")[:500]
                    yield self._fmt(i, total_steps, "error", f"✗ Шаг {i} провален: {err}")
                    logger.error("step_failed", step=i, label=label, error=err)
                    return
                yield self._fmt(i, total_steps, "success", f"✓ Шаг {i} завершен: {label}")
            except Exception as e:
                yield self._fmt(i, total_steps, "error", f"✗ Ошибка запуска шага {i}: {e}")
                logger.exception("step_crash", step=i, label=label)
                return

        yield self._fmt(total_steps, total_steps, "completed", "🎉 Полный цикл анализа завершен!")

    def _fmt(self, step: int, total: int, status: str, message: str) -> str:
        progress = int((step / total) * 100)
        data = {"step": step, "total": total, "status": status, "message": message, "progress": progress}
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
