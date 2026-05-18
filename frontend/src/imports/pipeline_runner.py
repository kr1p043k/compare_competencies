"""
Модуль для запуска пайплайна через pipeline.bat файл.
Отслеживает прогресс и отправляет события через SSE.
"""

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

import structlog

logger = structlog.get_logger("pipeline")


class PipelineRunner:
    """Запускает полный цикл анализа через pipeline.bat с отслеживанием прогресса."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    async def run_full_cycle(
        self,
        regions: str = "0",
    ) -> AsyncGenerator[str, None]:
        """
        Запускает полный цикл анализа через pipeline.bat

        Args:
            regions: ID регионов через запятую (0 = вся Россия)

        Yields:
            JSON строки с прогрессом в формате:
            {"step": 1, "total": 4, "status": "running", "message": "...", "progress": 25}
        """
        bat_file = self.base_dir / "pipeline.bat"

        if not bat_file.exists():
            yield self._format_progress(
                step=0,
                total=4,
                status="error",
                message=f"❌ Файл pipeline.bat не найден в {self.base_dir}",
            )
            logger.error("bat_file_not_found", path=str(bat_file))
            return

        # Определяем шаги для отслеживания
        steps = [
            "Сбор IT-вакансий с hh.ru",
            "Обучение кластеров вакансий",
            "Обучение ML-модели ранжирования",
            "GAP-анализ компетенций",
        ]

        total_steps = len(steps)
        current_step = 0

        logger.info("starting_pipeline", regions=regions, bat_file=str(bat_file))

        # Запускаем .bat файл с регионами как аргумент
        try:
            process = await asyncio.create_subprocess_exec(
                str(bat_file),
                regions,
                cwd=self.base_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            yield self._format_progress(
                step=0,
                total=total_steps,
                status="error",
                message=f"❌ Не удалось запустить pipeline.bat: {str(e)}",
            )
            logger.exception("failed_to_start_bat")
            return

        # Читаем вывод построчно для отслеживания прогресса
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            line_text = line.decode("utf-8", errors="ignore").strip()
            logger.debug("bat_output", line=line_text)

            # Отслеживаем шаги по выводу .bat файла
            if "[1/4]" in line_text and "Collecting" in line_text:
                current_step = 1
                yield self._format_progress(
                    step=current_step,
                    total=total_steps,
                    status="running",
                    message=f"Шаг {current_step}/{total_steps}: {steps[0]}",
                )
            elif "[1/4] ✓" in line_text:
                yield self._format_progress(
                    step=current_step,
                    total=total_steps,
                    status="success",
                    message=f"✓ Шаг {current_step} завершен: {steps[0]}",
                )

            elif "[2/4]" in line_text and "Training" in line_text and "cluster" in line_text:
                current_step = 2
                yield self._format_progress(
                    step=current_step,
                    total=total_steps,
                    status="running",
                    message=f"Шаг {current_step}/{total_steps}: {steps[1]}",
                )
            elif "[2/4] ✓" in line_text:
                yield self._format_progress(
                    step=current_step,
                    total=total_steps,
                    status="success",
                    message=f"✓ Шаг {current_step} завершен: {steps[1]}",
                )

            elif "[3/4]" in line_text and "Training" in line_text:
                current_step = 3
                yield self._format_progress(
                    step=current_step,
                    total=total_steps,
                    status="running",
                    message=f"Шаг {current_step}/{total_steps}: {steps[2]}",
                )
            elif "[3/4] ✓" in line_text:
                yield self._format_progress(
                    step=current_step,
                    total=total_steps,
                    status="success",
                    message=f"✓ Шаг {current_step} завершен: {steps[2]}",
                )

            elif "[4/4]" in line_text and "Running" in line_text:
                current_step = 4
                yield self._format_progress(
                    step=current_step,
                    total=total_steps,
                    status="running",
                    message=f"Шаг {current_step}/{total_steps}: {steps[3]}",
                )
            elif "[4/4] ✓" in line_text:
                yield self._format_progress(
                    step=current_step,
                    total=total_steps,
                    status="success",
                    message=f"✓ Шаг {current_step} завершен: {steps[3]}",
                )

        # Дождаться завершения процесса
        await process.wait()

        if process.returncode == 0:
            yield self._format_progress(
                step=total_steps,
                total=total_steps,
                status="completed",
                message="🎉 Полный цикл анализа завершен!",
            )
            logger.info("pipeline_completed_successfully")
        else:
            stderr = await process.stderr.read()
            error_msg = stderr.decode("utf-8", errors="ignore")[:500]
            yield self._format_progress(
                step=current_step if current_step > 0 else 1,
                total=total_steps,
                status="error",
                message=f"✗ Ошибка выполнения пайплайна",
            )
            logger.error("pipeline_failed", returncode=process.returncode, error=error_msg)

    def _format_progress(
        self, step: int, total: int, status: str, message: str
    ) -> str:
        """Форматирует прогресс в JSON для SSE."""
        progress = 0 if step == 0 else int((step / total) * 100)

        data = {
            "step": step,
            "total": total,
            "status": status,
            "message": message,
            "progress": progress,
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
