"""PipelineOrchestrator — управление последовательностью этапов с retry, timing и прогрессом."""

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.pipeline.progress import write as write_progress
from src.pipeline.stage import PipelineStage
from src import Err, Ok, Result
from src.errors import PipelineError
from src.pipeline.event_bus import EventBus, PipelineEvent
logger = structlog.get_logger(__name__)


@dataclass
class StageResult:
    name: str
    status: str
    elapsed: float
    error: str | None = None
    data: Any = None


@dataclass
class PipelineRun:
    stages: list[StageResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    status: str = "pending"

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at


class PipelineOrchestrator:
    def __init__(
        self,
        stages: list[PipelineStage],
        num_retries: int = 1,
        event_bus: EventBus | None = None,
    ):
        self.stages = stages
        self.num_retries = num_retries
        self.event_bus = event_bus or EventBus()

    def run(
        self,
        initial_ctx: dict | None = None,
        name: str = "pipeline",
    ) -> Result[PipelineRun, PipelineError]:
        ctx = dict(initial_ctx or {})
        run = PipelineRun(started_at=time.time())
        total = len(self.stages)

        write_progress(0, f"Запуск {name}...")
        logger.info("orchestrator_started", stages=[s.name for s in self.stages], retries=self.num_retries)
        self.event_bus.emit(PipelineEvent(stage="__pipeline__", status="start", elapsed=0, metadata={"name": name}))

        for idx, stage in enumerate(self.stages):
            stage_name = stage.name or stage.__class__.__name__
            pct_base = int(idx / total * 100)
            write_progress(pct_base, f"Этап {idx + 1}/{total}: {stage_name}")

            last_error = None
            for attempt in range(self.num_retries + 1):
                if attempt > 0:
                    logger.warning("stage_retry", stage=stage_name, attempt=attempt)
                    write_progress(pct_base, f"Повтор {attempt}/{self.num_retries}: {stage_name}")

                t0 = time.time()
                try:
                    match stage.run(**ctx):
                        case Ok(data):
                            if isinstance(data, dict):
                                ctx.update(data)
                            else:
                                ctx[f"{stage_name}_result"] = data
                            elapsed = time.time() - t0
                            run.stages.append(StageResult(name=stage_name, status="ok", elapsed=elapsed, data=data))
                            logger.info("stage_ok", stage=stage_name, elapsed=round(elapsed, 2))
                            write_progress(pct_base + int(100 / total * 0.9), f"✓ {stage_name}")
                            self.event_bus.emit(
                                PipelineEvent(stage=stage_name, status="ok", elapsed=elapsed, metadata={"retry": attempt})
                            )
                            break
                        case Err(err):
                            last_error = str(err)
                            elapsed = time.time() - t0
                            logger.error("stage_failed", stage=stage_name, error=last_error, attempt=attempt)
                            self.event_bus.emit(
                                PipelineEvent(stage=stage_name, status="fail", elapsed=elapsed, metadata={"error": str(err), "retry": attempt})
                            )
                            if attempt < self.num_retries:
                                continue
                            run.stages.append(
                                StageResult(name=stage_name, status="failed", elapsed=elapsed, error=last_error)
                            )
                except Exception as e:
                    last_error = str(e)
                    elapsed = time.time() - t0
                    logger.exception("stage_exception", stage=stage_name, attempt=attempt)
                    self.event_bus.emit(
                        PipelineEvent(stage=stage_name, status="error", elapsed=elapsed, metadata={"error": str(e), "retry": attempt})
                    )
                    if attempt < self.num_retries:
                        continue
                    run.stages.append(
                        StageResult(name=stage_name, status="error", elapsed=elapsed, error=last_error)
                    )

            if run.stages and run.stages[-1].status != "ok":
                for s in reversed(self.stages[: idx + 1]):
                    s.rollback()
                run.status = "failed"
                write_progress(pct_base, f"✗ {stage_name}: {last_error}")
                logger.error("pipeline_failed", stage=stage_name, error=last_error)
                run.finished_at = time.time()
                self.event_bus.emit(
                    PipelineEvent(stage=stage_name, status="fail", elapsed=run.elapsed, metadata={"error": last_error})
                )
                return Err(PipelineError(message=f"Pipeline failed at stage {stage_name}", stage=stage_name, detail=last_error))

        run.status = "completed"
        run.finished_at = time.time()
        write_progress(100, "Пайплайн завершён")
        logger.info("pipeline_completed", elapsed=round(run.elapsed, 2))
        self.event_bus.emit(
            PipelineEvent(stage="__pipeline__", status="finish", elapsed=run.elapsed, metadata={"stages": len(run.stages)})
        )
        return Ok(run)
