import { useEffect, useMemo, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import {
  Ban,
  BrainCircuit,
  ChartColumn,
  Check,
  CheckCircle2,
  CircleDashed,
  Clock,
  Database,
  Loader2,
  RotateCcw,
  Rocket,
  Search,
  Terminal,
  XCircle,
  type LucideIcon,
} from "lucide-react";

import { Badge, Button, Card, CardBody } from "../../design";
import { cn } from "./ui/utils";
import { ScrollArea } from "./ui/scroll-area";

export interface PipelineStep {
  step: number;
  total: number;
  status: "running" | "success" | "error" | "completed";
  message: string;
  progress: number;
  subProgress?: number;
  maxPages?: number;
  periodDays?: number;
  logs?: string[];
}

interface PipelineProgressProps {
  currentStep?: PipelineStep;
  onCancel?: () => void;
  onRestart?: () => void;
  showLogs?: boolean;
}

/**
 * Подписи шагов полного цикла. Соответствуют фазам бэкенда
 * (`_rotate_progress` в src/api_pkg/routers/pipeline.py): сбор -> подготовка ->
 * обучение -> GAP-анализ. Для номеров вне карты используется нейтральная метка.
 */
const STEP_META: Record<number, { title: string; hint: string; icon: LucideIcon }> = {
  1: {
    title: "Сбор вакансий",
    hint: "Поиск на hh.ru, оценка качества, извлечение навыков",
    icon: Search,
  },
  2: {
    title: "Подготовка данных",
    hint: "Очистка весов навыков, построение уровней",
    icon: Database,
  },
  3: {
    title: "Обучение моделей",
    hint: "Кластеры вакансий и LTR-модель ранжирования",
    icon: BrainCircuit,
  },
  4: {
    title: "GAP-анализ и рекомендации",
    hint: "Оценка профилей, генерация рекомендаций, графики",
    icon: ChartColumn,
  },
};

type StepState = "done" | "active" | "failed" | "pending";

/** Стиль кружка-иконки шага по состоянию (все цвета - дизайн-токены). */
const CIRCLE_CLASS: Record<StepState, string> = {
  done: "border-success/30 bg-success-soft text-success",
  active: "border-primary bg-primary text-primary-foreground shadow-focus",
  failed: "border-destructive/30 bg-destructive/10 text-destructive",
  pending: "border-border bg-muted text-muted-foreground",
};

const TITLE_CLASS: Record<StepState, string> = {
  done: "text-sm font-medium text-foreground",
  active: "text-sm font-semibold text-foreground",
  failed: "text-sm font-medium text-foreground",
  pending: "text-sm font-medium text-muted-foreground",
};

/** Цвет строки журнала по содержимому: краски - дизайн-токены обеих тем. */
function logLineClass(line: string): string {
  if (line.includes("ошибка") || line.includes("Error") || line.includes("❌")) {
    return "text-destructive";
  }
  if (line.includes("✅") || line.includes("успешно") || line.includes("завершён")) {
    return "text-success";
  }
  if (line.includes("⚠") || line.includes("warning")) {
    return "text-warning";
  }
  return "text-muted-foreground";
}

export function PipelineProgress({ currentStep, onCancel, onRestart, showLogs = true }: PipelineProgressProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!currentStep || currentStep.status !== "running") { setElapsed(0); return; }
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, [currentStep?.step, currentStep?.status]);

  const reduceMotion = useReducedMotion();

  const elapsedText = useMemo(() => {
    if (elapsed < 60) return `${elapsed} сек`;
    const m = Math.floor(elapsed / 60);
    const s = elapsed % 60;
    return `${m} мин ${s} сек`;
  }, [elapsed]);

  const progressPct = currentStep?.progress ?? 0;
  const subPct = currentStep?.subProgress ?? 0;
  const effectivePct = subPct > 0 ? subPct : progressPct;
  const estimatedText = useMemo(() => {
    if (effectivePct <= 0 || elapsed < 10) return null;
    const rate = Math.max(effectivePct, 1) / elapsed;
    const remaining = Math.max(0, (100 - effectivePct) / rate);
    if (remaining < 15) return null;
    if (remaining > 7200) return null;
    if (remaining < 60) return `~${Math.round(remaining)} сек`;
    return `~${Math.floor(remaining / 60)} мин ${Math.round(remaining % 60)} сек`;
  }, [elapsed, effectivePct]);

  // Журнал - чистый вывод из currentStep.logs: тик elapsed (1с) не
  // пересоздаёт строки терминала, они пересобираются только при новых логах.
  const logLines = useMemo(
    () =>
      (currentStep?.logs ?? []).map((line, i) => (
        <div key={i} className={logLineClass(line)}>
          {line}
        </div>
      )),
    [currentStep?.logs],
  );

  if (!currentStep) return null;

  const running = currentStep.status === "running";
  const finished = currentStep.status === "completed" || currentStep.status === "success";
  const failed = currentStep.status === "error";
  const cancelled = failed && currentStep.message.trim().toLowerCase() === "отменено";
  // Шаг 0: пайплайн упал до запуска первого шага - списки и бары не показываем.
  const startFailed = failed && currentStep.step < 1;

  const total = Math.max(1, currentStep.total);
  const displayStep = Math.min(Math.max(currentStep.step, 1), total);
  const pct = Math.min(100, Math.max(0, progressPct));

  const headerTitle = finished
    ? "Анализ завершён"
    : cancelled
      ? "Пайплайн остановлен"
      : startFailed
        ? "Ошибка запуска"
        : failed
          ? "Ошибка выполнения"
          : "Обновление данных";
  const headerDesc = currentStep.message;

  const pctTone = finished ? "success" : cancelled ? "warning" : failed ? "danger" : "primary";
  const barTone = finished ? "bg-success" : cancelled ? "bg-warning" : failed ? "bg-destructive" : "bg-primary";

  const steps = Array.from({ length: currentStep.total }, (_, i) => i + 1);
  const stepState = (n: number): StepState => {
    if (finished) return "done";
    if (n < displayStep) return "done";
    if (n === displayStep) return running ? "active" : failed ? "failed" : "pending";
    return "pending";
  };

  const badgeMeta = (st: StepState): { tone: "neutral" | "primary" | "success" | "warning" | "danger"; icon: LucideIcon; label: string; spin?: boolean } => {
    switch (st) {
      case "done":
        return { tone: "success", icon: Check, label: "Выполнено" };
      case "active":
        return { tone: "primary", icon: Loader2, label: "Выполняется", spin: true };
      case "failed":
        return cancelled
          ? { tone: "warning", icon: Ban, label: "Остановлено" }
          : { tone: "danger", icon: XCircle, label: "Ошибка" };
      default:
        return { tone: "neutral", icon: CircleDashed, label: "Ожидание" };
    }
  };

  const showSubBar = subPct > 0 && running;
  const logs = currentStep.logs ?? [];
  const showStepList = !startFailed && currentStep.total >= 1;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ type: "spring", stiffness: 200 }}
    >
      <Card>
        <CardBody className="p-5">
          {/* Header: статус, заголовок, живое сообщение, бейджи «шаг» и «процент» */}
          <div className="flex items-start gap-4">
            <motion.div
              animate={{ scale: running && !reduceMotion ? [1, 1.08, 1] : 1 }}
              transition={{ duration: 1.6, repeat: Infinity }}
              className={cn(
                "flex size-10 shrink-0 items-center justify-center rounded-full",
                running && "bg-primary-soft text-primary",
                finished && "bg-success-soft text-success",
                failed && !cancelled && "bg-destructive/10 text-destructive",
                cancelled && "bg-warning-soft text-warning",
              )}
            >
              {running ? (
                <Loader2 className="size-5 animate-spin" aria-hidden="true" />
              ) : finished ? (
                <CheckCircle2 className="size-5" aria-hidden="true" />
              ) : cancelled ? (
                <Ban className="size-5" aria-hidden="true" />
              ) : (
                <XCircle className="size-5" aria-hidden="true" />
              )}
            </motion.div>
            <div className="min-w-0 flex-1">
              <h3 className="text-base font-semibold text-foreground">{headerTitle}</h3>
              <p aria-live="polite" className="mt-0.5 text-sm text-muted-foreground">
                {headerDesc}
              </p>
            </div>
            <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
              <Badge tone={pctTone} className="tabular-nums">{pct}%</Badge>
              {!startFailed && (
                <Badge tone="neutral" className="tabular-nums">
                  Шаг {displayStep} из {total}
                </Badge>
              )}
            </div>
          </div>

          {/* Прогресс-бары (общий + текущего шага) */}
          {!startFailed && (
            <div className="mt-5 space-y-2">
              <div
                role="progressbar"
                aria-label="Общий прогресс пайплайна"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={pct}
                aria-valuetext={`${pct}%`}
                className="relative h-2 w-full overflow-hidden rounded-full bg-muted"
              >
                <motion.div
                  className={cn("absolute inset-y-0 left-0 rounded-full", barTone)}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                />
              </div>

              {showSubBar && (
                <div className="space-y-1">
                  <div
                    role="progressbar"
                    aria-label="Прогресс текущего шага"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={subPct}
                    aria-valuetext={`${subPct}%`}
                    className="relative h-1.5 w-full overflow-hidden rounded-full bg-muted"
                  >
                    <motion.div
                      className="absolute inset-y-0 left-0 rounded-full bg-info"
                      initial={{ width: 0 }}
                      animate={{ width: `${subPct}%` }}
                      transition={{ duration: 0.3, ease: "easeOut" }}
                    />
                  </div>
                  <p className="text-right text-xs text-muted-foreground tabular-nums">{subPct}%</p>
                </div>
              )}
            </div>
          )}

          {/* Степпер: список шагов с иконками, статусами и коннекторами */}
          {showStepList && (
            <ol aria-label="Шаги пайплайна" className="mt-5 flex flex-col">
              {steps.map((n) => {
                const st = stepState(n);
                const meta = STEP_META[n] ?? { title: `Шаг ${n}`, hint: "", icon: Rocket };
                const StepIcon = meta.icon;
                const badge = badgeMeta(st);
                const BadgeIcon = badge.icon;
                const isLast = n === steps.length;
                const connectorDone = finished || n < displayStep;
                return (
                  <li key={n} aria-current={st === "active" ? "step" : undefined} className="flex gap-3">
                    <div className="flex flex-col items-center self-stretch">
                      <span
                        className={cn(
                          "flex size-9 shrink-0 items-center justify-center rounded-full border transition-colors",
                          CIRCLE_CLASS[st],
                        )}
                      >
                        <StepIcon className="size-4" aria-hidden="true" />
                      </span>
                      {!isLast && (
                        <span
                          className={cn("min-h-4 w-px flex-1", connectorDone ? "bg-success/50" : "bg-border")}
                          aria-hidden="true"
                        />
                      )}
                    </div>
                    <div
                      className={cn(
                        "mb-4 flex flex-1 items-center justify-between gap-3 rounded-lg px-3 py-2",
                        isLast && "mb-0",
                        st === "active" && "bg-primary-soft",
                        st === "failed" && !cancelled && "bg-destructive/5",
                        cancelled && st === "failed" && "bg-warning-soft/50",
                      )}
                    >
                      <div className="min-w-0">
                        <p className={TITLE_CLASS[st]}>{meta.title}</p>
                        {meta.hint && <p className="mt-0.5 text-xs text-muted-foreground">{meta.hint}</p>}
                      </div>
                      <Badge tone={badge.tone} className="shrink-0">
                        <BadgeIcon className={cn("size-3", badge.spin && "animate-spin")} aria-hidden="true" />
                        {badge.label}
                      </Badge>
                    </div>
                  </li>
                );
              })}
            </ol>
          )}

          {/* Elapsed time */}
          {running && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-4 flex items-center gap-2 text-sm text-muted-foreground"
            >
              <Clock className="size-4" aria-hidden="true" />
              Прошло: {elapsedText}
              {estimatedText && <span className="text-muted-foreground/70">· Осталось: {estimatedText}</span>}
            </motion.div>
          )}

          {/* Actions */}
          <div className="mt-4 flex gap-2">
            {running && onCancel && (
              <Button
                variant="secondary"
                size="sm"
                onClick={onCancel}
                iconLeft={<XCircle className="size-3.5" aria-hidden="true" />}
                className="text-destructive border-destructive/30 hover:bg-destructive/10 hover:text-destructive"
              >
                Остановить
              </Button>
            )}
            {failed && onRestart && (
              <Button
                variant="primary"
                size="sm"
                onClick={onRestart}
                iconLeft={<RotateCcw className="size-3.5" aria-hidden="true" />}
              >
                Повторить
              </Button>
            )}
            {finished && onRestart && (
              <Button
                variant="secondary"
                size="sm"
                onClick={onRestart}
                iconLeft={<RotateCcw className="size-3.5" aria-hidden="true" />}
              >
                Запустить ещё
              </Button>
            )}
          </div>

          {/* Log Terminal (admin only) */}
          {showLogs && running && logs.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mt-4"
            >
              <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <Terminal className="size-3.5" aria-hidden="true" />
                Журнал пайплайна
              </div>
              <ScrollArea className="h-48 rounded-lg border border-border bg-muted/50 p-3">
                <div className="font-mono text-xs leading-relaxed">{logLines}</div>
              </ScrollArea>
            </motion.div>
          )}

          {/* Error banner */}
          {failed && !cancelled && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              role="alert"
              className="mt-4 rounded-lg border border-destructive/30 bg-destructive/10 p-4"
            >
              <p className="text-sm text-destructive">
                <strong>Ошибка:</strong> {currentStep.message}
              </p>
            </motion.div>
          )}

          {/* Cancelled banner */}
          {cancelled && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              role="status"
              className="mt-4 rounded-lg border border-warning/30 bg-warning-soft p-4"
            >
              <p className="text-sm text-warning">Выполнение остановлено вручную. Запустите анализ заново.</p>
            </motion.div>
          )}

          {/* Completed banner */}
          {finished && (
            <motion.div
              initial={{ opacity: 0, scale: reduceMotion ? 1 : 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              role="status"
              className="mt-4 rounded-lg border border-success/30 bg-success-soft p-4"
            >
              <p className="flex items-center gap-2 text-sm text-success">
                <CheckCircle2 className="size-5" aria-hidden="true" />
                <strong>Успешно!</strong> Все данные обновлены.
              </p>
            </motion.div>
          )}
        </CardBody>
      </Card>
    </motion.div>
  );
}
