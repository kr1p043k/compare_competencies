import { useMemo, useState, useEffect } from "react";
import { motion } from "motion/react";
import { CheckCircle2, Loader2, XCircle, Rocket, Clock, Terminal } from "lucide-react";
import { Card, CardContent } from "./ui/card";
import { ScrollArea } from "./ui/scroll-area";

interface PipelineStep {
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
}

export function PipelineProgress({ currentStep, onCancel, onRestart }: PipelineProgressProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!currentStep || currentStep.status !== "running") { setElapsed(0); return; }
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, [currentStep?.step, currentStep?.status]);

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

  if (!currentStep) return null;

  const getStatusIcon = () => {
    switch (currentStep.status) {
      case "running":
        return <Loader2 className="size-6 text-blue-600 animate-spin" />;
      case "success":
      case "completed":
        return <CheckCircle2 className="size-6 text-green-600" />;
      case "error":
        return <XCircle className="size-6 text-red-600" />;
      default:
        return <Rocket className="size-6 text-blue-600" />;
    }
  };

  const getStatusColor = () => {
    switch (currentStep.status) {
      case "running":
        return "bg-blue-600";
      case "success":
      case "completed":
        return "bg-green-600";
      case "error":
        return "bg-red-600";
      default:
        return "bg-gray-400";
    }
  };

  const showSubBar = subPct > 0 && currentStep.status === "running";
  const logs = currentStep.logs ?? [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ type: "spring", stiffness: 200 }}
    >
      <Card className="border border-gray-200 shadow-sm">
        <CardContent className="p-6">
          {/* Header */}
          <div className="flex items-center gap-4 mb-6">
            <motion.div
              animate={{
                scale: currentStep.status === "running" ? [1, 1.1, 1] : 1,
              }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              {getStatusIcon()}
            </motion.div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-gray-900">
                {currentStep.status === "completed"
                  ? "Анализ завершен!"
                  : `Шаг ${currentStep.step} из ${currentStep.total}`}
              </h3>
              <p className="text-sm text-gray-600">{currentStep.message}</p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-blue-600">
                {currentStep.progress}%
              </div>
            </div>
          </div>

          {/* Progress Bars */}
          <div className="space-y-3">
            <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
              <motion.div
                className={`absolute inset-y-0 left-0 ${getStatusColor()} rounded-full`}
                initial={{ width: 0 }}
                animate={{ width: `${currentStep.progress}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              />
            </div>

            {showSubBar && (
              <div className="space-y-1">
                <div className="relative h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <motion.div
                    className="absolute inset-y-0 left-0 bg-blue-400 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${subPct}%` }}
                    transition={{ duration: 0.3, ease: "easeOut" }}
                  />
                </div>
                <p className="text-xs text-gray-400 text-right">{subPct}%</p>
              </div>
            )}

            {/* Steps Indicator */}
            <div className="flex justify-between items-center pt-2">
              {Array.from({ length: currentStep.total }, (_, i) => i + 1).map(
                (stepNum) => (
                  <div key={stepNum} className="flex items-center">
                    <motion.div
                      className={`size-8 rounded-full flex items-center justify-center text-sm font-semibold transition-all ${
                        stepNum < currentStep.step
                          ? "bg-green-600 text-white"
                          : stepNum === currentStep.step
                            ? "bg-blue-600 text-white"
                            : "bg-gray-200 text-gray-500"
                      }`}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: stepNum * 0.1 }}
                    >
                      {stepNum < currentStep.step ? (
                        <CheckCircle2 className="size-4" />
                      ) : (
                        stepNum
                      )}
                    </motion.div>
                    {stepNum < currentStep.total && (
                      <div
                        className={`w-12 h-0.5 ${
                          stepNum < currentStep.step
                            ? "bg-green-600"
                            : "bg-gray-200"
                        }`}
                      />
                    )}
                  </div>
                )
              )}
            </div>
          </div>

          {/* Elapsed time */}
          {currentStep.status === "running" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-4 flex items-center gap-2 text-sm text-slate-500"
            >
              <Clock className="size-4" />
              Прошло: {elapsedText}
              {estimatedText && <span className="text-slate-400">· Осталось: {estimatedText}</span>}
            </motion.div>
          )}

          {/* Actions */}
          <div className="mt-4 flex gap-2">
            {currentStep.status === "running" && onCancel && (
              <button
                onClick={onCancel}
                className="px-3 py-1.5 text-sm font-medium text-red-600 bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 transition-colors"
              >
                Остановить
              </button>
            )}
            {(currentStep.status === "completed" || currentStep.status === "error") && onRestart && (
              <button
                onClick={onRestart}
                className="px-3 py-1.5 text-sm font-medium text-blue-600 bg-blue-50 border border-blue-200 rounded-lg hover:bg-blue-100 transition-colors"
              >
                {currentStep.status === "error" ? "Повторить" : "Запустить ещё"}
              </button>
            )}
          </div>

          {/* Log Terminal */}
          {currentStep.status === "running" && logs.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mt-4"
            >
              <div className="flex items-center gap-2 mb-2 text-xs font-medium text-slate-500">
                <Terminal className="size-3.5" />
                Журнал пайплайна
              </div>
              <ScrollArea className="h-48 rounded-lg border border-gray-200 bg-gray-950 p-3">
                <div className="font-mono text-xs leading-relaxed">
                  {logs.map((line, i) => (
                    <div
                      key={i}
                      className={`${
                        line.includes("ошибка") || line.includes("Error") || line.includes("❌")
                          ? "text-red-400"
                          : line.includes("✅") || line.includes("успешно") || line.includes("завершён")
                            ? "text-green-400"
                            : line.includes("⚠") || line.includes("warning")
                              ? "text-yellow-400"
                              : "text-gray-300"
                      }`}
                    >
                      {line}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </motion.div>
          )}

          {/* Error Status */}
          {currentStep.status === "error" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg"
            >
              <p className="text-sm text-red-800">
                <strong>Ошибка:</strong> {currentStep.message}
              </p>
            </motion.div>
          )}

          {currentStep.status === "completed" && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg"
            >
              <p className="text-sm text-green-800 flex items-center gap-2">
                <CheckCircle2 className="size-5" />
                <strong>Успешно!</strong> Все данные обновлены.
              </p>
            </motion.div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
