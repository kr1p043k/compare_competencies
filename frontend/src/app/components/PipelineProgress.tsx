import { motion } from "motion/react";
import { CheckCircle2, Loader2, XCircle, Rocket } from "lucide-react";
import { Card, CardContent } from "./ui/card";

interface PipelineStep {
  step: number;
  total: number;
  status: "running" | "success" | "error" | "completed";
  message: string;
  progress: number;
}

interface PipelineProgressProps {
  currentStep?: PipelineStep;
}

export function PipelineProgress({ currentStep }: PipelineProgressProps) {
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

          {/* Progress Bar */}
          <div className="space-y-3">
            <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
              <motion.div
                className={`absolute inset-y-0 left-0 ${getStatusColor()} rounded-full`}
                initial={{ width: 0 }}
                animate={{ width: `${currentStep.progress}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              />
            </div>

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

          {/* Status Message */}
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
                <strong>Успешно!</strong> Все данные обновлены. Обновите
                страницу для просмотра результатов.
              </p>
            </motion.div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
