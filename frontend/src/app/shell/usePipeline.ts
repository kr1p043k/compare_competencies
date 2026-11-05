import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, request } from "../../lib/api";
import { reportError } from "../../lib/errors";

/** Активная задача пайплайна из /api/pipeline/active. */
export interface PipelineActive {
  task_id?: string;
  status?: string;
}

/** Прогресс шага пайплайна для PipelineProgress. */
export interface PipelineStep {
  step: number;
  total: number;
  status: "running" | "success" | "error" | "completed";
  message: string;
  progress: number;
  maxPages?: number;
  periodDays?: number;
  logs?: string[];
}

interface UsePipelineParams {
  /** Текущий профиль компетенций (догрузка результатов после сбора). */
  profile: string;
  /** Реальная роль пользователя (предпросмотр роли не учитывается). */
  role: string | null;
  /** Записать свежие рекомендации (setLastResult из App). */
  setResult: (data: any) => void;
  /** Перейти на вкладку по завершении (setActiveTab из App). */
  navigate: (next: string) => void;
}

/**
 * Полный цикл HH-пайплайна: запуск, поллинг статуса задачи, отмена, рестарт
 * и переподключение к running-задаче после F5. Разрез из App.tsx без
 * изменения логики (метки reportError сохранены дословно).
 */
export function usePipeline({ profile, role, setResult, navigate }: UsePipelineParams) {
  const [pipelineStep, setPipelineStep] = useState<PipelineStep | null>(null);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [pipelineQuery, setPipelineQuery] = useState("");
  const [pipelineRegions, setPipelineRegions] = useState("");
  const [pipelineMaxPages, setPipelineMaxPages] = useState(20);
  const [pipelinePeriod, setPipelinePeriod] = useState(30);
  const [restartFlag, setRestartFlag] = useState(0);
  const pipelineTaskRef = useRef<string | null>(null);
  const pipelineLoadingRef = useRef(false);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const profileRef = useRef(profile);
  useEffect(() => { profileRef.current = profile; }, [profile]);
  const roleRef = useRef(role);
  useEffect(() => { roleRef.current = role; }, [role]);

  // reconnect to running task after page refresh
  useEffect(() => {
    request<PipelineActive | null>("/api/pipeline/active")
      .catch(() => null)
      .then(active => {
        if (active && active.task_id && active.status === "running") {
          pipelineTaskRef.current = active.task_id;
          sessionStorage.setItem("pipelineTaskId", active.task_id);
          setPipelineLoading(true);
          pipelineLoadingRef.current = true;
          setPipelineStep({ step: 1, total: 4, status: "running", message: "Переподключение...", progress: 5 });
          pollTimerRef.current = setTimeout(pollPipeline, 500);
          return;
        }
        const savedId = sessionStorage.getItem("pipelineTaskId");
        if (savedId) {
          pipelineTaskRef.current = savedId;
          setPipelineLoading(true);
          pipelineLoadingRef.current = true;
          setPipelineStep({ step: 1, total: 4, status: "running", message: "Переподключение...", progress: 5 });
          pollTimerRef.current = setTimeout(pollPipeline, 500);
        }
      });
  }, []);

  const startPipeline = (regionIds: string, profession: string, maxPages?: number, periodDays?: number) => {
    if (pipelineTaskRef.current || pipelineLoadingRef.current) return;
    pipelineLoadingRef.current = true;
    setPipelineLoading(true);
    setPipelineQuery(profession);
    setPipelineRegions(regionIds);
    const mp = maxPages || 20;
    const pd = periodDays || 30;
    setPipelineMaxPages(mp);
    setPipelinePeriod(pd);
    setPipelineStep({ step: 1, total: 4, status: "running", message: "Запуск сбора...", progress: 5, maxPages: mp, periodDays: pd });
    const params = new URLSearchParams({ regions: regionIds, max_pages: String(mp), period: String(pd) });
    if (profession) params.set("query", profession);
    request<{ output?: string }>(`/api/pipeline/full-cycle?${params}`, { method: "POST" })
      .then(data => {
        const m = data.output?.match(/Task ID: (\S+?)\.?\s/);
        if (!m) throw new Error("Не получен ID задачи");
        pipelineTaskRef.current = m[1];
        sessionStorage.setItem("pipelineTaskId", m[1]);
        pollTimerRef.current = setTimeout(pollPipeline, 1000);
      })
      .catch(e => {
        setPipelineStep({ step: 0, total: 1, status: "error", message: String(e), progress: 0 });
        setPipelineLoading(false);
        pipelineLoadingRef.current = false;
      });
  };

  const cancelPipeline = useCallback(() => {
    const taskId = pipelineTaskRef.current;
    if (!taskId) return;
    request<void>(`/api/pipeline/cancel/${taskId}`, { method: "POST" }).catch((err) =>
      reportError("app/App.tsx:cancel", err),
    );
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    setPipelineStep(prev => prev ? { ...prev, status: "error", message: "Отменено" } : null);
    setPipelineLoading(false);
    pipelineTaskRef.current = null;
    pipelineLoadingRef.current = false;
    // keep task id: refresh shows final state + restart button
  }, []);

  const restartPipeline = useCallback(() => {
    cancelPipeline();
    setTimeout(() => {
      setPipelineStep(null);
      setRestartFlag(n => n + 1);
    }, 300);
  }, [cancelPipeline]);

  const pollPipeline = () => {
    const taskId = pipelineTaskRef.current;
    if (!taskId) { setPipelineLoading(false); pipelineLoadingRef.current = false; return; }
    request<{
      step?: number;
      sub_progress?: number;
      status?: string;
      message?: string;
      logs?: string[];
      max_pages?: number;
      period_days?: number;
    }>(`/api/pipeline/task/${taskId}`)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 404) throw new Error("NOT_FOUND");
        throw err;
      })
      .then(s => {
        if (pipelineTaskRef.current !== taskId) return; // cancelled or replaced while fetching
        const step = Math.min(s.step || 1, 4);
        let subProgress = s.sub_progress ?? undefined;
        setPipelineStep(() => {
          const pct = s.status === "completed" ? 100 : Math.min(step * 25, 95);
          return {
            step,
            total: 4,
            status: s.status === "completed" ? "completed" : s.status === "failed" || s.status === "cancelled" ? "error" : "running",
            message: s.message || "Выполняется...",
            progress: pct,
            subProgress: subProgress,
            logs: s.logs ?? [],
          };
        });
            if (s.status === "completed") {
          setPipelineLoading(false);
          pipelineTaskRef.current = null;
          pipelineLoadingRef.current = false;
          sessionStorage.removeItem("pipelineTaskId");
          const prof = profileRef.current;
          request<any>(`/api/results/recommendations/${prof}`).catch(() => null).then(d => {
            if (d) { setResult(d); }
            if (roleRef.current === "student" && d) {
              const q = pipelineQuery;
              const url = q ? `/api/vacancies?limit=1&search=${encodeURIComponent(q)}` : "/api/vacancies/info";
              request<any>(url).catch(() => ({ total: 0 })).then(vi => {
                const vc = vi.total || 0;
                request("/api/student/log-action", {
                  method: "POST",
                  body: {
                    action_type: "analysis",
                    profession: q,
                    region: pipelineRegions,
                    vacancies_found: vc,
                    result_ref: JSON.stringify({ profile: prof }),
                    profile: prof,
                  },
                }).then(() => window.dispatchEvent(new CustomEvent("student-history-update"))).catch((err) => reportError("app/App.tsx", err));
              });
            }
          }).catch((err) => reportError("app/App.tsx", err));
          navigate("data");
          return;
        }
        if (s.status === "failed" || s.status === "cancelled") {
          setPipelineLoading(false);
          pipelineTaskRef.current = null;
          pipelineLoadingRef.current = false;
          sessionStorage.removeItem("pipelineTaskId");
          return;
        }
        pollTimerRef.current = setTimeout(pollPipeline, 2000);
      })
      .catch(e => {
        if (e?.message === "NOT_FOUND") {
          // Задача пропала из памяти (рестарт сервера): проверяем готовые
          // результаты через /api/pipeline/status, а не молча сбрасываем прогресс.
          request<any>("/api/pipeline/status").catch(() => null)
            .then(st => {
              const prof = profileRef.current;
              if (st?.recommendations_all_ready) {
                setPipelineStep({ step: 4, total: 4, status: "completed", message: "Сервер перезапускался, но результаты готовы.", progress: 100 });
                request<any>(`/api/results/recommendations/${prof}`).catch(() => null).then(d => {
                  if (d) { setResult(d); }
                }).catch((err) => reportError("app/App.tsx", err));
                navigate("data");
              } else {
                setPipelineStep(null);
              }
              setPipelineLoading(false);
              pipelineTaskRef.current = null;
              pipelineLoadingRef.current = false;
              sessionStorage.removeItem("pipelineTaskId");
            })
            .catch(() => {
              setPipelineStep(null);
              setPipelineLoading(false);
              pipelineTaskRef.current = null;
              pipelineLoadingRef.current = false;
              sessionStorage.removeItem("pipelineTaskId");
            });
          return;
        }
        if (pipelineTaskRef.current) {
          pollTimerRef.current = setTimeout(pollPipeline, 5000);
        } else {
          setPipelineLoading(false);
          pipelineLoadingRef.current = false;
        }
      });
  };

  useEffect(() => () => { if (pollTimerRef.current) clearTimeout(pollTimerRef.current); }, []);

  return {
    pipelineStep,
    pipelineLoading,
    pipelineMaxPages,
    pipelinePeriod,
    restartFlag,
    startPipeline,
    cancelPipeline,
    restartPipeline,
  };
}
