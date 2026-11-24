/**
 * Загрузка и обработка РПД для одного направления: файл PDF или папка на
 * Yandex Disk. Вынесено из TeacherDashboard, чтобы контейнер занимался
 * раскладкой, а не поллингом фоновой задачи.
 *
 * Поллинг не зависит от того, открыт ли дровер «Данные и РПД»: закрыл панель -
 * прогресс всё равно доедет до конца и обновит анализ.
 */
import { useEffect, useRef, useState } from "react";
import { ApiError, request } from "../../../lib/api";
import { errMessage, reportError } from "../../../lib/errors";
import type { RpdStatus } from "./types";

const STORAGE_KEY = "rpdRunId";

export function useRpdProcessing(dirCode: string, onCompleted: () => void) {
  const [sources, setSources] = useState<{ yandex_covered: string[] }>({ yandex_covered: [] });
  const [uploading, setUploading] = useState(false);
  const [collecting, setCollecting] = useState(false);
  const [yandexUrl, setYandexUrl] = useState("");
  const [run, setRun] = useState<{ run_id: string } | null>(null);
  const [status, setStatus] = useState<RpdStatus | null>(null);
  const [msg, setMsg] = useState("");
  const [cooldown, setCooldown] = useState(0);

  const onCompletedRef = useRef(onCompleted);
  onCompletedRef.current = onCompleted;

  useEffect(() => {
    request<{ yandex_covered: string[] }>("/api/teacher/rpd/sources")
      .then(setSources)
      .catch((err) => reportError("app/components/teacher-dashboard/useRpdProcessing.ts", err));
  }, []);

  // Возобновление поллинга после переключения вкладок (табы размонтируют экран).
  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = sessionStorage.getItem(STORAGE_KEY);
    } catch (err) {
      reportError("app/components/teacher-dashboard/useRpdProcessing.ts", err);
    }
    if (stored) {
      setRun({ run_id: stored });
      setCollecting(true);
    }
  }, []);

  useEffect(() => {
    if (cooldown <= 0) return;
    const t = setTimeout(() => setCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  useEffect(() => {
    if (!run?.run_id) return;
    let cancelled = false;
    let misses = 0;
    const poll = () => {
      request<RpdStatus>(`/api/teacher/rpd/status/${run.run_id}`)
        .then((s) => {
          if (cancelled) return;
          misses = 0;
          setStatus(s);
          if (s?.status === "completed" || s?.status === "failed" || s?.status === "cancelled") {
            setUploading(false);
            setCollecting(false);
            try {
              sessionStorage.removeItem(STORAGE_KEY);
            } catch (err) {
              reportError("app/components/teacher-dashboard/useRpdProcessing.ts", err);
            }
            setMsg(
              s.status === "completed"
                ? "Готово"
                : s.status === "cancelled"
                  ? "Отменено пользователем"
                  : `Ошибка: ${s.error || "unknown"}`,
            );
            if (s.status === "completed") onCompletedRef.current();
          } else {
            setTimeout(poll, 3000);
          }
        })
        .catch(() => {
          if (cancelled) return;
          // Разовый сбой сети/рестарт сервера - ждём, а не висим молча.
          misses += 1;
          if (misses < 10) {
            setMsg(`Сервер перезапускается, жду статус... (${misses})`);
            setTimeout(poll, 5000);
          } else {
            setUploading(false);
            setCollecting(false);
            setMsg("Ошибка: статус задачи недоступен. Проверьте сервер и повторите.");
          }
        });
    };
    poll();
    return () => {
      cancelled = true;
    };
  }, [run?.run_id]);

  async function uploadFile(file: File) {
    if (uploading) return;
    setUploading(true);
    setMsg("Загрузка...");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("dir_code", dirCode);
    try {
      const data = await request<{ run_id?: string }>("/api/teacher/rpd/upload", { method: "POST", body: fd });
      if (!data.run_id) throw new Error("Сервер не вернул ID задачи");
      try {
        sessionStorage.setItem(STORAGE_KEY, data.run_id);
      } catch (err) {
        reportError("app/components/teacher-dashboard/useRpdProcessing.ts", err);
      }
      setRun({ run_id: data.run_id });
      setMsg("Обработка запущена...");
    } catch (e) {
      setMsg("Ошибка: " + errMessage(e));
      setUploading(false);
    }
  }

  async function collect() {
    if (cooldown > 0) return;
    if (collecting) {
      // Повторный клик во время сбора = отмена.
      if (!run?.run_id) {
        setCollecting(false);
        setMsg("");
        return;
      }
      try {
        setMsg("Останавливаю сбор...");
        await request<void>(`/api/teacher/rpd/cancel/${run.run_id}`, { method: "POST" });
      } catch {
        setCollecting(false);
        setMsg("");
      }
      return;
    }
    setCollecting(true);
    setMsg("Сбор аннотаций с Yandex Disk...");
    try {
      const body = new URLSearchParams({ dir_code: dirCode });
      if (yandexUrl.trim()) body.set("public_url", yandexUrl.trim());
      let data: { run_id?: string };
      try {
        data = await request<{ run_id?: string }>("/api/teacher/rpd/collect", { method: "POST", body });
      } catch (err) {
        if (err instanceof ApiError && err.isRateLimited) {
          setCooldown(30);
          throw new Error("Слишком частые запросы - кнопка заблокирована на 30 секунд", { cause: err });
        }
        throw err;
      }
      if (!data.run_id) throw new Error("Сервер не вернул ID задачи");
      try {
        sessionStorage.setItem(STORAGE_KEY, data.run_id);
      } catch (err) {
        reportError("app/components/teacher-dashboard/useRpdProcessing.ts", err);
      }
      setRun({ run_id: data.run_id });
      setMsg("Сбор и обработка запущены...");
    } catch (e) {
      setMsg("Ошибка: " + errMessage(e));
      setCollecting(false);
    }
  }

  return {
    sources,
    uploading,
    collecting,
    yandexUrl,
    setYandexUrl,
    status,
    msg,
    cooldown,
    busy: uploading || collecting,
    uploadFile,
    collect,
  };
}
