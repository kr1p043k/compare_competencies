import { useState, useEffect, useRef } from "react";
import { Label } from "./components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Button } from "./components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { GapAnalysisVisualizer } from "./components/GapAnalysisVisualizer";
import { Footer } from "./components/Footer";
import { VacanciesList } from "./components/VacanciesList";
import { ArticlesPage } from "./components/ArticlesPage";
import { ScientificTrendsTab } from "./components/ScientificTrendsTab";
import { PipelineProgress } from "./components/PipelineProgress";
import { DataViewer } from "./components/DataViewer";
import { RecommendationsReport } from "./components/RecommendationsReport";
import { SummaryReport } from "./components/SummaryReport";
import { PredictionsTab } from "./components/PredictionsTab";
import { MonitoringTab } from "./components/MonitoringTab";
import { LogsTab } from "./components/LogsTab";
import { LoginPage } from "./components/LoginPage";
import { AdminDashboard } from "./components/AdminDashboard";
import { TeacherDashboard } from "./components/TeacherDashboard";
import { StudentDashboard } from "./components/StudentDashboard";
import { FaqPage } from "./components/FaqPage";
import { authHeaders, useAuth, apiFetch } from "../lib/auth";
import { initApiLogger } from "../lib/logger";
import { motion, AnimatePresence } from "motion/react";
import {
  Database,
  Sparkles,
  Search,
  FileText,
  FileSpreadsheet,
  Download,
  BarChart3,
  Zap,
  Award,
  Briefcase,
  TrendingUp,
  TrendingDown,
  Info,
  AlertCircle,
  LogOut,
  Shield,
  GraduationCap,
  UserCheck,
  History,
  Activity,
  HelpCircle,
} from "lucide-react";

const API = "/api";

interface PipelineStep {
  step: number;
  total: number;
  status: "running" | "success" | "error" | "completed";
  message: string;
  progress: number;
  maxPages?: number;
  periodDays?: number;
  logs?: string[];
}

function MaintenanceScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-900 to-indigo-900 text-white p-6">
      <div className="text-center max-w-md">
        <div className="text-5xl mb-4">🔧</div>
        <h1 className="text-2xl font-bold mb-3">Извините, на сайте ведутся технические работы</h1>
        <p className="text-gray-200 leading-relaxed mb-6">
          Мы обновляем систему и скоро вернёмся. Пожалуйста, попробуйте зайти чуть позже.
        </p>
        <div className="h-7 w-7 border-[3px] border-white/20 border-t-white rounded-full animate-spin mx-auto" />
      </div>
    </div>
  );
}

export default function App() {
  useEffect(() => { initApiLogger(); }, []);
  const [backendDown, setBackendDown] = useState(false);
  const [profile, setProfile] = useState("base");
  const [status, setStatus] = useState<{
    type: "success" | "error" | "info" | null;
    message: string;
  }>({ type: null, message: "" });
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState<any>(null);
  const [gapRunning, setGapRunning] = useState(false);
  const [gapMsg, setGapMsg] = useState("");
  const [resultLoadedAt, setResultLoadedAt] = useState<Record<string, string>>(() => {
    try {
      return JSON.parse(localStorage.getItem("resultLoadedAt") || "{}");
    } catch {
      return {};
    }
  });
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("vacancies");

  const handleProfileChange = (newProfile: string) => {
    setProfile(newProfile);
    setLastResult(null);
    setAnalysisData(null);
    delete autoLoadedRef.current[newProfile];
  };
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

  const { isAuth, login, logout, role, name } = useAuth();
  const roleRef = useRef(role);
  useEffect(() => { roleRef.current = role; }, [role]);

  // Дата подгрузки переживает перезагрузку (localStorage), а сами результаты — нет.
  // Если штамп есть, а результата в памяти нет — подтягиваем автоматически.
  const autoLoadedRef = useRef<Record<string, boolean>>({});
  useEffect(() => {
    if (!isAuth) return;
    if (autoLoadedRef.current[profile]) return;
    let saved: Record<string, string> | null = null;
    try {
      saved = JSON.parse(localStorage.getItem("resultLoadedAt") || "{}");
    } catch {}
    if (saved && saved[profile]) {
      autoLoadedRef.current[profile] = true;
      loadRecommendations();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuth, profile]);

  // Показываем экран техработ, если backend недоступен (пересборка/рестарт).
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 5000);
        const res = await fetch("/api/health", { signal: ctrl.signal });
        clearTimeout(t);
        if (!cancelled) setBackendDown(!res.ok);
      } catch {
        if (!cancelled) setBackendDown(true);
      }
    };
    check();
    const id = setInterval(check, 8000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // reconnect to running task after page refresh
  useEffect(() => {
    fetch("/api/pipeline/active")
      .then(r => r.ok ? r.json() : null)
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
    fetch(`/api/pipeline/full-cycle?${params}`, { method: "POST" })
      .then(r => r.ok ? r.json() : Promise.reject("Не удалось запустить"))
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

  const cancelPipeline = () => {
    const taskId = pipelineTaskRef.current;
    if (!taskId) return;
    fetch(`/api/pipeline/cancel/${taskId}`, { method: "POST" }).catch(() => {});
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    setPipelineStep(prev => prev ? { ...prev, status: "error", message: "Отменено" } : null);
    setPipelineLoading(false);
    pipelineTaskRef.current = null;
    pipelineLoadingRef.current = false;
    // keep task id: refresh shows final state + restart button
  };

  const restartPipeline = () => {
    cancelPipeline();
    setTimeout(() => {
      setPipelineStep(null);
      setRestartFlag(n => n + 1);
    }, 300);
  };

  const pollPipeline = () => {
    const taskId = pipelineTaskRef.current;
    if (!taskId) { setPipelineLoading(false); pipelineLoadingRef.current = false; return; }
    fetch(`/api/pipeline/task/${taskId}`)
      .then(r => {
        if (r.status === 404) throw new Error("NOT_FOUND");
        return r.ok ? r.json() : Promise.reject("Ошибка статуса");
      })
      .then(s => {
        if (pipelineTaskRef.current !== taskId) return; // cancelled or replaced while fetching
        const step = Math.min(s.step || 1, 4);
        let subProgress = s.sub_progress ?? undefined;
        setPipelineStep(prev => {
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
          fetch(`/api/results/recommendations/${prof}`).then(r => r.ok && r.json()).then(d => {
            if (d) { setAnalysisData(d); setLastResult(d); }
            if (roleRef.current === "student" && d) {
              const q = pipelineQuery;
              const url = q ? `/api/vacancies?limit=1&search=${encodeURIComponent(q)}` : "/api/vacancies/info";
              fetch(url).then(r => r.ok ? r.json() : { total: 0 }).then(vi => {
                const vc = vi.total || 0;
                apiFetch("/api/student/log-action", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    action_type: "analysis",
                    profession: q,
                    region: pipelineRegions,
                    vacancies_found: vc,
                    result_ref: JSON.stringify({ profile: prof }),
                    profile: prof,
                  }),
                }).then(() => window.dispatchEvent(new CustomEvent("student-history-update"))).catch(() => {});
              });
            }
          }).catch(() => {});
          setActiveTab("data");
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
          fetch("/api/pipeline/status")
            .then(r => r.ok ? r.json() : null)
            .then(st => {
              const prof = profileRef.current;
              if (st?.recommendations_all_ready) {
                setPipelineStep({ step: 4, total: 4, status: "completed", message: "Сервер перезапускался, но результаты готовы.", progress: 100 });
                fetch(`/api/results/recommendations/${prof}`).then(r => r.ok && r.json()).then(d => {
                  if (d) { setAnalysisData(d); setLastResult(d); }
                }).catch(() => {});
                setActiveTab("data");
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

  // navigate-analysis event from StudentDashboard
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.profile) {
        handleProfileChange(detail.profile);
        setActiveTab("data");
        fetch(`/api/results/recommendations/${detail.profile}`).then(r => r.ok && r.json()).then(d => { if (d) { setAnalysisData(d); setLastResult(d); } }).catch(() => {});
      }
    };
    window.addEventListener("navigate-analysis", handler);
    return () => window.removeEventListener("navigate-analysis", handler);
  }, []);

  function showStatus(type: "success" | "error" | "info", message: string) {
    setStatus({ type, message });
    setTimeout(() => setStatus({ type: null, message: "" }), 5000);
  }

  async function apiCall(endpoint: string, method = "GET", body: any = null) {
    try {
      setLoading(true);
      showStatus("info", "Выполнение...");
      const opts: RequestInit = {
        method,
        headers: { "Content-Type": "application/json" },
      };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(`${API}${endpoint}`, opts);
      const data = res.ok ? await res.json() : await res.text().then(t => { throw new Error(t || res.statusText); });
      setLastResult(data);
      showStatus(
        res.ok ? "success" : "error",
        res.ok ? "✓ Готово" : "✗ Ошибка"
      );
      return data;
    } catch (e: any) {
      showStatus("error", `✗ ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function runGapAnalysis() {
    if (gapRunning) return;
    setGapRunning(true);
    setGapMsg("Запуск gap-анализа...");
    const checkResultsReady = async (): Promise<boolean> => {
      try {
        const st = await fetch("/api/pipeline/status").then((x) =>
          x.ok ? x.json() : null
        );
        return !!st?.recommendations_all_ready;
      } catch {
        return false;
      }
    };
    try {
      const r = await fetch("/api/pipeline/gap-analysis", { method: "POST" });
      if (!r.ok) throw new Error("Не удалось запустить задачу gap-анализа");
      const data = await r.json();
      const m = String(data.output || "").match(/Task ID: (\S+?)\.?\s/);
      if (!m) throw new Error("Не получен ID задачи");
      const taskId = m[1];
      let statusMisses = 0;
      for (;;) {
        await new Promise((res) => setTimeout(res, 3000));
        let s: any = null;
        try {
          s = await fetch(`/api/pipeline/task/${taskId}`).then((x) =>
            x.ok ? x.json() : null
          );
        } catch {
          s = null;
        }
        if (!s) {
          // Сервер может перезапускаться: не сдаёмся сразу, ждём до ~15 сек.
          statusMisses += 1;
          setGapMsg(`Сервер перезапускается, жду статус... (${statusMisses})`);
          if (statusMisses < 5) continue;
          setGapMsg("Статус задачи недоступен, проверяю результат...");
          break;
        }
        statusMisses = 0;
        setGapMsg(`${s.message || "Выполняется..."} (шаг ${s.step ?? "?"})`);
        if (s.status === "completed") {
          setGapMsg("Готово, обновляю данные...");
          break;
        }
        if (s.status === "failed" || s.status === "cancelled") {
          const msg = String(s.message || "");
          // После рестарта бэкенд помечает висевшую задачу как
          // "Прерван перезапуском сервера", хотя файлы результата уже готовы.
          // Проверяем факты вместо доверия протухшему статусу.
          if (/перезапуск/i.test(msg)) {
            const ready = await checkResultsReady();
            if (ready) {
              setGapMsg("Сервер перезапускался, но результаты готовы. Обновляю данные...");
              break;
            }
          }
          throw new Error(msg || "Задача завершилась с ошибкой");
        }
      }
      loadProfileDetail();
      loadRecommendations();
    } catch (e: any) {
      setGapMsg("");
      alert(`Ошибка gap-анализа: ${e?.message || e}`);
    } finally {
      setGapRunning(false);
    }
  }

  function loadProfileDetail() {
    apiCall(`/profiles/${profile}`);
  }

  async function loadRecommendations() {
    const data = await apiCall(`/results/recommendations/${profile}`);
    if (data) {
      setAnalysisData(data);
      const hasResults = !!(data.recommendations || data.closest_roles);
      const notFound = typeof data.message === "string" && data.message.includes("не найдены");
      if (hasResults && !notFound && typeof data.generated_at === "string") {
        const stamp = data.generated_at as string;
        setResultLoadedAt((prev) => {
          const next = { ...prev, [profile]: stamp };
          try {
            localStorage.setItem("resultLoadedAt", JSON.stringify(next));
          } catch {}
          return next;
        });
      }
    }
  }

  async function handleDownloadExcel() {
    try {
      const response = await fetch(`/api/teacher/export/vacancies`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `vacancies_${new Date().toISOString().split("T")[0]}.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else if (response.status === 429) {
        alert("Слишком частые запросы. Подождите 20 секунд.");
      } else {
        const d = await response.json().catch(() => ({}));
        alert(d.detail || "Ошибка выгрузки Excel");
      }
    } catch (error) {
      console.error("Failed to download Excel:", error);
    }
  }

  async function handleDownloadReport() {
    try {
      const response = await fetch(`/api/results/recommendations/${profile}`);
      if (response.ok) {
        const data = await response.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `analysis_report_${profile}_${new Date().toISOString().split("T")[0]}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error("Failed to download analysis report:", error);
    }
  }

  function loadMarket() {
    apiCall("/market-competencies");
  }

  function loadSummary() {
    apiCall("/results/summary");
  }

  async function loadHealth() {
    try {
      setLoading(true);
      showStatus("info", "Выполнение...");
      const res = await fetch("/health");
      const data = await res.json();
      setLastResult(data);
      showStatus(res.ok ? "success" : "error", res.ok ? "✓ Готово" : `✗ ${data.detail || "Ошибка"}`);
      return data;
    } catch (e: any) {
      showStatus("error", `✗ ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  if (backendDown) {
    return <MaintenanceScreen />;
  }

  if (!isAuth) {
    return <LoginPage onLogin={login} />;
  }

  const roleIcon = role === "admin" ? <Shield className="size-4" /> : (role === "teacher" || role === "rop") ? <UserCheck className="size-4" /> : <GraduationCap className="size-4" />;
  const roleLabel = role === "admin" ? "Администратор" : role === "teacher" ? "Преподаватель" : role === "rop" ? "Руководитель ОП" : "Студент";

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center justify-center w-12 h-12 bg-blue-600 rounded-xl">
                <TrendingUp className="size-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Competency Gap Analyzer
                </h1>
                <p className="text-sm text-gray-600">
                  AI-powered competency analysis platform
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                {roleIcon}
                <span>{name || roleLabel}</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100">{roleLabel}</span>
              </div>
              <Button variant="ghost" size="sm" onClick={() => { fetch("/api/auth/logout", { method: "POST", headers: authHeaders() }).catch(() => {}); logout(); }} className="text-gray-500 hover:text-red-600">
                <LogOut className="size-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Status */}
        <AnimatePresence>
          {status.type && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6"
            >
              <div
                className={`px-4 py-3 rounded-lg border ${
                  status.type === "success"
                    ? "bg-green-50 border-green-200 text-green-800"
                    : status.type === "error"
                      ? "bg-red-50 border-red-200 text-red-800"
                      : "bg-blue-50 border-blue-200 text-blue-800"
                }`}
              >
                {status.message}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="inline-flex h-12 items-center justify-center rounded-lg bg-gray-100 p-1">
            <TabsTrigger
              value="vacancies"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <Briefcase className="size-4" />
              Вакансии
            </TabsTrigger>
            <TabsTrigger
              value="data"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <Database className="size-4" />
              Данные
            </TabsTrigger>
            {role !== "teacher" && (
              <TabsTrigger
                value="visualization"
                className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
              >
                <BarChart3 className="size-4" />
                Визуализация
              </TabsTrigger>
            )}
            <TabsTrigger
              value="predictions"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <TrendingUp className="size-4" />
              Прогнозы
            </TabsTrigger>
            <TabsTrigger
              value="articles"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <BarChart3 className="size-4" />
              Аналитика рынка
            </TabsTrigger>
            <TabsTrigger
              value="scientific-trends"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <TrendingUp className="size-4" />
              Научные тренды
            </TabsTrigger>
            <TabsTrigger
              value="help"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <HelpCircle className="size-4" />
              Помощь
            </TabsTrigger>
            {role === "admin" && (
              <TabsTrigger value="monitoring" className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm">
                <Activity className="size-4" />
                Мониторинг
              </TabsTrigger>
            )}
            {role === "admin" && (
              <TabsTrigger value="logs" className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm">
                <FileText className="size-4" />
                Логи
              </TabsTrigger>
            )}
            {role === "admin" && (
              <TabsTrigger value="admin" className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm">
                <Shield className="size-4" />
                Админ
              </TabsTrigger>
            )}
            {(role === "teacher" || role === "rop") && (
              <TabsTrigger value="teacher" className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm">
                <BarChart3 className="size-4" />
                Статистика
              </TabsTrigger>
            )}
            {role === "student" && (
              <TabsTrigger value="student" className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm">
                <History className="size-4" />
                Мои запросы
              </TabsTrigger>
            )}
          </TabsList>

          {/* Pipeline progress */}
          {pipelineStep && (
            <div className="mb-4">
              <PipelineProgress currentStep={pipelineStep} onCancel={cancelPipeline} onRestart={restartPipeline} showLogs={activeTab === "admin"} />
            </div>
          )}

          {/* Vacancies Tab */}
          <TabsContent value="vacancies">
            <VacanciesList
              pipelineStep={pipelineStep}
              pipelineLoading={pipelineLoading}
              restartFlag={restartFlag}
              onStartPipeline={(regionIds, profession, maxPages, periodDays) => startPipeline(regionIds, profession, maxPages, periodDays)}
              pipelineMaxPages={pipelineMaxPages}
              pipelinePeriod={pipelinePeriod}
            />
          </TabsContent>

          {/* Data Tab */}
          <TabsContent value="data">
            <Card className="border border-gray-200 shadow-sm">
              <CardHeader className="border-b border-gray-200 bg-gray-50">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 bg-emerald-600 rounded-lg">
                    <Database className="size-5 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-xl font-semibold text-gray-900">
                      Данные и результаты
                    </CardTitle>
                    <CardDescription className="text-sm text-gray-600">
                      Просмотр профилей, рекомендаций и статистики
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-6 space-y-6">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-gray-900">
                    Профиль компетенций
                  </Label>
                  <Select value={profile} onValueChange={handleProfileChange}>
                    <SelectTrigger className="h-11 bg-white border-gray-300">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="base">
                        <div className="flex items-center gap-2">
                          <Award className="size-4 text-blue-600" />
                          <span>BASE (junior)</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="dc">
                        <div className="flex items-center gap-2">
                          <Award className="size-4 text-purple-600" />
                          <span>DATA SCIENTIST (middle)</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="top_dc">
                        <div className="flex items-center gap-2">
                          <Award className="size-4 text-pink-600" />
                          <span>TOP DATA SCIENTIST (senior)</span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                  <Button
                    onClick={loadRecommendations}
                    disabled={loading}
                    className="h-11 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
                  >
                    <Search className="mr-2 size-4" />
                    Загрузить результаты
                  </Button>
                  <Button
                    onClick={loadProfileDetail}
                    disabled={loading}
                    className="h-11 bg-emerald-600 hover:bg-emerald-700 text-white"
                  >
                    <FileText className="mr-2 size-4" />
                    Профиль
                  </Button>
                  <Button
                    onClick={loadRecommendations}
                    disabled={loading}
                    className="h-11 bg-purple-600 hover:bg-purple-700 text-white"
                  >
                    <Sparkles className="mr-2 size-4" />
                    Рекомендации
                  </Button>
                  <Button
                    onClick={loadMarket}
                    disabled={loading}
                    variant="outline"
                    className="h-11 border-gray-300 text-gray-700 hover:bg-gray-50"
                  >
                    <BarChart3 className="mr-2 size-4" />
                    Рынок
                  </Button>
                  <Button
                    onClick={loadSummary}
                    disabled={loading}
                    variant="outline"
                    className="h-11 border-gray-300 text-gray-700 hover:bg-gray-50"
                  >
                    <FileText className="mr-2 size-4" />
                    Сводка
                  </Button>
                  <Button
                    onClick={loadHealth}
                    disabled={loading}
                    variant="outline"
                    className="h-11 border-gray-300 text-gray-700 hover:bg-gray-50"
                  >
                    <Zap className="mr-2 size-4" />
                    Проверка
                  </Button>
                  <Button
                    onClick={runGapAnalysis}
                    disabled={loading || gapRunning}
                    className="h-11 bg-amber-600 hover:bg-amber-700 text-white"
                  >
                    <Zap className="mr-2 size-4" />
                    Запустить gap-анализ
                  </Button>
                </div>
                {gapRunning && (
                  <p className="text-sm text-amber-700">{gapMsg || "Выполняется..."}</p>
                )}
                <p className="text-xs text-gray-500">
                  Последняя подгрузка результатов [{profile}]: {(() => {
                    const iso = resultLoadedAt[profile];
                    if (!iso) return "ещё не подгружались";
                    const d = new Date(iso);
                    return isNaN(d.getTime()) ? iso : d.toLocaleString("ru-RU");
                  })()}
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="border-2 border-green-200 dark:border-green-800 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20">
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-green-600 rounded-lg">
                          <FileSpreadsheet className="size-5 text-white" />
                        </div>
                        <div>
                          <CardTitle className="text-lg">Excel вакансий</CardTitle>
                          <CardDescription>Скачать список вакансий с навыками</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Button
                        onClick={handleDownloadExcel}
                        variant="outline"
                        className="w-full border-green-300 dark:border-green-700 hover:bg-green-100 dark:hover:bg-green-900/50"
                      >
                        <Download className="size-4 mr-2" />
                        Скачать Excel
                      </Button>
                    </CardContent>
                  </Card>

                  <Card className="border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20">
                    <CardHeader>
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-600 rounded-lg">
                          <FileText className="size-5 text-white" />
                        </div>
                        <div>
                          <CardTitle className="text-lg">Отчёт по анализу</CardTitle>
                          <CardDescription>Скачать результаты gap-анализа</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Button
                        onClick={handleDownloadReport}
                        variant="outline"
                        className="w-full border-blue-300 dark:border-blue-700 hover:bg-blue-100 dark:hover:bg-blue-900/50"
                      >
                        <Download className="size-4 mr-2" />
                        Скачать отчёт
                      </Button>
                    </CardContent>
                  </Card>
                </div>

                {lastResult && (() => {
                  const d = lastResult as Record<string, unknown>;
                  if (d.recommendations || d.closest_roles) {
                    return <RecommendationsReport data={lastResult as any} />;
                  }
                  if (d.evaluations && Array.isArray(d.profiles)) {
                    return <SummaryReport data={lastResult as any} />;
                  }
                  const msg = d.message as string | undefined;
                  if (msg && (msg.includes("не найдены") || msg.includes("not found"))) {
                    return (
                      <Card className="border-2 border-amber-200 bg-amber-50/50">
                        <CardContent className="pt-6 text-center py-12">
                          <AlertCircle className="size-12 text-amber-400 mx-auto mb-4" />
                          <h3 className="text-lg font-semibold text-amber-800 mb-2">{msg}</h3>
                          <p className="text-sm text-amber-600 mb-4">Запустите gap-анализ для расчёта покрытия</p>
                          <Button
                            onClick={runGapAnalysis}
                            disabled={gapRunning}
                            className="bg-amber-600 hover:bg-amber-700"
                          >
                            <Zap className="size-4 mr-2" />
                            Запустить gap-анализ
                          </Button>
                          {gapRunning && (
                            <p className="text-sm text-amber-700 mt-3">{gapMsg || "Выполняется..."}</p>
                          )}
                        </CardContent>
                      </Card>
                    );
                  }
                  return <DataViewer data={lastResult} />;
                })()}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Visualization Tab */}
          {role !== "teacher" && (
            <TabsContent value="visualization">
              <GapAnalysisVisualizer profile={profile} onProfileChange={handleProfileChange} />
            </TabsContent>
          )}

          <TabsContent value="predictions">
            <PredictionsTab />
          </TabsContent>
          <TabsContent value="articles">
            <ArticlesPage />
          </TabsContent>
          <TabsContent value="scientific-trends">
            <ScientificTrendsTab />
          </TabsContent>
          <TabsContent value="help">
            <FaqPage />
          </TabsContent>
          {role === "admin" && (
            <TabsContent value="monitoring">
              <MonitoringTab />
            </TabsContent>
          )}
          {role === "admin" && (
            <TabsContent value="logs">
              <LogsTab />
            </TabsContent>
          )}
          {role === "admin" && (
            <TabsContent value="admin">
              <AdminDashboard />
            </TabsContent>
          )}
          {(role === "teacher" || role === "rop") && (
            <TabsContent value="teacher">
              <TeacherDashboard />
            </TabsContent>
          )}
          {role === "student" && (
            <TabsContent value="student">
              <StudentDashboard />
            </TabsContent>
          )}
        </Tabs>

        <div className="mt-12">
          <Footer />
        </div>
      </main>
    </div>
  );
}
