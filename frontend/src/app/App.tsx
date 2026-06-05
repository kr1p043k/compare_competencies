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
import { AnalysisTab } from "./components/AnalysisTab";
import { PipelineProgress } from "./components/PipelineProgress";
import { DataViewer } from "./components/DataViewer";
import { RecommendationsReport } from "./components/RecommendationsReport";
import { PredictionsTab } from "./components/PredictionsTab";
import { MonitoringTab } from "./components/MonitoringTab";
import { LoginPage } from "./components/LoginPage";
import { AdminDashboard } from "./components/AdminDashboard";
import { TeacherDashboard } from "./components/TeacherDashboard";
import { StudentDashboard } from "./components/StudentDashboard";
import { authHeaders, useAuth, apiFetch } from "../lib/auth";
import { initApiLogger } from "../lib/logger";
import { motion, AnimatePresence } from "motion/react";
import {
  Database,
  Sparkles,
  FileText,
  BarChart3,
  Zap,
  Award,
  Briefcase,
  TrendingUp,
  TrendingDown,
  Target,
  Info,
  AlertCircle,
  LogOut,
  Shield,
  GraduationCap,
  UserCheck,
  History,
  Activity,
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

export default function App() {
  useEffect(() => { initApiLogger(); }, []);
  const [profile, setProfile] = useState("base");
  const [status, setStatus] = useState<{
    type: "success" | "error" | "info" | null;
    message: string;
  }>({ type: null, message: "" });
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState<any>(null);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("vacancies");

  const handleProfileChange = (newProfile: string) => {
    setProfile(newProfile);
    setLastResult(null);
    setAnalysisData(null);
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
    sessionStorage.removeItem("pipelineTaskId");
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
            if (role === "student" && d) {
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
          setActiveTab("analysis");
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
          setPipelineStep(null);
          setPipelineLoading(false);
          pipelineTaskRef.current = null;
          pipelineLoadingRef.current = false;
          sessionStorage.removeItem("pipelineTaskId");
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
        setActiveTab("analysis");
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
      const data = await res.json();
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

  function loadProfileDetail() {
    apiCall(`/profiles/${profile}`);
  }

  async function loadRecommendations() {
    const data = await apiCall(`/results/recommendations/${profile}`);
    if (data) setAnalysisData(data);
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

  const { isAuth, login, logout, role, name } = useAuth();

  if (!isAuth) {
    return <LoginPage onLogin={login} />;
  }

  const roleIcon = role === "admin" ? <Shield className="size-4" /> : role === "teacher" ? <UserCheck className="size-4" /> : <GraduationCap className="size-4" />;
  const roleLabel = role === "admin" ? "Администратор" : role === "teacher" ? "Преподаватель" : "Студент";

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
            <TabsTrigger
              value="visualization"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <BarChart3 className="size-4" />
              Визуализация
            </TabsTrigger>
            <TabsTrigger
              value="predictions"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <TrendingUp className="size-4" />
              Прогнозы
            </TabsTrigger>
            <TabsTrigger
              value="analysis"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <Target className="size-4" />
              Анализ
            </TabsTrigger>
            {role === "admin" && (
              <TabsTrigger value="monitoring" className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm">
                <Activity className="size-4" />
                Мониторинг
              </TabsTrigger>
            )}
            {role === "admin" && (
              <TabsTrigger value="admin" className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm">
                <Shield className="size-4" />
                Админ
              </TabsTrigger>
            )}
            {role === "teacher" && (
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
                </div>

                {lastResult && (() => {
                  const d = lastResult as Record<string, unknown>;
                  if (d.recommendations || d.closest_roles) {
                    return <RecommendationsReport data={lastResult as any} />;
                  }
                  const msg = d.message as string | undefined;
                  if (msg && (msg.includes("не найдены") || msg.includes("not found"))) {
                    return (
                      <Card className="border-2 border-amber-200 bg-amber-50/50">
                        <CardContent className="pt-6 text-center py-12">
                          <AlertCircle className="size-12 text-amber-400 mx-auto mb-4" />
                          <h3 className="text-lg font-semibold text-amber-800 mb-2">{msg}</h3>
                          <p className="text-sm text-amber-600">Запустите анализ компетенций через вкладку «Анализ»</p>
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
          <TabsContent value="visualization">
            <GapAnalysisVisualizer profile={profile} />
          </TabsContent>

          {/* Analysis Tab */}
          <TabsContent value="analysis">
            <AnalysisTab
              selectedProfile={profile}
              onProfileChange={handleProfileChange}
              pipelineQuery={pipelineQuery}
              pipelineRegions={pipelineRegions}
              analysisData={analysisData}
              onDataLoaded={(data) => {
                setAnalysisData(data); setLastResult(data);
                if (role === "student" && data) {
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
                        result_ref: JSON.stringify({ profile }),
                        profile,
                      }),
                    }).then(() => window.dispatchEvent(new CustomEvent("student-history-update"))).catch(() => {});
                  });
                }
              }}
            />
          </TabsContent>
          <TabsContent value="predictions">
            <PredictionsTab />
          </TabsContent>
          {role === "admin" && (
            <TabsContent value="monitoring">
              <MonitoringTab />
            </TabsContent>
          )}
          {role === "admin" && (
            <TabsContent value="admin">
              <AdminDashboard />
            </TabsContent>
          )}
          {role === "teacher" && (
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
