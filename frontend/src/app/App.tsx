import { useState, useEffect, useRef, Suspense, lazy, memo } from "react";
import { Tabs, TabsContent } from "./components/ui/tabs";
import ErrorBoundary from "../design/ErrorBoundary";
import { Skeleton } from "../design";
const GapAnalysisVisualizer = lazy(() => import("./components/GapAnalysisVisualizer").then(m => ({ default: m.GapAnalysisVisualizer })));
import { Footer } from "./components/Footer";
import { VacanciesList } from "./components/VacanciesList";
const ArticlesPage = lazy(() => import("./components/ArticlesPage").then(m => ({ default: m.ArticlesPage })));
import { ScientificTrendsTab } from "./components/ScientificTrendsTab";
const PublicationsTab = lazy(() => import("./components/PublicationsTab").then(m => ({ default: m.PublicationsTab })));
import { ArticlesPanel } from "./components/ArticlesPanel";
import { PipelineProgress } from "./components/PipelineProgress";
import { PredictionsTab } from "./components/PredictionsTab";
const MonitoringTab = lazy(() => import("./components/MonitoringTab").then(m => ({ default: m.MonitoringTab })));
const LogsTab = lazy(() => import("./components/LogsTab").then(m => ({ default: m.LogsTab })));
import { LoginPage } from "./components/LoginPage";
const AdminDashboard = lazy(() => import("./components/AdminDashboard").then(m => ({ default: m.AdminDashboard })));
const StudentsTab = lazy(() => import("./components/StudentsTab").then(m => ({ default: m.StudentsTab })));
const TeacherDashboard = lazy(() => import("./components/TeacherDashboard").then(m => ({ default: m.TeacherDashboard })));
const StudentDashboard = lazy(() => import("./components/StudentDashboard").then(m => ({ default: m.StudentDashboard })));
const AdminProfilePage = lazy(() => import("./components/ProfilePages").then(m => ({ default: m.AdminProfilePage })));
const TeacherProfilePage = lazy(() => import("./components/ProfilePages").then(m => ({ default: m.TeacherProfilePage })));
const StudentProfilePage = lazy(() => import("./components/ProfilePages").then(m => ({ default: m.StudentProfilePage })));
const FaqPage = lazy(() => import("./components/FaqPage").then(m => ({ default: m.FaqPage })));
const TaxonomyBrowser = lazy(() => import("./components/TaxonomyBrowser").then(m => ({ default: m.TaxonomyBrowser })));
import { useAuth } from "../lib/auth";
import { download, request, saveBlob } from "../lib/api";
import { useTheme } from "../lib/theme";
import { initApiLogger } from "../lib/logger";
import { motion, AnimatePresence } from "motion/react";
import { reportError } from "./../lib/errors";
import { AppHeader, AppNavTabs } from "./shell/AppHeader";
import { isTabAllowedForRole } from "./shell/tabConfig";
import { useTabNavigation } from "./shell/useTabNavigation";
import { usePipeline } from "./shell/usePipeline";
import { DataTab } from "./components/DataTab";

const API = "/api";

// Fallback для ЛОКАЛЬНОГО Suspense одной вкладки: маленький инлайн-спиннер.
// Грузится только переключаемая вкладка - вся область контента больше не гасится.
function TabSpinner() {
  return (
    <div
      className="p-6 flex items-center gap-2 text-sm text-gray-500"
      role="status"
      aria-live="polite"
    >
      <span
        className="mr-2 inline-block size-4 shrink-0 rounded-full border-2 border-gray-300 border-t-gray-600 animate-spin motion-reduce:animate-none"
        aria-hidden="true"
      />
      Загрузка раздела…
    </div>
  );
}

// Fallback вкладки «Аналитика рынка»: в её графе recharts (vendor-charts),
// поэтому до догрузки чанка вместо графиков показываем layout-скелетон.
// ArticlesPage - React.lazy (см. объявление ниже), стилистика фолбэка
// повторяет сетку графиков AnalyticsCharts, пропсы самой вкладки не меняются.
function TabSkeleton() {
  return (
    <div className="space-y-4" role="status" aria-live="polite">
      <span className="sr-only">Загружается аналитика рынка…</span>
      <Skeleton className="h-8 w-72" />
      <div className="grid gap-4 lg:grid-cols-2">
        <Skeleton className="h-72 w-full" />
        <Skeleton className="h-72 w-full" />
      </div>
      <Skeleton className="h-40 w-full" />
    </div>
  );
}

// Тяжёлые списки под React.memo: не перерисовываются от ререндеров App
// (health-poll, статусы, шаги пайплайна), только от собственного стейта.
const MemoPredictionsTab = memo(PredictionsTab);
const MemoScientificTrendsTab = memo(ScientificTrendsTab);
const MemoPipelineProgress = memo(PipelineProgress);

function MaintenanceScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-900 text-white p-6">
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
  useEffect(() => { initApiLogger(); loadProfiles(); }, []);
  const [backendDown, setBackendDown] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(t);
  }, [toast]);
  const [profile, setProfile] = useState("base");
  const [profilesList, setProfilesList] = useState<string[]>(["base", "dc", "top_dc"]);

  async function loadProfiles(select?: string) {
    try {
      const d = await request<{ profiles?: string[] }>(`${API}/profiles`);
      const names = Array.isArray(d.profiles) && d.profiles.length ? d.profiles : ["base", "dc", "top_dc"];
      setProfilesList(names);
      if (select && names.includes(select)) setProfile(select);
      else if (!names.includes(profile)) setProfile(names[0]);
    } catch { /* keep hardcoded fallback */ }
  }

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
  const [professionsList, setProfessionsList] = useState<string[]>([]);
  const [targetProfession, setTargetProfession] = useState("");

  // Дата подгрузки переживает перезагрузку (localStorage), а сами результаты - нет.
  // Если штамп есть, а результата в памяти нет - подтягиваем автоматически.
  const autoLoadedRef = useRef<Record<string, boolean>>({});

  const handleProfileChange = (newProfile: string) => {
    setProfile(newProfile);
    setLastResult(null);
    delete autoLoadedRef.current[newProfile];
  };

  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const gapCancelRef = useRef(false);

  // Размонтирование: гасим таймер статуса и останавливаем поллинг gap-анализа
  // (иначе цикл живёт в фоне и делает setState в никуда).
  // Таймер поллинга пайплайна чистится внутри usePipeline.
  useEffect(() => () => {
    gapCancelRef.current = true;
    if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
  }, []);

  const { isAuth, login, logout, role, name, username } = useAuth();
  const { theme, toggle: toggleTheme } = useTheme();

  // Предпросмотр чужой роли (только admin, только UI: API-права не меняются).
  const [rolePreview, setRolePreview] = useState<string | null>(() => {
    try {
      const v = localStorage.getItem("rolePreview");
      return v === "admin" || v === "teacher" || v === "rop" || v === "student" ? v : null;
    } catch { return null; }
  });
  const canPreview = role === "admin";
  const effectiveRole = canPreview && rolePreview ? rolePreview : role;
  const previewActive = canPreview && !!rolePreview && rolePreview !== role;

  // Вкладка синхронизирована с адресом: #/teacher и т.п. (deep links + история),
  // при смене роли (или предпросмотра) - уход с недоступной вкладки на «vacancies».
  const [activeTab, setActiveTab] = useTabNavigation("vacancies", effectiveRole);

  const setPreview = (v: string | null) => {
    setRolePreview(v);
    try {
      if (v) localStorage.setItem("rolePreview", v);
      else localStorage.removeItem("rolePreview");
    } catch (err) { reportError("app/App.tsx:274", err); }
    setActiveTab("vacancies");
  };
  useEffect(() => {
    if (role !== "admin") {
      setRolePreview(null);
      try { localStorage.removeItem("rolePreview"); } catch (err) { reportError("app/App.tsx:280", err); }
    }
  }, [role]);

  // Полный цикл HH-сбора: запуск, поллинг, отмена, рестарт, переподключение.
  const {
    pipelineStep,
    pipelineLoading,
    pipelineMaxPages,
    pipelinePeriod,
    restartFlag,
    startPipeline,
    cancelPipeline,
    restartPipeline,
  } = usePipeline({ profile, role, setResult: setLastResult, navigate: setActiveTab });

  useEffect(() => {
    if (!isAuth) return;
    if (autoLoadedRef.current[profile]) return;
    let saved: Record<string, string> | null = null;
    try {
      saved = JSON.parse(localStorage.getItem("resultLoadedAt") || "{}");
    } catch (err) { reportError("app/App.tsx:307", err); }
    if (saved && saved[profile]) {
      autoLoadedRef.current[profile] = true;
      loadRecommendations();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuth, profile]);

  // Показываем экран техработ, если backend недоступен (пересборка/рестарт).
  // Один упавший poll - ещё не даун: тяжёлый запрос (прогнозы) может на
  // секунды занять event loop. Maintenance только после 3 фейлов подряд.
  useEffect(() => {
    let cancelled = false;
    let fails = 0;
    const check = async () => {
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 5000);
        try {
          await request<{ status: string }>("/api/health", { signal: ctrl.signal });
          fails = 0;
        } finally {
          clearTimeout(t);
        }
      } catch {
        fails += 1;
      }
      if (!cancelled) setBackendDown(fails >= 3);
    };
    check();
    const id = setInterval(check, 8000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // navigate-analysis event from StudentDashboard
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.profile) {
        handleProfileChange(detail.profile);
        setActiveTab("data");
        request<any>(`/api/results/recommendations/${detail.profile}`).catch(() => null).then(d => { if (d) { setLastResult(d); } }).catch((err) => reportError("app/App.tsx", err));
      }
    };
    window.addEventListener("navigate-analysis", handler);
    return () => window.removeEventListener("navigate-analysis", handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function showStatus(type: "success" | "error" | "info", message: string) {
    setStatus({ type, message });
    // Один таймер на все статусы: иначе прошлый setTimeout гасит новый статус
    // через остаток своих 5 с.
    if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
    statusTimerRef.current = setTimeout(() => {
      setStatus({ type: null, message: "" });
      statusTimerRef.current = null;
    }, 5000);
  }

  async function apiCall(endpoint: string, method = "GET", body: any = null) {
    try {
      setLoading(true);
      showStatus("info", "Выполнение...");
      const data = await request<any>(`${API}${endpoint}`, {
        method,
        ...(body ? { body } : {}),
      });
      setLastResult(data);
      showStatus("success", "✓ Готово");
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
        // request() уже отдаёт распарсенный JSON: раньше здесь вызывался
        // r.ok / r.json() и st всегда был null - проверка готовности не работала.
        const st = await request<any>("/api/pipeline/status").catch(() => null);
        return !!st?.recommendations_all_ready;
      } catch {
        return false;
      }
    };
    try {
      const data = await request<{ output?: string }>("/api/pipeline/gap-analysis", {
        method: "POST",
      });
      const m = String(data.output || "").match(/Task ID: (\S+?)\.?\s/);
      if (!m) throw new Error("Не получен ID задачи");
      const taskId = m[1];
      let statusMisses = 0;
      gapCancelRef.current = false;
      for (;;) {
        await new Promise((res) => setTimeout(res, 3000));
        if (gapCancelRef.current) return;
        let s: any = null;
        try {
          s = await request<any>(`/api/pipeline/task/${taskId}`).catch(() => null);
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
      setToast(`Ошибка gap-анализа: ${e?.message || e}`);
    } finally {
      if (!gapCancelRef.current) setGapRunning(false);
    }
  }

  function loadProfileDetail() {
    apiCall(`/profiles/${profile}`);
  }

  async function loadRecommendations() {
    const data = await apiCall(`/results/recommendations/${profile}`);
    if (data) {
      const hasResults = !!(data.recommendations || data.closest_roles);
      const notFound = typeof data.message === "string" && data.message.includes("не найдены");
      if (hasResults && !notFound && typeof data.generated_at === "string") {
        const stamp = data.generated_at as string;
        setResultLoadedAt((prev) => {
          const next = { ...prev, [profile]: stamp };
          try {
            localStorage.setItem("resultLoadedAt", JSON.stringify(next));
          } catch (err) { reportError("app/App.tsx:648", err); }
          return next;
        });
      }
    }
  }

  async function handleDownloadPdf() {
    try {
      const { blob, filename } = await download(`/api/results/report/${profile}`);
      saveBlob(blob, filename ?? `report_${profile}_${new Date().toISOString().split("T")[0]}.pdf`);
    } catch (error) {
      console.error("Failed to download PDF report:", error);
    }
  }

  function handlePrintReport() {
    // Вариант B (frontend): печать текущей вкладки «Данные» в PDF через браузер.
    window.print();
  }

  function loadMarket() {
    apiCall("/market-competencies");
  }

  // Список профессий таксономии - для выбора цели сравнения на вкладке «Данные».
  useEffect(() => {
    if (activeTab !== "data" || !isAuth || professionsList.length > 0) return;
    // request() уже возвращает распарсенный JSON (lib/api.ts): раньше здесь
    // было r.ok ? r.json() - r.ok был undefined и список профессий не грузился.
    request<any>(`${API}/taxonomy/professions`).catch(() => null)
      .then((d) => {
        const names = ((d?.professions || []) as any[])
          .map((p) => p?.name)
          .filter(Boolean);
        if (names.length > 0) {
          setProfessionsList(names);
          setTargetProfession((prev) => prev || names[0]);
        }
      })
      .catch((err) => reportError("app/App.tsx", err));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, isAuth]);

  function compareWithProfession() {
    if (!targetProfession) return;
    apiCall(`/profiles/${profile}/profession-evaluation?profession=${encodeURIComponent(targetProfession)}`);
  }

  function loadSummary() {
    apiCall("/results/summary");
  }

  // Выход: убиваем сессию на бэкенде, чистим предпросмотр роли, разлогиниваемся.
  const handleLogout = () => {
    request<void>("/api/auth/logout", { method: "POST" }).catch((err) => reportError("app/App.tsx:logout", err));
    try { localStorage.removeItem("rolePreview"); } catch (err) { reportError("app/App.tsx:785", err); }
    setRolePreview(null);
    logout();
  };

  if (backendDown) {
    return <MaintenanceScreen />;
  }

  if (!isAuth) {
    return <LoginPage onLogin={login} />;
  }

  const isStudent = effectiveRole === "student";

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      {toast && (
        <div className="fixed top-4 right-4 z-[60] max-w-sm rounded-lg border border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950 px-4 py-3 text-sm text-red-800 dark:text-red-200 shadow-sm flex items-start gap-2">
          <span className="flex-1">{toast}</span>
          <button
            type="button"
            onClick={() => setToast(null)}
            className="text-red-500 hover:text-red-700 dark:hover:text-red-300"
            aria-label="Закрыть"
          >
            ×
          </button>
        </div>
      )}
      <AppHeader
        name={name}
        role={role}
        canPreview={canPreview}
        rolePreview={rolePreview}
        onPreviewChange={setPreview}
        previewActive={previewActive}
        onResetPreview={() => setPreview(null)}
        theme={theme}
        onToggleTheme={toggleTheme}
        onOpenProfile={() => setActiveTab("profile")}
        onLogout={handleLogout}
      />

      {/* Main Content */}
      <main className="flex-1 w-full max-w-[1760px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
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
                    ? "bg-green-50 border-green-200 text-green-800 dark:bg-green-950/40 dark:border-green-800 dark:text-green-200"
                    : status.type === "error"
                      ? "bg-red-50 border-red-200 text-red-800 dark:bg-red-950/40 dark:border-red-800 dark:text-red-200"
                      : "bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-950/40 dark:border-blue-800 dark:text-blue-200"
                }`}
              >
                {status.message}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <AppNavTabs activeTab={activeTab} effectiveRole={effectiveRole} onSelect={setActiveTab} />

          {/* Pipeline progress */}
          {pipelineStep && (
            <div className="mb-4">
              <MemoPipelineProgress currentStep={pipelineStep} onCancel={cancelPipeline} onRestart={restartPipeline} showLogs={activeTab === "admin"} />
            </div>
          )}

          {/* Vacancies Tab */}
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Загрузка раздела…</div>}>
          <TabsContent value="vacancies">
            <ErrorBoundary scope="vacancies">
              <VacanciesList
                pipelineStep={pipelineStep}
                pipelineLoading={pipelineLoading}
                restartFlag={restartFlag}
                onStartPipeline={effectiveRole === "student" ? undefined : (regionIds, profession, maxPages, periodDays) => startPipeline(regionIds, profession, maxPages, periodDays)}
                pipelineMaxPages={pipelineMaxPages}
                pipelinePeriod={pipelinePeriod}
                canRunPipeline={effectiveRole !== "student"}
              />
            </ErrorBoundary>
          </TabsContent>

          {/* Data Tab */}
          <TabsContent value="data">
            <ErrorBoundary scope="data">
              <DataTab
                profile={profile}
                profilesList={profilesList}
                onProfileChange={handleProfileChange}
                loading={loading}
                isStudent={isStudent}
                gapRunning={gapRunning}
                gapMsg={gapMsg}
                onLoadRecommendations={loadRecommendations}
                onLoadProfileDetail={loadProfileDetail}
                onLoadMarket={loadMarket}
                onLoadSummary={loadSummary}
                onRunGapAnalysis={runGapAnalysis}
                professionsList={professionsList}
                targetProfession={targetProfession}
                onTargetProfessionChange={setTargetProfession}
                onCompareWithProfession={compareWithProfession}
                resultLoadedAt={resultLoadedAt}
                lastResult={lastResult}
                onDownloadPdf={handleDownloadPdf}
                onPrintReport={handlePrintReport}
              />
            </ErrorBoundary>
          </TabsContent>

          {/* Visualization Tab */}
          {isTabAllowedForRole("visualization", effectiveRole) && (
            <TabsContent value="visualization">
              <ErrorBoundary scope="visualization">
                <Suspense fallback={<TabSpinner />}>
                  <GapAnalysisVisualizer profile={profile} onProfileChange={handleProfileChange} />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
          )}

          <TabsContent value="predictions">
            <ErrorBoundary scope="predictions">
              <MemoPredictionsTab />
            </ErrorBoundary>
          </TabsContent>
          <TabsContent value="articles">
            <ErrorBoundary scope="articles">
              <Suspense fallback={<TabSkeleton />}>
                <ArticlesPage onStartGapAnalysis={() => { runGapAnalysis(); setActiveTab("data"); }} />
              </Suspense>
            </ErrorBoundary>
          </TabsContent>
          <TabsContent value="scientific-trends">
            <ErrorBoundary scope="scientific-trends">
              <MemoScientificTrendsTab />
              <div className="mt-6">
                <ArticlesPanel />
              </div>
            </ErrorBoundary>
          </TabsContent>
          <TabsContent value="publications">
            <ErrorBoundary scope="publications">
              <Suspense fallback={<TabSpinner />}>
                <PublicationsTab />
              </Suspense>
            </ErrorBoundary>
          </TabsContent>
          <TabsContent value="help">
            <ErrorBoundary scope="help">
              <Suspense fallback={<TabSpinner />}>
                <FaqPage />
              </Suspense>
            </ErrorBoundary>
          </TabsContent>
          {isTabAllowedForRole("taxonomy", effectiveRole) && (
            <TabsContent value="taxonomy">
              <ErrorBoundary scope="taxonomy">
                <Suspense fallback={<TabSpinner />}>
                  <TaxonomyBrowser showSuggest />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
          )}
          {isTabAllowedForRole("monitoring", effectiveRole) && (
            <TabsContent value="monitoring">
              <ErrorBoundary scope="monitoring">
                <Suspense fallback={<TabSpinner />}>
                  <MonitoringTab />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
          )}
          {isTabAllowedForRole("logs", effectiveRole) && (
            <TabsContent value="logs">
              <ErrorBoundary scope="logs">
                <Suspense fallback={<TabSpinner />}>
                  <LogsTab />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
          )}
          {isTabAllowedForRole("admin", effectiveRole) && (
            <TabsContent value="admin">
              <ErrorBoundary scope="admin">
                <Suspense fallback={<TabSpinner />}>
                  <AdminDashboard />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
          )}
          {isTabAllowedForRole("teacher", effectiveRole) && (
            <TabsContent value="teacher">
              <ErrorBoundary scope="teacher">
                <Suspense fallback={<TabSpinner />}>
                  <TeacherDashboard />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
          )}
          {isTabAllowedForRole("students", effectiveRole) && (
            <TabsContent value="students">
              <ErrorBoundary scope="students">
                <Suspense fallback={<TabSpinner />}>
                  <StudentsTab />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
          )}
          {isTabAllowedForRole("student", effectiveRole) && (
            <TabsContent value="student">
              <ErrorBoundary scope="student">
                <Suspense fallback={<TabSpinner />}>
                  <StudentDashboard onNavigate={setActiveTab} />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
          )}
          <TabsContent value="profile">
            <ErrorBoundary scope="profile">
              <Suspense fallback={<TabSpinner />}>
                {effectiveRole === "admin" ? (
                  <AdminProfilePage displayName={name ?? undefined} email={username ?? undefined} onNavigate={setActiveTab} />
                ) : effectiveRole === "teacher" || effectiveRole === "rop" ? (
                  <TeacherProfilePage displayName={name ?? undefined} email={username ?? undefined} onNavigate={setActiveTab} />
                ) : (
                  <StudentProfilePage displayName={name ?? undefined} email={username ?? undefined} onNavigate={setActiveTab} />
                )}
              </Suspense>
            </ErrorBoundary>
          </TabsContent>
          </Suspense>
        </Tabs>

        <div className="mt-auto pt-12">
          <Footer />
        </div>
      </main>
    </div>
  );
}
