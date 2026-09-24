import { useState, useEffect, useRef } from "react";
import { Label } from "./components/ui/label";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
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
import { Tabs, TabsContent, TabsList } from "./components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./components/ui/dropdown-menu";
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
import { useTheme } from "../lib/theme";
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
  Moon,
  Sun,
  Shield,
  GraduationCap,
  UserCheck,
  History,
  Activity,
  HelpCircle,
  ChevronDown,
  FolderOpen,
  LineChart,
} from "lucide-react";

const API = "/api";

type NavItem = { value: string; label: string; Icon: any };

function NavGroup({
  title,
  items,
  activeTab,
  onSelect,
}: {
  title: string;
  items: NavItem[];
  activeTab: string;
  onSelect: (v: string) => void;
}) {
  if (items.length === 0) return null;
  const active = items.find((i) => i.value === activeTab);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className={`inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all cursor-pointer ${
            active
              ? "bg-white text-gray-900 shadow-sm dark:bg-slate-800 dark:text-slate-100"
              : "text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
          }`}
        >
          {active ? <active.Icon className="size-4" /> : null}
          {active ? active.label : title}
          <ChevronDown className="size-3.5 opacity-60" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-52">
        {items.map(({ value, label, Icon }) => (
          <DropdownMenuItem
            key={value}
            onSelect={() => onSelect(value)}
            className={`gap-2 cursor-pointer ${value === activeTab ? "font-semibold text-blue-700 dark:text-blue-300" : ""}`}
          >
            <Icon className="size-4" />
            {label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

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
  useEffect(() => { initApiLogger(); loadProfiles(); }, []);
  const [backendDown, setBackendDown] = useState(false);
  const [profile, setProfile] = useState("base");
  const [profilesList, setProfilesList] = useState<string[]>(["base", "dc", "top_dc"]);
  const [cpName, setCpName] = useState("");
  const [cpLevel, setCpLevel] = useState("middle");
  const [cpCodes, setCpCodes] = useState("");
  const [cpSkills, setCpSkills] = useState("");
  const [cpMsg, setCpMsg] = useState("");
  const [cpSaving, setCpSaving] = useState(false);
  const [cpOpen, setCpOpen] = useState(false);

  const parseList = (s: string) =>
    s.split(/[,\n;]+/).map((x) => x.trim()).filter(Boolean);

  async function loadProfiles(select?: string) {
    try {
      const r = await fetch(`${API}/profiles`);
      if (!r.ok) return;
      const d = await r.json();
      const names = Array.isArray(d.profiles) && d.profiles.length ? d.profiles : ["base", "dc", "top_dc"];
      setProfilesList(names);
      if (select && names.includes(select)) setProfile(select);
      else if (!names.includes(profile)) setProfile(names[0]);
    } catch { /* keep hardcoded fallback */ }
  }

  async function createCustomProfile() {
    setCpMsg("");
    const name = cpName.trim().toLowerCase();
    if (!/^[a-z0-9_]{2,32}$/.test(name)) {
      setCpMsg("Имя: латиница/цифры/_, 2-32 символа");
      return;
    }
    setCpSaving(true);
    try {
      const r = await fetch(`${API}/profiles/custom`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          target_level: cpLevel,
          competencies: parseList(cpCodes),
          skills: parseList(cpSkills),
        }),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((d as any).detail || r.statusText);
      setCpMsg(`Профиль «${name}» создан: компетенций ${d.competencies_count}, навыков ${d.skills_count}`);
      setCpName(""); setCpCodes(""); setCpSkills("");
      await loadProfiles(name);
    } catch (e: any) {
      setCpMsg("Ошибка: " + e.message);
    } finally {
      setCpSaving(false);
    }
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
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("vacancies");
  const [professionsList, setProfessionsList] = useState<string[]>([]);
  const [targetProfession, setTargetProfession] = useState("");

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
  const { theme, toggle: toggleTheme } = useTheme();
  const roleRef = useRef(role);
  useEffect(() => { roleRef.current = role; }, [role]);

  // Дата подгрузки переживает перезагрузку (localStorage), а сами результаты – нет.
  // Если штамп есть, а результата в памяти нет – подтягиваем автоматически.
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
  // Один упавший poll – ещё не даун: тяжёлый запрос (прогнозы) может на
  // секунды занять event loop. Maintenance только после 3 фейлов подряд.
  useEffect(() => {
    let cancelled = false;
    let fails = 0;
    const check = async () => {
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 5000);
        const res = await fetch("/api/health", { signal: ctrl.signal });
        clearTimeout(t);
        fails = res.ok ? 0 : fails + 1;
      } catch {
        fails += 1;
      }
      if (!cancelled) setBackendDown(fails >= 3);
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

  // Список профессий таксономии — для выбора цели сравнения на вкладке «Данные».
  useEffect(() => {
    if (activeTab !== "data" || !isAuth || professionsList.length > 0) return;
    fetch(`${API}/taxonomy/professions`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        const names = ((d?.professions || []) as any[])
          .map((p) => p?.name)
          .filter(Boolean);
        if (names.length > 0) {
          setProfessionsList(names);
          setTargetProfession((prev) => prev || names[0]);
        }
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, isAuth]);

  function compareWithProfession() {
    if (!targetProfession) return;
    apiCall(`/profiles/${profile}/profession-evaluation?profession=${encodeURIComponent(targetProfession)}`);
  }

  function loadSummary() {
    apiCall("/results/summary");
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
    <div className="min-h-screen bg-white text-gray-900 dark:bg-slate-950 dark:text-slate-100">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white dark:border-slate-800 dark:bg-slate-950">
        <div className="max-w-[1760px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center justify-center w-12 h-12 bg-blue-600 rounded-xl">
                <TrendingUp className="size-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-slate-100">
                  Competency Gap Analyzer
                </h1>
                <p className="text-sm text-gray-600 dark:text-slate-400">
                  AI-powered competency analysis platform
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-slate-400">
                {roleIcon}
                <span>{name || roleLabel}</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 dark:bg-slate-800 dark:text-slate-200">{roleLabel}</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleTheme}
                title={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
                className="text-gray-500 hover:text-gray-900 dark:text-slate-400 dark:hover:text-slate-100"
              >
                {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => { fetch("/api/auth/logout", { method: "POST", headers: authHeaders() }).catch(() => {}); logout(); }} className="text-gray-500 dark:text-slate-400 hover:text-red-600">
                <LogOut className="size-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1760px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
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
          <TabsList className="inline-flex h-12 items-center justify-center gap-1 rounded-lg bg-gray-100 p-1 dark:bg-slate-900">
            <NavGroup
              title="Работа"
              activeTab={activeTab}
              onSelect={setActiveTab}
              items={[
                { value: "vacancies", label: "Вакансии", Icon: Briefcase },
                { value: "data", label: "Результаты", Icon: Database },
                ...(role !== "teacher"
                  ? [{ value: "visualization", label: "Визуализация", Icon: BarChart3 }]
                  : []),
              ]}
            />
            <NavGroup
              title="Анализ"
              activeTab={activeTab}
              onSelect={setActiveTab}
              items={[
                { value: "predictions", label: "Прогнозы", Icon: TrendingUp },
                { value: "articles", label: "Аналитика рынка", Icon: LineChart },
                { value: "scientific-trends", label: "Научные тренды", Icon: FolderOpen },
                ...(role === "teacher" || role === "rop"
                  ? [{ value: "teacher", label: "Преподавательский анализ", Icon: BarChart3 }]
                  : []),
              ]}
            />
            <NavGroup
              title="Система"
              activeTab={activeTab}
              onSelect={setActiveTab}
              items={[
                ...(role === "admin"
                  ? [
                      { value: "monitoring", label: "Мониторинг", Icon: Activity },
                      { value: "logs", label: "Логи", Icon: FileText },
                      { value: "admin", label: "Админ", Icon: Shield },
                    ]
                  : []),
                ...(role === "student"
                  ? [{ value: "student", label: "Мои запросы", Icon: History }]
                  : []),
                { value: "help", label: "Помощь", Icon: HelpCircle },
              ]}
            />
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
            <Card className="border border-gray-200 dark:border-slate-700 shadow-sm overflow-hidden">
              <CardHeader className="border-b border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-900">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 bg-emerald-600 rounded-lg">
                    <Database className="size-5 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-xl font-semibold text-gray-900 dark:text-slate-100">
                      Результаты
                    </CardTitle>
                    <CardDescription className="text-sm text-gray-600 dark:text-slate-400">
                      Просмотр профилей, рекомендаций и статистики
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-6 space-y-6">
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-gray-900 dark:text-slate-100">
                    Профиль компетенций
                  </Label>
                  <Select value={profile} onValueChange={handleProfileChange}>
                    <SelectTrigger className="h-11 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {profilesList.map((p) => (
                        <SelectItem key={p} value={p}>
                          <div className="flex items-center gap-2">
                            <Award className={`size-4 ${p === profile ? "text-emerald-600" : "text-gray-400 dark:text-slate-500"}`} />
                            <span>{p === "base" ? "BASE (junior)" : p === "dc" ? "DATA SCIENTIST (middle)" : p === "top_dc" ? "TOP DATA SCIENTIST (senior)" : p}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                  <Button
                    onClick={loadRecommendations}
                    disabled={loading}
                    className="h-11 bg-blue-700 hover:bg-blue-800 text-white transition-colors focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:outline-none"
                  >
                    <Search className="mr-2 size-4" />
                    Показать сохранённые
                  </Button>
                  <Button
                    onClick={loadProfileDetail}
                    disabled={loading}
                    className="h-11 bg-emerald-600 hover:bg-emerald-700 text-white transition-colors focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:outline-none"
                  >
                    <FileText className="mr-2 size-4" />
                    Профиль
                  </Button>
                  <Button
                    onClick={loadRecommendations}
                    disabled={loading}
                    className="h-11 bg-blue-600 hover:bg-blue-700 text-white transition-colors focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:outline-none"
                  >
                    <Sparkles className="mr-2 size-4" />
                    Рекомендации
                  </Button>
                  <Button
                    onClick={loadMarket}
                    disabled={loading}
                    variant="outline"
                    className="h-11 border-gray-300 dark:border-slate-600 text-gray-700 dark:text-slate-300 hover:bg-gray-50 dark:hover:bg-slate-800 transition-colors focus-visible:ring-2 focus-visible:ring-gray-400 focus-visible:outline-none"
                  >
                    <BarChart3 className="mr-2 size-4" />
                    Рынок
                  </Button>
                  <Button
                    onClick={loadSummary}
                    disabled={loading}
                    variant="outline"
                    className="h-11 border-gray-300 dark:border-slate-600 text-gray-700 dark:text-slate-300 hover:bg-gray-50 dark:hover:bg-slate-800 transition-colors focus-visible:ring-2 focus-visible:ring-gray-400 focus-visible:outline-none"
                  >
                    <FileText className="mr-2 size-4" />
                    Сводка
                  </Button>
                  <Button
                    onClick={runGapAnalysis}
                    disabled={loading || gapRunning}
                    className="h-11 bg-amber-600 hover:bg-amber-700 text-white transition-colors focus-visible:ring-2 focus-visible:ring-amber-500 focus-visible:outline-none"
                  >
                    <Zap className="mr-2 size-4" />
                    Запустить gap-анализ
                  </Button>
                </div>
                {gapRunning && (
                  <p className="text-sm text-amber-700 dark:text-amber-300">{gapMsg || "Выполняется..."}</p>
                )}
                <div className="space-y-2 rounded-lg border border-gray-200 dark:border-slate-700 p-4">
                  <Label className="text-sm font-medium text-gray-900 dark:text-slate-100">
                    Целевая профессия для сравнения
                  </Label>
                  <div className="flex flex-col sm:flex-row gap-3">
                    <Select value={targetProfession} onValueChange={setTargetProfession}>
                      <SelectTrigger className="h-11 flex-1 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600">
                        <SelectValue placeholder="Выберите профессию..." />
                      </SelectTrigger>
                      <SelectContent>
                        {professionsList.map((p) => (
                          <SelectItem key={p} value={p}>
                            {p}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button
                      onClick={compareWithProfession}
                      disabled={loading || !targetProfession}
                      className="h-11 bg-violet-600 hover:bg-violet-700 text-white transition-colors focus-visible:ring-2 focus-visible:ring-violet-500 focus-visible:outline-none sm:w-auto w-full"
                    >
                      <Briefcase className="mr-2 size-4" />
                      Сравнить с профессией
                    </Button>
                  </div>
                </div>
                <p className="text-xs text-gray-500 dark:text-slate-400">
                  Последняя подгрузка результатов [{profile}]: {(() => {
                    const iso = resultLoadedAt[profile];
                    if (!iso) return "ещё не подгружались";
                    const d = new Date(iso);
                    return isNaN(d.getTime()) ? iso : d.toLocaleString("ru-RU");
                  })()}
                </p>
                {(() => {
                  const g = (lastResult as any)?.generated_at;
                  if (typeof g !== "string") return null;
                  const age = Date.now() - new Date(g).getTime();
                  if (isNaN(age) || age < 7 * 864e5) return null;
                  const days = Math.floor(age / 864e5);
                  return <span className="ml-2 px-2 py-0.5 rounded bg-amber-100 dark:bg-amber-950/30 text-amber-800 dark:text-amber-200">данные устарели ({days} дн.)</span>;
                })()}

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
                    return (
                      <>
                        {(d as any).focus_mode && (d as any).target_profession && (
                          <Card className="border-2 border-violet-200 dark:border-violet-800 bg-gradient-to-br from-violet-50 to-indigo-50 dark:from-violet-950/20 dark:to-indigo-950/20 mb-4">
                            <CardContent className="pt-5">
                              <div className="flex items-center gap-3 mb-3">
                                <div className="p-2 bg-violet-600 rounded-lg">
                                  <Briefcase className="size-5 text-white" />
                                </div>
                                <div>
                                  <div className="font-bold text-gray-900 dark:text-slate-100">
                                    Фокус: {(d as any).target_profession} · профиль {String((d as any).profile || "")}
                                  </div>
                                  <div className="text-xs text-gray-500 dark:text-slate-400">
                                    {((d as any).target_domains || []).join(", ")}
                                  </div>
                                </div>
                              </div>
                              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                                <div className="rounded-lg bg-white/70 dark:bg-slate-950/40 p-3">
                                  <div className="text-2xl font-bold text-violet-700 dark:text-violet-300">{Number((d as any).profession_coverage || 0).toFixed(1)}%</div>
                                  <div className="text-xs text-gray-500 dark:text-slate-400">покрытие профессии</div>
                                </div>
                                <div className="rounded-lg bg-white/70 dark:bg-slate-950/40 p-3">
                                  <div className="text-2xl font-bold text-emerald-700 dark:text-emerald-300">{Number((d as any).skill_coverage || 0).toFixed(1)}%</div>
                                  <div className="text-xs text-gray-500 dark:text-slate-400">навыки: {(d as any).skill_strict_has ?? "–"} из {(d as any).skill_strict_total ?? "–"}</div>
                                </div>
                                <div className="rounded-lg bg-white/70 dark:bg-slate-950/40 p-3">
                                  <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">{Number((d as any).readiness_score || 0).toFixed(1)}%</div>
                                  <div className="text-xs text-gray-500 dark:text-slate-400">готовность</div>
                                </div>
                                <div className="rounded-lg bg-white/70 dark:bg-slate-950/40 p-3">
                                  <div className="text-2xl font-bold text-slate-700 dark:text-slate-200">{Number((d as any).domain_coverage_score || 0).toFixed(1)}%</div>
                                  <div className="text-xs text-gray-500 dark:text-slate-400">покрытие доменов</div>
                                </div>
                              </div>
                              {(d as any).krm_note && (
                                <p className="mt-3 text-xs text-amber-700 dark:text-amber-300">{String((d as any).krm_note)}</p>
                              )}
                            </CardContent>
                          </Card>
                        )}
                        <RecommendationsReport data={lastResult as any} />
                      </>
                    );
                  }
                  if (d.evaluations && Array.isArray(d.profiles)) {
                    return <SummaryReport data={lastResult as any} />;
                  }
                  const msg = d.message as string | undefined;
                  if (msg && (msg.includes("не найдены") || msg.includes("not found"))) {
                    return (
                      <Card className="border-2 border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/30">
                        <CardContent className="pt-6 text-center py-12">
                          <AlertCircle className="size-12 text-amber-400 mx-auto mb-4" />
                          <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-200 mb-2">{msg}</h3>
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
                            <p className="text-sm text-amber-700 dark:text-amber-300 mt-3">{gapMsg || "Выполняется..."}</p>
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
            <ArticlesPage onStartGapAnalysis={() => { runGapAnalysis(); setActiveTab("data"); }} />
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
