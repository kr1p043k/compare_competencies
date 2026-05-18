import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Badge } from "./components/ui/badge";
import { RegionCombobox } from "./components/RegionCombobox";
import { GapAnalysisVisualizer } from "./components/GapAnalysisVisualizer";
import { StatsCards } from "./components/StatsCards";
import { Footer } from "./components/Footer";
import { VacanciesList } from "./components/VacanciesList";
import { PipelineProgress } from "./components/PipelineProgress";
import { motion } from "motion/react";
import {
  Database,
  PlayCircle,
  TrendingUp,
  Brain,
  CheckCircle2,
  Activity,
  Download,
  FileText,
  Sparkles,
  RefreshCw,
  Search,
  MapPin,
  Calendar,
  FileBarChart,
  Loader2,
  BarChart3,
  Zap,
  Target,
  Award,
  Briefcase,
} from "lucide-react";

const REGIONS = [
  { id: 0, name: "Вся Россия" },
  { id: 1, name: "Москва" },
  { id: 2, name: "Санкт-Петербург" },
  { id: 3, name: "Екатеринбург" },
  { id: 4, name: "Новосибирск" },
  { id: 5, name: "Нижний Новгород" },
  { id: 6, name: "Казань" },
  { id: 7, name: "Самара" },
  { id: 8, name: "Ростов-на-Дону" },
  { id: 9, name: "Уфа" },
  { id: 10, name: "Красноярск" },
  { id: 11, name: "Пермь" },
  { id: 12, name: "Воронеж" },
  { id: 13, name: "Волгоград" },
  { id: 14, name: "Краснодар" },
  { id: 15, name: "Саратов" },
  { id: 16, name: "Тюмень" },
  { id: 17, name: "Тольятти" },
  { id: 18, name: "Ижевск" },
  { id: 19, name: "Барнаул" },
  { id: 20, name: "Иркутск" },
  { id: 21, name: "Ульяновск" },
  { id: 22, name: "Хабаровск" },
  { id: 23, name: "Владивосток" },
  { id: 24, name: "Ярославль" },
  { id: 25, name: "Махачкала" },
  { id: 26, name: "Томск" },
  { id: 27, name: "Оренбург" },
  { id: 28, name: "Кемерово" },
  { id: 29, name: "Новокузнецк" },
  { id: 30, name: "Рязань" },
];

const API = "/api";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: "spring",
      stiffness: 100,
    },
  },
};

export default function App() {
  const [regions, setRegions] = useState("0");
  const [profile, setProfile] = useState("base");
  const [status, setStatus] = useState<{
    type: "success" | "error" | "info" | null;
    message: string;
  }>({ type: null, message: "" });
  const [loading, setLoading] = useState(false);
  const [pipelineStep, setPipelineStep] = useState<{
    step: number;
    total: number;
    status: "running" | "success" | "error" | "completed";
    message: string;
    progress: number;
  } | null>(null);

  function showStatus(type: "success" | "error" | "info", message: string) {
    setStatus({ type, message });
    setTimeout(() => setStatus({ type: null, message: "" }), 5000);
  }

  useEffect(() => {
    async function loadStats() {
      try {
        // Загружаем статистику по вакансиям
        const vacanciesRes = await fetch(`${API}/vacancies/stats/summary`);
        let totalVacancies = undefined;

        if (vacanciesRes.ok) {
          try {
            const vacanciesData = await vacanciesRes.json();
            totalVacancies = vacanciesData.total || 0;
          } catch {
            // Ignore JSON parse errors
          }
        }

        // Загружаем результаты анализа (если есть)
        const summaryRes = await fetch(`${API}/results/summary`);
        let coverage, recommendations, accuracy;

        if (summaryRes.ok) {
          try {
            const summaryData = await summaryRes.json();
            if (summaryData.profiles?.base) {
              coverage = summaryData.profiles.base.market_coverage_score
                ? Math.round(summaryData.profiles.base.market_coverage_score * 100)
                : undefined;
              recommendations = summaryData.profiles.base.recommendations_count || undefined;
              accuracy = summaryData.profiles.base.ml_accuracy
                ? Math.round(summaryData.profiles.base.ml_accuracy * 100)
                : undefined;
            }
          } catch {
            // Ignore JSON parse errors
          }
        }

        setStats({
          totalVacancies,
          coverage,
          recommendations,
          accuracy,
        });
      } catch (e) {
        // API недоступен - это нормально при первом запуске
        // Просто не показываем статистику
        console.log("API not available, stats will show placeholders");
      }
    }
    loadStats();
  }, []);

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

  function runFullCycle() {
    const params = new URLSearchParams({
      query,
      area_id: area,
      max_pages: pages.toString(),
      period: period.toString(),
    });
    apiCall(`/pipeline/full-cycle?${params}`, "POST");
  }

  function step1Collect() {
    const params = new URLSearchParams({
      query,
      area_id: area,
      max_pages: pages.toString(),
      period: period.toString(),
    });
    apiCall(`/pipeline/step1-collect?${params}`, "POST");
  }

  function step2Clusters() {
    apiCall("/pipeline/step2-train-clusters", "POST");
  }

  function step3Model() {
    apiCall("/pipeline/step3-train-model", "POST");
  }

  function step4Gap() {
    apiCall("/pipeline/step4-gap-analysis", "POST");
  }

  function loadProfileDetail() {
    apiCall(`/profiles/${profile}`);
  }

  function loadRecommendations() {
    apiCall(`/results/recommendations/${profile}`);
  }

  function loadMarket() {
    apiCall("/market-competencies");
  }

  function loadSummary() {
    apiCall("/results/summary");
  }

  function loadHealth() {
    apiCall("/health");
  }

  function checkClusters() {
    apiCall("/scripts/check-clusters", "POST");
  }

  function extendSkills() {
    apiCall("/scripts/extend-skills?auto_confirm=true", "POST");
  }

  function fullRebuild() {
    if (confirm("Это удалит все кэши и пересоберёт проект. Продолжить?")) {
      apiCall("/scripts/full-rebuild", "POST");
    }
  }

  async function runPipelineWithProgress() {
    try {
      setLoading(true);
      setPipelineStep(null);

      const params = new URLSearchParams({
        query,
        area_id: area,
        max_pages: pages.toString(),
        period: period.toString(),
      });

      const response = await fetch(
        `${API}/pipeline/full-cycle?${params}`,
        {
          method: "POST",
          headers: {
            Accept: "text/event-stream",
          },
        }
      );

      if (!response.ok || !response.body) {
        showStatus("error", "Не удалось запустить пайплайн");
        setLoading(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.slice(6);
            try {
              const data = JSON.parse(jsonStr);
              setPipelineStep(data);

              if (data.status === "completed") {
                setHasRunBefore(true);
                showStatus("success", "Пайплайн завершен!");
                // Перезагрузить статистику
                setTimeout(() => window.location.reload(), 2000);
              } else if (data.status === "error") {
                showStatus("error", "Ошибка выполнения пайплайна");
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
    } catch (e: any) {
      showStatus("error", `Ошибка: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function quickRefresh() {
    try {
      setLoading(true);
      setPipelineStep(null);

      const response = await fetch(`${API}/pipeline/quick-refresh`, {
        method: "POST",
        headers: {
          Accept: "text/event-stream",
        },
      });

      if (!response.ok || !response.body) {
        showStatus("error", "Не удалось запустить обновление");
        setLoading(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.slice(6);
            try {
              const data = JSON.parse(jsonStr);
              setPipelineStep(data);

              if (data.status === "completed") {
                showStatus("success", "Обновление завершено!");
                setTimeout(() => window.location.reload(), 2000);
              } else if (data.status === "error") {
                showStatus("error", "Ошибка обновления");
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
    } catch (e: any) {
      showStatus("error", `Ошибка: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-950 dark:via-slate-900 dark:to-indigo-950 relative overflow-hidden">
      {/* Animated background gradient orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute -top-1/2 -left-1/4 w-[800px] h-[800px] bg-gradient-to-br from-blue-400/20 to-purple-500/20 rounded-full blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, 50, 0],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute -bottom-1/2 -right-1/4 w-[800px] h-[800px] bg-gradient-to-tl from-indigo-400/20 to-pink-500/20 rounded-full blur-3xl"
          animate={{
            x: [0, -100, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </div>

      <motion.div
        className="relative z-10 max-w-[1400px] mx-auto px-6 py-12"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Header */}
        <motion.div variants={itemVariants} className="text-center mb-12">
          <motion.div
            className="inline-flex items-center justify-center gap-4 mb-6"
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 400 }}
          >
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl blur-xl opacity-50 animate-pulse" />
              <div className="relative bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-4 rounded-2xl shadow-2xl">
                <Brain className="size-12 text-white" />
              </div>
            </div>
            <div className="text-left">
              <h1 className="text-5xl font-black bg-gradient-to-r from-slate-900 via-blue-800 to-purple-900 dark:from-white dark:via-blue-200 dark:to-purple-200 bg-clip-text text-transparent tracking-tight">
                Competency Gap Analyzer
              </h1>
              <p className="text-lg text-slate-600 dark:text-slate-400 font-medium mt-1">
                AI-powered competency analysis platform
              </p>
            </div>
          </motion.div>
          <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Анализируйте соответствие учебных компетенций требованиям рынка
            труда с помощью передовых ML-алгоритмов и визуализации данных
          </p>
        </motion.div>

        {/* Status Badge */}
        {status.type && (
          <motion.div
            className="flex justify-center mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Badge
              variant={
                status.type === "success"
                  ? "default"
                  : status.type === "error"
                    ? "destructive"
                    : "secondary"
              }
              className="py-2.5 px-6 text-sm font-semibold shadow-lg backdrop-blur-sm"
            >
              {status.message}
            </Badge>
          </motion.div>
        )}

        {/* Stats Cards */}
        <motion.div variants={itemVariants}>
          <StatsCards stats={stats} />
        </motion.div>

        <Tabs defaultValue="vacancies" className="space-y-8">
          <motion.div variants={itemVariants}>
            <TabsList className="grid w-full grid-cols-5 h-14 bg-white/50 dark:bg-slate-900/50 backdrop-blur-xl border border-slate-200/50 dark:border-slate-700/50 shadow-lg p-1.5">
              <TabsTrigger
                value="vacancies"
                className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white data-[state=active]:shadow-lg"
              >
                <Briefcase className="size-4" />
                <span className="hidden sm:inline">Вакансии</span>
              </TabsTrigger>
              <TabsTrigger
                value="data"
                className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white data-[state=active]:shadow-lg"
              >
                <Database className="size-4" />
                <span className="hidden sm:inline">Данные</span>
              </TabsTrigger>
              <TabsTrigger
                value="visualization"
                className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white data-[state=active]:shadow-lg"
              >
                <BarChart3 className="size-4" />
                <span className="hidden sm:inline">Визуализация</span>
              </TabsTrigger>
              <TabsTrigger
                value="pipeline"
                className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white data-[state=active]:shadow-lg"
              >
                <Zap className="size-4" />
                <span className="hidden sm:inline">Пайплайн</span>
              </TabsTrigger>
              <TabsTrigger
                value="scripts"
                className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white data-[state=active]:shadow-lg"
              >
                <Activity className="size-4" />
                <span className="hidden sm:inline">Скрипты</span>
              </TabsTrigger>
            </TabsList>
          </motion.div>

          {/* Vacancies Tab */}
          <TabsContent value="vacancies" className="space-y-6">
            <motion.div variants={itemVariants}>
              <VacanciesList />
            </motion.div>
          </TabsContent>

          {/* Pipeline Tab */}
          <TabsContent value="pipeline" className="space-y-6">
            {/* Progress Bar */}
            {pipelineStep && (
              <motion.div variants={itemVariants}>
                <PipelineProgress currentStep={pipelineStep} />
              </motion.div>
            )}

            <motion.div variants={itemVariants}>
              <Card className="border-0 shadow-2xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
                <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-br from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg">
                      <Zap className="size-5 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-2xl">
                        Пайплайн анализа
                      </CardTitle>
                      <CardDescription className="text-base mt-1">
                        Полный цикл сбора данных и ML-анализа
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-8 space-y-6">
                  {/* Main Action Button */}
                  <div className="space-y-4">
                    {!hasRunBefore ? (
                      <>
                        <div className="flex items-center gap-3 mb-4">
                          <div className="p-2 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg">
                            <Sparkles className="size-4 text-white" />
                          </div>
                          <div>
                            <h3 className="text-lg font-bold text-slate-800 dark:text-slate-200">
                              Первый запуск
                            </h3>
                            <p className="text-sm text-slate-600 dark:text-slate-400">
                              Сбор вакансий → Обучение кластеров → Обучение модели → GAP-анализ
                            </p>
                          </div>
                        </div>
                        <motion.div
                          whileHover={{ scale: 1.01 }}
                          whileTap={{ scale: 0.99 }}
                        >
                          <Button
                            onClick={runPipelineWithProgress}
                            disabled={loading}
                            className="w-full h-16 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-700 hover:via-purple-700 hover:to-pink-700 text-white font-bold text-lg shadow-2xl hover:shadow-blue-500/50 transition-all duration-300"
                            size="lg"
                          >
                            {loading ? (
                              <>
                                <Loader2 className="mr-3 size-6 animate-spin" />
                                Выполнение полного цикла...
                              </>
                            ) : (
                              <>
                                <PlayCircle className="mr-3 size-6" />
                                Запустить полный цикл анализа
                              </>
                            )}
                          </Button>
                        </motion.div>
                      </>
                    ) : (
                      <>
                        <div className="flex items-center gap-3 mb-4">
                          <div className="p-2 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-lg">
                            <RefreshCw className="size-4 text-white" />
                          </div>
                          <div>
                            <h3 className="text-lg font-bold text-slate-800 dark:text-slate-200">
                              Обновление данных
                            </h3>
                            <p className="text-sm text-slate-600 dark:text-slate-400">
                              Быстрое обновление без сбора новых вакансий
                            </p>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <motion.div
                            whileHover={{ scale: 1.01 }}
                            whileTap={{ scale: 0.99 }}
                          >
                            <Button
                              onClick={quickRefresh}
                              disabled={loading}
                              className="w-full h-14 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-bold shadow-xl"
                              size="lg"
                            >
                              {loading ? (
                                <>
                                  <Loader2 className="mr-2 size-5 animate-spin" />
                                  Обновление...
                                </>
                              ) : (
                                <>
                                  <RefreshCw className="mr-2 size-5" />
                                  Получить свежие данные
                                </>
                              )}
                            </Button>
                          </motion.div>
                          <motion.div
                            whileHover={{ scale: 1.01 }}
                            whileTap={{ scale: 0.99 }}
                          >
                            <Button
                              onClick={runPipelineWithProgress}
                              disabled={loading}
                              variant="outline"
                              className="w-full h-14 border-2 border-purple-500 hover:bg-purple-50 dark:hover:bg-purple-950/30 font-bold"
                              size="lg"
                            >
                              <PlayCircle className="mr-2 size-5" />
                              Полный цикл заново
                            </Button>
                          </motion.div>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Info Card */}
                  <div className="p-4 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                    <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
                      Что делает полный цикл?
                    </h4>
                    <ul className="space-y-1.5 text-sm text-blue-800 dark:text-blue-400">
                      <li className="flex items-start gap-2">
                        <span className="text-blue-600 mt-0.5">1.</span>
                        <span>Сбор актуальных IT-вакансий с hh.ru (регионы Москва, СПб)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-600 mt-0.5">2.</span>
                        <span>Обучение кластеров для группировки вакансий по ролям</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-600 mt-0.5">3.</span>
                        <span>Обучение ML-модели для ранжирования навыков (XGBoost)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-600 mt-0.5">4.</span>
                        <span>GAP-анализ компетенций и генерация рекомендаций</span>
                      </li>
                    </ul>
                    <p className="mt-3 text-xs text-blue-700 dark:text-blue-500">
                      ⏱️ Время выполнения: ~10-15 минут
                    </p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Data Tab */}
          <TabsContent value="data" className="space-y-6">
            <motion.div variants={itemVariants}>
              <Card className="border-0 shadow-2xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
                <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-br from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl shadow-lg">
                      <Database className="size-5 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-2xl">
                        Данные и результаты
                      </CardTitle>
                      <CardDescription className="text-base mt-1">
                        Просмотр профилей, рекомендаций и статистики
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-8 space-y-6">
                  <div className="space-y-3">
                    <Label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                      Профиль компетенций
                    </Label>
                    <Select value={profile} onValueChange={setProfile}>
                      <SelectTrigger className="h-12 border-2 shadow-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="base">
                          <div className="flex items-center gap-2">
                            <Award className="size-4 text-blue-500" />
                            <span>BASE (junior)</span>
                          </div>
                        </SelectItem>
                        <SelectItem value="dc">
                          <div className="flex items-center gap-2">
                            <Award className="size-4 text-purple-500" />
                            <span>DATA CENTER (middle)</span>
                          </div>
                        </SelectItem>
                        <SelectItem value="top_dc">
                          <div className="flex items-center gap-2">
                            <Award className="size-4 text-pink-500" />
                            <span>TOP DATA CENTER (senior)</span>
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                      <Button
                        onClick={loadProfileDetail}
                        disabled={loading}
                        className="w-full h-12 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 shadow-lg"
                      >
                        <FileText className="mr-2 size-4" />
                        Профиль + Gap
                      </Button>
                    </motion.div>
                    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                      <Button
                        onClick={loadRecommendations}
                        disabled={loading}
                        className="w-full h-12 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 shadow-lg"
                      >
                        <Sparkles className="mr-2 size-4" />
                        Рекомендации
                      </Button>
                    </motion.div>
                    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                      <Button
                        onClick={loadMarket}
                        disabled={loading}
                        variant="outline"
                        className="w-full h-12 border-2"
                      >
                        <TrendingUp className="mr-2 size-4" />
                        Рынок
                      </Button>
                    </motion.div>
                    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                      <Button
                        onClick={loadSummary}
                        disabled={loading}
                        variant="outline"
                        className="w-full h-12 border-2"
                      >
                        <FileBarChart className="mr-2 size-4" />
                        Сводка
                      </Button>
                    </motion.div>
                    <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                      <Button
                        onClick={loadHealth}
                        disabled={loading}
                        variant="outline"
                        className="w-full h-12 border-2"
                      >
                        <Activity className="mr-2 size-4" />
                        Проверка
                      </Button>
                    </motion.div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Visualization Tab - NEW */}
          <TabsContent value="visualization" className="space-y-6">
            <motion.div variants={itemVariants}>
              <GapAnalysisVisualizer profile={profile} />
            </motion.div>
          </TabsContent>

          {/* Scripts Tab */}
          <TabsContent value="scripts" className="space-y-6">
            <motion.div variants={itemVariants}>
              <Card className="border-0 shadow-2xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
                <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-br from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl shadow-lg">
                      <Activity className="size-5 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-2xl">
                        Служебные скрипты
                      </CardTitle>
                      <CardDescription className="text-base mt-1">
                        Инструменты для обслуживания и отладки системы
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-8 space-y-3">
                  <motion.div whileHover={{ x: 4 }}>
                    <Button
                      onClick={checkClusters}
                      disabled={loading}
                      variant="outline"
                      className="w-full justify-start h-14 border-2 hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-950/30"
                    >
                      <Search className="mr-3 size-5 text-blue-600" />
                      <div className="text-left">
                        <div className="font-semibold">Проверить кластеры</div>
                        <div className="text-xs text-muted-foreground">
                          Анализ обученных кластеров вакансий
                        </div>
                      </div>
                    </Button>
                  </motion.div>
                  <motion.div whileHover={{ x: 4 }}>
                    <Button
                      onClick={extendSkills}
                      disabled={loading}
                      variant="outline"
                      className="w-full justify-start h-14 border-2 hover:border-purple-500 hover:bg-purple-50 dark:hover:bg-purple-950/30"
                    >
                      <FileText className="mr-3 size-5 text-purple-600" />
                      <div className="text-left">
                        <div className="font-semibold">Расширить навыки</div>
                        <div className="text-xs text-muted-foreground">
                          Добавление новых IT-навыков в базу
                        </div>
                      </div>
                    </Button>
                  </motion.div>
                  <motion.div whileHover={{ x: 4 }}>
                    <Button
                      onClick={fullRebuild}
                      disabled={loading}
                      variant="destructive"
                      className="w-full justify-start h-14 shadow-lg"
                    >
                      <RefreshCw className="mr-3 size-5" />
                      <div className="text-left">
                        <div className="font-semibold">Полная пересборка</div>
                        <div className="text-xs opacity-90">
                          Очистка кэшей и пересборка проекта
                        </div>
                      </div>
                    </Button>
                  </motion.div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <Footer />
      </motion.div>
    </div>
  );
}
