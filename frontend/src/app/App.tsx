import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Button } from "./components/ui/button";
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
import { Footer } from "./components/Footer";
import { VacanciesList } from "./components/VacanciesList";
import { PipelineProgress } from "./components/PipelineProgress";
import { AnalysisTab } from "./components/AnalysisTab";
import { motion, AnimatePresence } from "motion/react";
import {
  Database,
  Sparkles,
  FileText,
  Loader2,
  BarChart3,
  Zap,
  Award,
  Briefcase,
  Rocket,
  TrendingUp,
  Target,
  LogIn,
  User,
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
];

const API = "/api";

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
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userName, setUserName] = useState<string | null>(null);

  function showStatus(type: "success" | "error" | "info", message: string) {
    setStatus({ type, message });
    setTimeout(() => setStatus({ type: null, message: "" }), 5000);
  }

  function handleSSOLogin() {
    // Заглушка для SSO ЮФУ
    showStatus("info", "SSO аутентификация ЮФУ находится в разработке");
    // В будущем здесь будет редирект на SSO:
    // window.location.href = "https://sso.sfedu.ru/oauth/authorize?...";
  }

  function handleLogout() {
    setIsAuthenticated(false);
    setUserName(null);
    showStatus("info", "Вы вышли из системы");
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

  async function runPipelineWithProgress() {
    try {
      setLoading(true);
      setPipelineStep(null);

      const params = new URLSearchParams({ regions });

      const response = await fetch(`${API}/pipeline/full-cycle?${params}`, {
        method: "POST",
        headers: {
          Accept: "text/event-stream",
        },
      });

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
                showStatus("success", "Пайплайн завершен!");
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

            {/* Auth Button */}
            <div>
              {isAuthenticated ? (
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-lg">
                    <User className="size-4 text-gray-600" />
                    <span className="text-sm font-medium text-gray-900">
                      {userName || "Студент ЮФУ"}
                    </span>
                  </div>
                  <Button
                    onClick={handleLogout}
                    variant="outline"
                    className="border-gray-300"
                  >
                    Выйти
                  </Button>
                </div>
              ) : (
                <Button
                  onClick={handleSSOLogin}
                  className="bg-blue-600 hover:bg-blue-700 text-white shadow-sm"
                >
                  <LogIn className="size-4 mr-2" />
                  Войти в профиль ЮФУ
                </Button>
              )}
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
        <Tabs defaultValue="vacancies" className="space-y-6">
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
              value="pipeline"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <Zap className="size-4" />
              Пайплайн
            </TabsTrigger>
            <TabsTrigger
              value="analysis"
              className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-white data-[state=active]:text-gray-900 data-[state=active]:shadow-sm"
            >
              <Target className="size-4" />
              Анализ
            </TabsTrigger>
          </TabsList>

          {/* Vacancies Tab */}
          <TabsContent value="vacancies">
            <VacanciesList />
          </TabsContent>

          {/* Pipeline Tab */}
          <TabsContent value="pipeline" className="space-y-6">
            {pipelineStep && <PipelineProgress currentStep={pipelineStep} />}

            <Card className="border border-gray-200 shadow-sm">
              <CardHeader className="border-b border-gray-200 bg-gray-50">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 bg-blue-600 rounded-lg">
                    <Rocket className="size-5 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-xl font-semibold text-gray-900">
                      Пайплайн анализа
                    </CardTitle>
                    <CardDescription className="text-sm text-gray-600">
                      Полный цикл сбора данных и ML-анализа
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-6 space-y-6">
                {/* Region selector */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-gray-900">
                    Регион сбора вакансий
                  </Label>
                  <RegionCombobox
                    regions={REGIONS}
                    value={regions}
                    onChange={setRegions}
                  />
                </div>

                {/* Main button */}
                <Button
                  onClick={runPipelineWithProgress}
                  disabled={loading}
                  className="w-full h-12 bg-blue-600 hover:bg-blue-700 text-white font-medium shadow-sm"
                  size="lg"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 size-5 animate-spin" />
                      Выполнение полного цикла...
                    </>
                  ) : (
                    <>
                      <Rocket className="mr-2 size-5" />
                      Запустить полный цикл анализа
                    </>
                  )}
                </Button>

                {/* Info */}
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">
                    Что включает полный цикл?
                  </h4>
                  <ul className="space-y-1 text-sm text-gray-700">
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-semibold">1.</span>
                      <span>Сбор IT-вакансий с hh.ru</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-semibold">2.</span>
                      <span>Обучение кластеров вакансий</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-semibold">3.</span>
                      <span>Обучение ML-модели ранжирования</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-600 font-semibold">4.</span>
                      <span>GAP-анализ и генерация рекомендаций</span>
                    </li>
                  </ul>
                  <p className="mt-3 text-xs text-gray-600">
                    ⏱️ Ожидаемое время выполнения: 10-15 минут
                  </p>
                </div>
              </CardContent>
            </Card>
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
                  <Select value={profile} onValueChange={setProfile}>
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
                          <span>DATA CENTER (middle)</span>
                        </div>
                      </SelectItem>
                      <SelectItem value="top_dc">
                        <div className="flex items-center gap-2">
                          <Award className="size-4 text-pink-600" />
                          <span>TOP DATA CENTER (senior)</span>
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
              onProfileChange={setProfile}
            />
          </TabsContent>
        </Tabs>

        <div className="mt-12">
          <Footer />
        </div>
      </main>
    </div>
  );
}
