import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { VacancyCard } from "./VacancyCard";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Search, X, Filter, Briefcase, TrendingUp, Loader2, AlertCircle, ChevronLeft, ChevronRight, LayoutGrid, List, Database, Sparkles, Rocket, CheckCircle2, Globe, MapPin, ChevronDown, Download } from "lucide-react";

const HH_REGIONS = [
  "Москва", "Санкт-Петербург", "Екатеринбург", "Новосибирск",
  "Нижний Новгород", "Казань", "Самара", "Ростов-на-Дону", "Уфа", "Красноярск",
  "Пермь", "Воронеж", "Волгоград", "Краснодар", "Саратов", "Тюмень",
  "Тольятти", "Ижевск", "Барнаул", "Иркутск", "Ульяновск", "Хабаровск",
  "Владивосток", "Ярославль", "Махачкала", "Томск", "Оренбург", "Кемерово",
  "Новокузнецк", "Рязань", "Астрахань", "Пенза", "Набережные Челны", "Липецк",
  "Тула", "Киров", "Чебоксары", "Калининград", "Брянск", "Курск", "Иваново",
  "Магнитогорск", "Тверь", "Ставрополь", "Белгород", "Сочи", "Нижний Тагил",
  "Владимир", "Архангельск", "Калуга", "Сургут", "Чита", "Грозный",
  "Смоленск", "Волжский", "Курган", "Орёл", "Череповец", "Вологда",
  "Мурманск", "Саранск", "Якутск", "Подольск", "Стерлитамак", "Петрозаводск",
  "Кострома", "Новороссийск", "Йошкар-Ола", "Таганрог",
  "Комсомольск-на-Амуре", "Сыктывкар", "Нижневартовск", "Нальчик", "Шахты",
  "Дзержинск", "Благовещенск", "Прокопьевск", "Рыбинск", "Бийск",
  "Великий Новгород", "Северодвинск", "Псков", "Новочеркасск",
  "Южно-Сахалинск", "Батайск", "Кызыл", "Абакан", "Майкоп", "Черкесск",
  "Элиста", "Магадан", "Анадырь", "Биробиджан", "Нарьян-Мар", "Салехард",
  "Ханты-Мансийск", "Горно-Алтайск", "Улан-Удэ", "Петропавловск-Камчатский",
  "Севастополь",
];

const HH_REGION_MAP: Record<string, number> = {
  "Москва": 1, "Санкт-Петербург": 2, "Екатеринбург": 3, "Новосибирск": 4,
  "Нижний Новгород": 66, "Казань": 88, "Самара": 78, "Ростов-на-Дону": 76,
  "Уфа": 99, "Красноярск": 54, "Пермь": 72, "Воронеж": 26, "Волгоград": 24,
  "Краснодар": 53, "Саратов": 79, "Тюмень": 95, "Тольятти": 212, "Ижевск": 96,
  "Барнаул": 11, "Иркутск": 35, "Ульяновск": 98, "Хабаровск": 102,
  "Владивосток": 22, "Ярославль": 112, "Махачкала": 29, "Томск": 90,
  "Оренбург": 70, "Кемерово": 47, "Новокузнецк": 1240, "Рязань": 77,
  "Астрахань": 15, "Пенза": 71, "Набережные Челны": 1641, "Липецк": 58,
  "Тула": 92, "Киров": 49, "Чебоксары": 107, "Калининград": 41, "Брянск": 19,
  "Курск": 56, "Иваново": 32, "Магнитогорск": 1399, "Тверь": 89,
  "Ставрополь": 84, "Белгород": 17, "Сочи": 237, "Нижний Тагил": 1291,
  "Владимир": 23, "Архангельск": 14, "Калуга": 43, "Сургут": 1381,
  "Чита": 106, "Грозный": 105, "Смоленск": 83, "Волжский": 1512,
  "Курган": 55, "Орёл": 69, "Череповец": 1753, "Вологда": 25,
  "Мурманск": 64, "Саранск": 63, "Якутск": 80, "Подольск": 2061,
  "Стерлитамак": 1364, "Петрозаводск": 73, "Кострома": 52,
  "Новороссийск": 1454, "Йошкар-Ола": 61, "Таганрог": 1550,
  "Комсомольск-на-Амуре": 1979, "Сыктывкар": 51, "Нижневартовск": 1375,
  "Нальчик": 39, "Шахты": 1552, "Дзержинск": 247, "Благовещенск": 12,
  "Прокопьевск": 1243, "Рыбинск": 1814, "Бийск": 1220,
  "Великий Новгород": 67, "Северодвинск": 1017, "Псков": 75,
  "Новочеркасск": 1545, "Южно-Сахалинск": 81, "Батайск": 1533,
  "Кызыл": 91, "Абакан": 103, "Майкоп": 8, "Черкесск": 46, "Элиста": 42,
  "Магадан": 60, "Анадырь": 219, "Биробиджан": 31, "Нарьян-Мар": 1986,
  "Салехард": 304, "Ханты-Мансийск": 147, "Горно-Алтайск": 10,
  "Улан-Удэ": 20, "Петропавловск-Камчатский": 44, "Севастополь": 130,
};

interface Vacancy {
  id: string;
  name: string;
  experience: string;
  salary_from?: number;
  salary_to?: number;
  salary_currency?: string;
  employer_name: string;
  employer_logo?: string;
  area: string;
  published_at: string;
  alternate_url: string;
  skills: string[];
  snippet?: {
    requirement?: string;
    responsibility?: string;
  };
}

interface VacanciesResponse {
  items: Vacancy[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

interface PipelineStep {
  step: number;
  total: number;
  status: "running" | "success" | "error" | "completed";
  message: string;
  progress: number;
}

interface VacanciesListProps {
  pipelineStep?: PipelineStep | null;
  pipelineLoading?: boolean;
  restartFlag?: number;
  onStartPipeline?: (regionIds: string, profession: string, maxPages?: number, periodDays?: number) => void;
  pipelineMaxPages?: number;
  pipelinePeriod?: number;
}

export function VacanciesList({ pipelineStep, pipelineLoading, restartFlag, onStartPipeline, pipelineMaxPages, pipelinePeriod }: VacanciesListProps) {
  const [vacancies, setVacancies] = useState<Vacancy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [experienceFilter, setExperienceFilter] = useState<string>("all");
  const [cityFilter, setCityFilter] = useState<string>("all");
  const [availableCities, setAvailableCities] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [vacancyInfo, setVacancyInfo] = useState<{ count: number; file_modified: string | null; date_range: { from: string; to: string } | null; load_error: string | null } | null>(null);
  const [showPipelineSetup, setShowPipelineSetup] = useState(false);
  const [pipelineRegion, setPipelineRegion] = useState("0");
  const [selectedCities, setSelectedCities] = useState<string[]>([]);
  const [cityMode, setCityMode] = useState(false);
  const [pipelineProfession, setPipelineProfession] = useState("");
  const [pipelineMaxPagesLocal, setPipelineMaxPagesLocal] = useState(20);
  const [pipelinePeriodLocal, setPipelinePeriodLocal] = useState(30);
  const [showAllMarketInfo, setShowAllMarketInfo] = useState(false);
  const [allMarketVacancyCount, setAllMarketVacancyCount] = useState(0);
  const handledCompleteRef = useRef(false);
  const itemsPerPage = 12;

  useEffect(() => {
    loadVacancies();
  }, [currentPage, experienceFilter]);

  useEffect(() => {
    if (!pipelineStep) { handledCompleteRef.current = false; return; }
    if (pipelineStep.status === "completed" && !handledCompleteRef.current) {
      handledCompleteRef.current = true;
      refreshVacancies();
    }
  }, [pipelineStep]);

  useEffect(() => {
    if (restartFlag && restartFlag > 0) {
      setShowPipelineSetup(true);
    }
  }, [restartFlag]);

  const loadVacancies = async () => {
    setLoading(true);
    setError(null);
    try {
      const infoR = await fetch("/api/vacancies/info");
      if (infoR.ok) { const d = await infoR.json(); setVacancyInfo(d); }
    } catch {}

    try {
      const offset = (currentPage - 1) * itemsPerPage;
      const params = new URLSearchParams({
        limit: itemsPerPage.toString(),
        offset: offset.toString(),
      });

      if (experienceFilter && experienceFilter !== "all") {
        params.append("experience", experienceFilter);
      }

      if (cityFilter && cityFilter !== "all") {
        params.append("region", cityFilter);
      }

      if (searchQuery.trim()) {
        params.append("search", searchQuery.trim());
      }

      const response = await fetch(`/api/vacancies?${params}`);
      if (!response.ok) {
        throw new Error("Ошибка загрузки вакансий");
      }

      const data: VacanciesResponse = await response.json();

      let filteredItems = data.items;

      setVacancies(filteredItems);
      setTotal(data.total);

      const cities = Array.from(new Set(data.items.map(v => v.area)))
        .filter(city => city && city !== "Не указано")
        .sort();
      setAvailableCities(cities);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }; 

  const refreshVacancies = async () => {
    try {
      const offset = (currentPage - 1) * itemsPerPage;
      const params = new URLSearchParams({ limit: itemsPerPage.toString(), offset: offset.toString() });
      if (experienceFilter && experienceFilter !== "all") params.append("experience", experienceFilter);
      if (searchQuery.trim()) params.append("search", searchQuery.trim());
      const response = await fetch(`/api/vacancies?${params}`);
      if (response.ok) {
        const data: VacanciesResponse = await response.json();
        setVacancies(data.items);
        setTotal(data.total);
      }
    } catch {}
    try {
      const r = await fetch("/api/vacancies/info");
      if (r.ok) { const d = await r.json(); setVacancyInfo(d); }
    } catch {}
  };

  const handleSearch = () => {
    setCurrentPage(1);
    loadVacancies();
  };

  const handleSearchKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const runPipeline = (regionIds?: string) => {
    setShowPipelineSetup(false);
    setShowAllMarketInfo(false);
    const regionParam = (regionIds && regionIds !== "0") ? regionIds : "0";
    const profession = pipelineProfession.trim() || "";
    onStartPipeline?.(regionParam, profession, pipelineMaxPagesLocal, pipelinePeriodLocal);
  };

  const totalPages = Math.ceil(total / itemsPerPage);

  const allCityOptions = [...new Set([...HH_REGIONS, ...availableCities])];

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <div className="inline-flex items-center justify-center gap-3 mb-2">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl blur-xl opacity-50 animate-pulse" />
            <div className="relative bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 rounded-2xl shadow-2xl">
              <Briefcase className="size-8 text-white" />
            </div>
          </div>
          <h2 className="text-4xl font-black bg-gradient-to-r from-slate-900 via-blue-800 to-purple-900 dark:from-white dark:via-blue-200 dark:to-purple-200 bg-clip-text text-transparent">
            Вакансии с hh.ru
          </h2>
        </div>
        <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Найдено <span className="font-bold text-blue-600 dark:text-blue-400">{total}</span> актуальных вакансий
        </p>
        {vacancyInfo && (
          <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-slate-400 dark:text-slate-500">
            {vacancyInfo.date_range && (
              <span>{vacancyInfo.date_range.from} &mdash; {vacancyInfo.date_range.to}</span>
            )}
            <span>файл: {vacancyInfo.file_modified}</span>
            <span>{vacancyInfo.count} вакансий</span>
            {pipelineMaxPages && <span>стр: {pipelineMaxPages}</span>}
            {pipelinePeriod && <span>период: {pipelinePeriod} дн.</span>}
          </div>
        )}
      </motion.div>

      {/* Pipeline trigger / settings panel */}
      {showPipelineSetup && (!pipelineStep || pipelineStep.status !== "running") ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <Card className="border-0 shadow-xl bg-gradient-to-br from-sky-50 to-indigo-50 dark:from-sky-950/20 dark:to-indigo-950/20">
            <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gradient-to-br from-sky-500 to-indigo-600 rounded-lg shadow-md">
                    <Rocket className="size-5 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-lg">Настройки сбора вакансий</CardTitle>
                    <CardDescription>Выберите профессию и города для поиска</CardDescription>
                  </div>
                </div>
                <Button variant="ghost" size="icon" onClick={() => setShowPipelineSetup(false)}>
                  <X className="size-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-6 space-y-5">
              {/* Profession */}
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Профессия
                </label>
                <Input
                  value={pipelineProfession}
                  onChange={(e) => setPipelineProfession(e.target.value)}
                  placeholder="IT-специалист / Data Scientist / Все"
                  disabled={!cityMode}
                  className="h-10"
                />
                <p className="text-xs text-slate-400">{!cityMode ? "Весь рынок — поиск по всем IT-профессиям" : "Оставьте пустым для поиска по всем профессиям"}</p>
              </div>

              {/* Search params */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                    Страниц
                  </label>
                  <Input
                    type="number"
                    min={1}
                    max={100}
                    value={pipelineMaxPagesLocal}
                    onChange={(e) => setPipelineMaxPagesLocal(Math.max(1, Math.min(100, Number(e.target.value) || 20)))}
                    className="h-10"
                  />
                  <p className="text-xs text-slate-400">1-100 (по 100 вакансий на страницу)</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                    Период (дней)
                  </label>
                  <Input
                    type="number"
                    min={1}
                    max={365}
                    value={pipelinePeriodLocal}
                    onChange={(e) => setPipelinePeriodLocal(Math.max(1, Math.min(365, Number(e.target.value) || 30)))}
                    className="h-10"
                  />
                  <p className="text-xs text-slate-400">1–365, по умолч. 30</p>
                </div>
              </div>

              {/* Region mode toggle */}
              <div className="flex gap-2">
                <Button
                  variant={!cityMode ? "default" : "outline"}
                  onClick={() => { setCityMode(false); setSelectedCities([]); setPipelineRegion("0"); setPipelineProfession(""); }}
                  className="flex-1 h-10 gap-2"
                >
                  <Globe className="size-4" />
                  Весь рынок
                </Button>
                <Button
                  variant={cityMode ? "default" : "outline"}
                  onClick={() => { setCityMode(true); if (selectedCities.length === 0) setSelectedCities([...HH_REGIONS]); }}
                  className="flex-1 h-10 gap-2"
                >
                  <MapPin className="size-4" />
                  Выбрать города
                </Button>
              </div>

              {/* City selection */}
              {cityMode && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                      Города ({selectedCities.length} из {HH_REGIONS.length})
                    </label>
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => setSelectedCities([...HH_REGIONS])}
                      >
                        Все
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => setSelectedCities([])}
                      >
                        Сброс
                      </Button>
                    </div>
                  </div>
                  <div className="max-h-48 overflow-y-auto border border-slate-200 dark:border-slate-700 rounded-lg p-2 grid grid-cols-2 sm:grid-cols-3 gap-1">
                    {HH_REGIONS.map((city) => (
                      <label
                        key={city}
                        className="flex items-center gap-2 px-2 py-1.5 rounded-md text-sm cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800 select-none"
                      >
                        <input
                          type="checkbox"
                          checked={selectedCities.includes(city)}
                          onChange={() => {
                            setSelectedCities(prev =>
                              prev.includes(city)
                                ? prev.filter(c => c !== city)
                                : [...prev, city]
                            );
                          }}
                          className="rounded border-slate-300"
                        />
                        <span className="truncate">{city}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {/* All market info */}
              {!cityMode && (
                <>
                  <Button
                    variant="outline"
                    onClick={() => setShowAllMarketInfo(!showAllMarketInfo)}
                    className="w-full h-9 gap-2 text-sm"
                  >
                    <ChevronDown className={`size-4 transition-transform ${showAllMarketInfo ? "rotate-180" : ""}`} />
                    Показать список городов и вакансий для сбора
                  </Button>
                  {showAllMarketInfo && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4 space-y-3"
                    >
                      <div className="flex items-center gap-2 text-sm font-medium text-blue-800 dark:text-blue-200">
                        <Globe className="size-4" />
                        Поиск по всему рынку
                      </div>
                      <p className="text-xs text-blue-700 dark:text-blue-300">
                        Будут собраны вакансии {vacancyInfo?.count ? `(текущая база: ${vacancyInfo.count} шт.)` : ""} по всем IT-направлениям: Data Scientist, ML Engineer, Python/Java/Fullstack/Frontend/Backend Developer, DevOps, QA, Security, SRE, Mobile Dev, Analyst, Architect, Team Lead, UX/UI Designer, Game Dev и другим
                      </p>
                      <div className="text-xs text-blue-600 dark:text-blue-400">
                        <span className="font-medium">Города ({HH_REGIONS.length}):</span>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {HH_REGIONS.map(c => (
                            <span key={c} className="inline-block px-2 py-0.5 bg-white/70 dark:bg-slate-800/70 rounded-full">
                              {c}
                            </span>
                          ))}
                        </div>
                      </div>
                      {vacancyInfo?.date_range && (
                        <p className="text-xs text-blue-500">
                          Данные за период: {vacancyInfo.date_range.from} — {vacancyInfo.date_range.to}
                        </p>
                      )}
                    </motion.div>
                  )}
                </>
              )}

              {/* Run / Cancel */}
              <div className="flex gap-3 pt-2">
                <Button
                  onClick={() => {
                    const regionIds = cityMode && selectedCities.length > 0
                      ? selectedCities.map(c => HH_REGION_MAP[c]).filter(id => id !== undefined).join(",")
                      : "0";
                    runPipeline(regionIds);
                  }}
                  disabled={pipelineLoading}
                  className="flex-1 h-11 bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 gap-2"
                >
                  {pipelineLoading ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Rocket className="size-4" />
                  )}
                  Запустить сбор
                </Button>
                <Button
                  variant="outline"
                  onClick={() => { setShowPipelineSetup(false); setSelectedCities([]); setCityMode(false); setPipelineProfession(""); }}
                  className="h-11"
                >
                  Отмена
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.07 }}
        >
          <Card className="border-0 shadow-xl bg-gradient-to-br from-sky-50 to-indigo-50 dark:from-sky-950/20 dark:to-indigo-950/20">
            <CardContent className="p-4">
              <div className="flex flex-col md:flex-row items-start md:items-center gap-3">
                <div className="flex items-center gap-3 flex-1">
                  <div className="p-2 bg-gradient-to-br from-sky-500 to-indigo-600 rounded-lg shrink-0">
                    <Rocket className="size-5 text-white" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                      Сбор вакансий
                    </p>
                    <p className="text-xs text-slate-500">
                      Загрузить актуальные вакансии с hh.ru для всех профилей
                    </p>
                  </div>
                </div>
                <Button
                  onClick={() => setShowPipelineSetup(true)}
                  disabled={pipelineLoading}
                  className="h-9 bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 px-6 whitespace-nowrap"
                  size="sm"
                >
                  {pipelineLoading ? (
                    <Loader2 className="size-4 mr-1 animate-spin" />
                  ) : (
                    <Rocket className="size-4 mr-1" />
                  )}
                  Собрать вакансии
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
          <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg shadow-md">
                  <Filter className="size-5 text-white" />
                </div>
                <div>
                  <CardTitle>Фильтры и поиск</CardTitle>
                  <CardDescription>Настройте параметры для поиска вакансий</CardDescription>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant={viewMode === "grid" ? "default" : "outline"}
                  size="icon"
                  onClick={() => setViewMode("grid")}
                  className="size-9"
                >
                  <LayoutGrid className="size-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "default" : "outline"}
                  size="icon"
                  onClick={() => setViewMode("list")}
                  className="size-9"
                >
                  <List className="size-4" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {/* Search */}
              <div className="md:col-span-2 space-y-2">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Поиск по названию
                </label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-slate-400" />
                    <Input
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyPress={handleSearchKeyPress}
                      placeholder="Введите должность или компанию..."
                      className="pl-10 h-11 border-2"
                    />
                  </div>
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button
                      onClick={handleSearch}
                      disabled={loading}
                      className="h-11 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                    >
                      {loading ? (
                        <Loader2 className="size-4 animate-spin" />
                      ) : (
                        <Search className="size-4" />
                      )}
                    </Button>
                  </motion.div>
                </div>
              </div>

              {/* Experience filter */}
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Уровень опыта
                </label>
                <Select value={experienceFilter} onValueChange={(v) => { setExperienceFilter(v); setCurrentPage(1); }}>
                  <SelectTrigger className="h-11 border-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Все уровни</SelectItem>
                    <SelectItem value="junior">Junior</SelectItem>
                    <SelectItem value="middle">Middle</SelectItem>
                    <SelectItem value="senior">Senior</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* City filter */}
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Город
                </label>
                <Select value={cityFilter} onValueChange={(v) => { setCityFilter(v); setCurrentPage(1); }}>
                  <SelectTrigger className="h-11 border-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="max-h-80">
                    <SelectItem value="all">Все города</SelectItem>
                    {allCityOptions.map(city => (
                      <SelectItem key={city} value={city}>
                        {city}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Active filters */}
            {(experienceFilter !== "all" || cityFilter !== "all" || searchQuery) && (
              <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-slate-200/50 dark:border-slate-700/50">
                <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
                  Активные фильтры:
                </span>
                {experienceFilter !== "all" && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => setExperienceFilter("all")}
                  >
                    {experienceFilter} ✕
                  </Badge>
                )}
                {cityFilter !== "all" && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => { setCityFilter("all"); setCurrentPage(1); }}
                  >
                    {cityFilter} ✕
                  </Badge>
                )}
                {searchQuery && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => {
                      setSearchQuery("");
                      handleSearch();
                    }}
                  >
                    "{searchQuery}" ✕
                  </Badge>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Vacancies Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center space-y-4">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            >
              <Loader2 className="size-16 text-blue-600 mx-auto" />
            </motion.div>
            <p className="text-slate-600 dark:text-slate-400">Загрузка вакансий...</p>
          </div>
        </div>
      ) : error ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <Card className="border-2 border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/20">
            <CardContent className="pt-6 text-center">
              <AlertCircle className="size-12 text-red-600 dark:text-red-400 mx-auto mb-3" />
              <h3 className="text-lg font-semibold text-red-900 dark:text-red-100 mb-2">
                Ошибка загрузки
              </h3>
              <p className="text-red-700 dark:text-red-300">{error}</p>

              {vacancyInfo?.load_error?.startsWith("corrupted:") ? (
                <div className="mt-4 p-4 bg-orange-50 border border-orange-200 rounded-lg text-left max-w-lg mx-auto">
                  <p className="text-sm text-orange-800">
                    <AlertCircle className="size-4 inline mr-1" />
                    <strong>Файл вакансий повреждён.</strong> Файл существует (<code className="text-xs bg-orange-100 px-1 rounded">{vacancyInfo.file_modified ?? "неизвестно"}</code>), но не может быть прочитан.
                  </p>
                  <p className="text-xs text-orange-700 mt-2">
                    Попробуйте запустить повторный сбор вакансий — файлы будут перезаписаны.
                  </p>
                </div>
              ) : !vacancyInfo?.file_modified ? (
                <div className="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-lg text-left max-w-lg mx-auto">
                  <p className="text-sm text-amber-800">
                    <Database className="size-4 inline mr-1" />
                    <strong>Вакансии не собраны.</strong> Нажмите кнопку <strong>«Собрать вакансии»</strong> выше на этой странице.
                  </p>
                  <p className="text-xs text-amber-700 mt-2">
                    После сбора вакансий данные кэшируются. Если вы уже запускали сбор — проверьте, что бэкенд запущен.
                  </p>
                </div>
              ) : (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-left max-w-lg mx-auto">
                  <p className="text-sm text-red-800">
                    <AlertCircle className="size-4 inline mr-1" />
                    <strong>Не удалось загрузить данные из файла.</strong> Файл существует, но возникла ошибка при обработке.
                  </p>
                  {vacancyInfo?.load_error && (
                    <p className="text-xs text-red-600 mt-1 font-mono">{vacancyInfo.load_error}</p>
                  )}
                  <p className="text-xs text-red-700 mt-2">
                    Попробуйте перезапустить сервер или запустить повторный сбор.
                  </p>
                </div>
              )}

              <Button
                onClick={loadVacancies}
                variant="outline"
                className="mt-4 border-red-300 dark:border-red-700"
              >
                Попробовать снова
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      ) : vacancies.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <Card className="border-2 border-slate-200 dark:border-slate-700">
            <CardContent className="pt-6 text-center py-20">
              <Database className="size-16 text-slate-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Вакансии не найдены
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                По вашему запросу ничего не найдено
              </p>
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg text-left max-w-lg mx-auto">
                <p className="text-sm text-blue-800">
                  <Database className="size-4 inline mr-1" />
                  <strong>Если вакансии ещё не собраны</strong> — нажмите кнопку <strong>«Собрать вакансии»</strong> выше на этой странице.
                </p>
                <ul className="mt-2 text-xs text-blue-700 space-y-1 list-disc list-inside">
                  <li>После нажатия запустится полный цикл сбора (10-15 минут)</li>
                  <li>Прогресс будет отображаться на этой же странице</li>
                  <li>После завершения данные обновятся автоматически</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      ) : (
        <>
          {/* Count + Export */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-between"
          >
            <p className="text-sm text-slate-500">
              Найдено <span className="font-semibold text-slate-700 dark:text-slate-300">{vacancies.length}</span> вакансий
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={async () => {
                try {
                  const r = await fetch("/api/admin/export/excel");
                  if (r.ok) {
                    const blob = await r.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `vacancies_skills_${new Date().toISOString().split("T")[0]}.xlsx`;
                    document.body.appendChild(a);
                    a.click();
                    URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                  }
                } catch (e) {
                  console.error("Export failed:", e);
                }
              }}
              className="border-green-300 dark:border-green-700 hover:bg-green-100 dark:hover:bg-green-900/50 gap-2"
            >
              <Download className="size-4" />
              Excel
            </Button>
          </motion.div>

          <motion.div
            className={`grid gap-6 ${
              viewMode === "grid"
                ? "grid-cols-1 lg:grid-cols-2"
                : "grid-cols-1"
            }`}
            layout
          >
            <AnimatePresence mode="popLayout">
              {vacancies.map((vacancy, index) => (
                <motion.div
                  key={vacancy.id}
                  layout
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <VacancyCard vacancy={vacancy} />
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>

          {/* Pagination */}
          {totalPages > 1 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="flex items-center justify-center gap-2"
            >
              <Button
                variant="outline"
                size="icon"
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1 || loading}
                className="size-10"
              >
                <ChevronLeft className="size-4" />
              </Button>

              {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                let pageNum: number;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (currentPage <= 3) {
                  pageNum = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }

                return (
                  <Button
                    key={pageNum}
                    variant={currentPage === pageNum ? "default" : "outline"}
                    size="icon"
                    onClick={() => setCurrentPage(pageNum)}
                    disabled={loading}
                    className={`size-10 ${
                      currentPage === pageNum
                        ? "bg-gradient-to-r from-blue-600 to-purple-600"
                        : ""
                    }`}
                  >
                    {pageNum}
                  </Button>
                );
              })}

              <Button
                variant="outline"
                size="icon"
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages || loading}
                className="size-10"
              >
                <ChevronRight className="size-4" />
              </Button>
            </motion.div>
          )}
        </>
      )}
    </div>
  );
}
