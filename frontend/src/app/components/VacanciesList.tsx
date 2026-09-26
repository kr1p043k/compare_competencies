import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { VacancyCard } from "./VacancyCard";
import { VacancyDetailPanel } from "./VacancyDetailPanel";
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
import { Search, X, Filter, Briefcase, TrendingUp, Loader2, AlertCircle, ChevronLeft, ChevronRight, ChevronUp, LayoutGrid, List, Database, Sparkles, Rocket, CheckCircle2, Globe, MapPin, ChevronDown, Download } from "lucide-react";

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
  canRunPipeline?: boolean;
}

export function VacanciesList({ pipelineStep, pipelineLoading, restartFlag, onStartPipeline, pipelineMaxPages, pipelinePeriod, canRunPipeline = true }: VacanciesListProps) {
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
  const [drawerVacancy, setDrawerVacancy] = useState<Vacancy | null>(null);
  const [vacancyInfo, setVacancyInfo] = useState<{ count: number; with_skills?: number; file_modified: string | null; date_range: { from: string; to: string } | null; load_error: string | null } | null>(null);
  const [showPipelineSetup, setShowPipelineSetup] = useState(false);
  const [pipelineRegion, setPipelineRegion] = useState("0");
  const [selectedCities, setSelectedCities] = useState<string[]>([]);
  const [cityQuery, setCityQuery] = useState("");
  const [openLetters, setOpenLetters] = useState<Record<string, boolean>>({});
  const [filterOpen, setFilterOpen] = useState(false);
  const [cityMode, setCityMode] = useState(false);
  const [pipelineProfession, setPipelineProfession] = useState("");
  const [pipelineMaxPagesLocal, setPipelineMaxPagesLocal] = useState(20);
  const [pipelinePeriodLocal, setPipelinePeriodLocal] = useState(30);
  const [showAllMarketInfo, setShowAllMarketInfo] = useState(false);
  const [allMarketVacancyCount, setAllMarketVacancyCount] = useState(0);
  const [monthsFilter, setMonthsFilter] = useState<number | null>(null);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [applied, setApplied] = useState<{ search: string; experience: string; city: string; months: number | null; date_from: string; date_to: string }>({ search: "", experience: "all", city: "all", months: null, date_from: "", date_to: "" });
  const activeFilterCount = [
    experienceFilter !== "all",
    cityFilter !== "all",
    searchQuery.trim() !== "",
    monthsFilter !== null,
    dateFrom !== "" || dateTo !== "",
  ].filter(Boolean).length;
  const handledCompleteRef = useRef(false);
  const itemsPerPage = 12;
  const fmtDateRU = (iso: string) => {
    const p = (iso || "").slice(0, 10).split("-");
    return p.length === 3 ? `${p[2]}.${p[1]}.${p[0]}` : iso;
  };
  const PERIOD_PRESETS: { value: number | null; label: string }[] = [
    { value: null, label: "Всё время" },
    { value: 1, label: "Месяц" },
    { value: 3, label: "3 месяца" },
    { value: 6, label: "Полгода" },
    { value: 12, label: "Год" },
  ];

  useEffect(() => {
    loadVacancies();
  }, [currentPage, applied]);

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

      if (applied.experience && applied.experience !== "all") {
        params.append("experience", applied.experience);
      }

      if (applied.city && applied.city !== "all") {
        params.append("region", applied.city);
      }

      if (applied.search.trim()) {
        params.append("search", applied.search.trim());
      }

      if (applied.months) {
        params.append("months", applied.months.toString());
      }

      if (applied.date_from) {
        params.append("date_from", applied.date_from);
      }
      if (applied.date_to) {
        params.append("date_to", applied.date_to);
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
      if (applied.experience && applied.experience !== "all") params.append("experience", applied.experience);
      if (applied.city && applied.city !== "all") params.append("region", applied.city);
      if (applied.search.trim()) params.append("search", applied.search.trim());
      if (applied.months) params.append("months", applied.months.toString());
      if (applied.date_from) params.append("date_from", applied.date_from);
      if (applied.date_to) params.append("date_to", applied.date_to);
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

  const applyFilters = (over: Partial<{ search: string; experience: string; city: string; months: number | null; date_from: string; date_to: string }> = {}) => {
    setApplied({
      search: searchQuery,
      experience: experienceFilter,
      city: cityFilter,
      months: monthsFilter,
      date_from: dateFrom,
      date_to: dateTo,
      ...over,
    });
    setCurrentPage(1);
  };

  const handleSearch = () => {
    applyFilters();
  };

  const clearFilters = () => {
    setSearchQuery("");
    setExperienceFilter("all");
    setCityFilter("all");
    setMonthsFilter(null);
    setDateFrom("");
    setDateTo("");
    setApplied({ search: "", experience: "all", city: "all", months: null, date_from: "", date_to: "" });
    setCurrentPage(1);
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
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500 dark:from-blue-950/30 to-purple-600 rounded-2xl blur-xl opacity-30 dark:opacity-50 animate-pulse" />
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
              <span>Записи с {fmtDateRU(vacancyInfo.date_range.from)}</span>
            )}
            <span>файл: {vacancyInfo.file_modified}</span>
            <span>{vacancyInfo.count} вакансий</span>
            {pipelineMaxPages && <span>стр: {pipelineMaxPages}</span>}
            {pipelinePeriod && <span>период: {pipelinePeriod} дн.</span>}
          </div>
        )}
      </motion.div>

      {/* Pipeline trigger / settings panel */}
      {canRunPipeline && (showPipelineSetup && (!pipelineStep || pipelineStep.status !== "running") ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <Card className="border-0 shadow-xl bg-gradient-to-br from-sky-50 to-indigo-50 dark:from-sky-950/20 dark:to-indigo-950/20">
            <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gradient-to-br from-sky-500 dark:from-sky-950/30 to-indigo-600 rounded-lg shadow-md">
                    <Rocket className="size-5 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-lg">Настройки сбора вакансий</CardTitle>
                    <CardDescription>Выберите профессию и города для поиска</CardDescription>
                  </div>
                </div>
                <Button variant="ghost" size="icon" onClick={() => setShowPipelineSetup(false)} title="Свернуть настройки">
                  <ChevronUp className="size-4" />
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
                <p className="text-xs text-slate-400">{!cityMode ? "Весь рынок – поиск по всем IT-профессиям" : "Оставьте пустым для поиска по всем профессиям"}</p>
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
                  className={`flex-1 h-10 gap-2 ${!cityMode ? "bg-slate-900 text-white hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-200" : ""}`}
                >
                  <Globe className="size-4" />
                  Весь рынок
                </Button>
                <Button
                  variant={cityMode ? "default" : "outline"}
                  onClick={() => { setCityMode(true); if (selectedCities.length === 0) setSelectedCities([...HH_REGIONS]); }}
                  className={`flex-1 h-10 gap-2 ${cityMode ? "bg-slate-900 text-white hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-200" : ""}`}
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
                          className="rounded border-slate-300 dark:border-slate-600"
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
                        Будут собраны вакансии {vacancyInfo?.count ? `(в базе: ${vacancyInfo.count} шт.${vacancyInfo.with_skills ? `, с навыками: ${vacancyInfo.with_skills} шт.` : ""})` : ""} по всем IT-направлениям: Data Scientist, ML Engineer, Python/Java/Fullstack/Frontend/Backend Developer, DevOps, QA, Security, SRE, Mobile Dev, Analyst, Architect, Team Lead, UX/UI Designer, Game Dev и другим
                      </p>
                      <div className="text-xs text-blue-600 dark:text-blue-400">
                        <span className="font-medium">Города ({HH_REGIONS.length}):</span>
                        <input
                          value={cityQuery}
                          onChange={(e) => setCityQuery(e.target.value)}
                          placeholder="Найти город..."
                          className="mt-2 w-full h-8 px-3 text-xs rounded-lg border border-blue-200 dark:border-blue-800 bg-white dark:bg-slate-950 text-gray-900 dark:text-slate-100 outline-none"
                        />
                        <div className="mt-2 max-h-56 overflow-y-auto rounded-lg border border-blue-200/60 dark:border-blue-800/60 divide-y divide-blue-100 dark:divide-blue-900/40">
                          {(() => {
                            const q = cityQuery.trim().toLowerCase();
                            const filtered = HH_REGIONS.filter((c) => !q || c.toLowerCase().includes(q));
                            const groups = new Map<string, string[]>();
                            for (const c of filtered) {
                              const letter = (c[0] || "#").toUpperCase();
                              if (!groups.has(letter)) groups.set(letter, []);
                              groups.get(letter)!.push(c);
                            }
                            if (filtered.length === 0) {
                              return <p className="p-3 text-xs text-blue-500">Ничего не найдено</p>;
                            }
                            return [...groups.entries()]
                              .sort(([a], [b]) => a.localeCompare(b, "ru"))
                              .map(([letter, cities]) => {
                                const sel = cities.filter((c) => selectedCities.includes(c)).length;
                                const open = openLetters[letter] ?? q.length > 0 ?? sel > 0;
                                return (
                                  <div key={letter}>
                                    <button
                                      onClick={() => setOpenLetters((p) => ({ ...p, [letter]: !(p[letter] ?? q.length > 0 ?? sel > 0) }))}
                                      className="w-full flex items-center gap-2 px-3 py-1.5 text-xs font-semibold text-blue-800 dark:text-blue-200 hover:bg-white/60 dark:hover:bg-slate-800/60 cursor-pointer"
                                    >
                                      <ChevronDown className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`} />
                                      {letter}
                                      <span className="ml-auto font-normal text-blue-500">
                                        {sel > 0 ? `${sel}/${cities.length}` : cities.length}
                                      </span>
                                    </button>
                                    {open && (
                                      <div className="px-3 pb-2 flex flex-wrap gap-1">
                                        {cities.map((c) => {
                                          const on = selectedCities.includes(c);
                                          return (
                                            <button
                                              key={c}
                                              onClick={() => setSelectedCities((prev) => (on ? prev.filter((x) => x !== c) : [...prev, c]))}
                                              title={on ? "Убрать из выборки" : "Добавить к выборке"}
                                              className={`px-2 py-0.5 text-xs rounded-full border transition-colors cursor-pointer ${
                                                on
                                                  ? "bg-blue-600 text-white border-blue-600"
                                                  : "bg-white/70 dark:bg-slate-800/70 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800 hover:border-blue-500"
                                              }`}
                                            >
                                              {c}
                                            </button>
                                          );
                                        })}
                                      </div>
                                    )}
                                  </div>
                                );
                              });
                          })()}
                        </div>
                        {selectedCities.length > 0 && (
                          <p className="mt-1 text-xs text-blue-500">Выбрано для сбора: {selectedCities.length}</p>
                        )}
                      </div>
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
                  <div className="p-2 bg-gradient-to-br from-sky-500 dark:from-sky-950/30 to-indigo-600 rounded-lg shrink-0">
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
      ))}

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
                <div className="p-2 bg-blue-700 rounded-lg">
                  <Filter className="size-5 text-white" />
                </div>
                <div>
                  <CardTitle>Фильтры и поиск</CardTitle>
                  <CardDescription>Настройте параметры для поиска вакансий</CardDescription>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setFilterOpen(true)}
                  className="gap-2"
                >
                  <Filter className="size-4" />
                  Фильтры
                  {activeFilterCount > 0 && (
                    <span className="ml-1 px-1.5 py-0.5 rounded-full bg-blue-600 text-white text-[11px] font-semibold">
                      {activeFilterCount}
                    </span>
                  )}
                </Button>
                <Button
                  variant={viewMode === "grid" ? "default" : "outline"}
                  size="icon"
                  onClick={() => setViewMode("grid")}
                  className={`size-9 ${viewMode === "grid" ? "bg-slate-900 text-white hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-200" : ""}`}
                >
                  <LayoutGrid className="size-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "default" : "outline"}
                  size="icon"
                  onClick={() => setViewMode("list")}
                  className={`size-9 ${viewMode === "list" ? "bg-slate-900 text-white hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-200" : ""}`}
                >
                  <List className="size-4" />
                </Button>
              </div>
            </div>
          </CardHeader>
        </Card>
      </motion.div>

      {/* Фильтры выезжают справа */}
      <AnimatePresence>
        {filterOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-slate-950/50"
              onClick={() => setFilterOpen(false)}
            />
            <motion.aside
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "tween", duration: 0.22, ease: "easeOut" }}
              className="fixed top-0 right-0 bottom-0 z-50 w-full sm:max-w-lg bg-white dark:bg-slate-950 border-l border-gray-200 dark:border-slate-700 shadow-2xl flex flex-col"
            >
              <div className="flex items-center justify-between gap-3 p-5 border-b border-gray-200 dark:border-slate-700">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-700 rounded-lg">
                    <Filter className="size-5 text-white" />
                  </div>
                  <div>
                    <div className="text-lg font-semibold text-gray-900 dark:text-slate-100">Фильтры и поиск</div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">
                      {activeFilterCount > 0 ? `Активно: ${activeFilterCount}` : "Показаны все вакансии"}
                    </div>
                  </div>
                </div>
                <Button variant="ghost" size="icon" onClick={() => setFilterOpen(false)} title="Закрыть">
                  <X className="size-4" />
                </Button>
              </div>
              <div className="flex-1 overflow-y-auto p-6">
            <div className="space-y-6">
              {/* Search */}
              <div className="space-y-2">
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
                      placeholder="Должность или компания"
                      className="pl-10 h-10 rounded-lg focus-visible:ring-2 focus-visible:ring-blue-500"
                    />
                  </div>
                  <Button
                    onClick={handleSearch}
                    disabled={loading}
                    aria-label="Найти"
                    className="h-10 w-11 bg-blue-700 hover:bg-blue-800 text-white rounded-lg"
                  >
                    {loading ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <Search className="size-4" />
                    )}
                  </Button>
                </div>
              </div>

              <div className="border-t border-slate-200 dark:border-slate-800" />

              {/* Experience + city */}
              <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Опыт
                </label>
                <Select value={experienceFilter} onValueChange={(v) => { setExperienceFilter(v); }}>
                  <SelectTrigger className="h-10 rounded-lg">
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
                <Select value={cityFilter} onValueChange={(v) => { setCityFilter(v); }}>
                  <SelectTrigger className="h-10 rounded-lg">
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

              <div className="border-t border-slate-200 dark:border-slate-800" />

              {/* Period presets */}
              <div className="space-y-3">
                <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Период
                </label>
                <div className="flex flex-wrap gap-2">
                  {PERIOD_PRESETS.map(p => {
                    const isActive = monthsFilter === p.value && !dateFrom && !dateTo;
                    const dimmed = (dateFrom !== "" || dateTo !== "") && p.value !== null;
                    return (
                      <button
                        key={p.label}
                        type="button"
                        onClick={() => { setMonthsFilter(p.value); if (p.value !== null) { setDateFrom(""); setDateTo(""); } }}
                        className={`h-9 px-4 rounded-full border text-sm font-medium transition-colors focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:outline-none ${
                          isActive
                            ? "bg-blue-700 border-blue-700 text-white"
                            : "border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 hover:border-blue-500 hover:text-blue-700 dark:hover:text-blue-400"
                        } ${dimmed ? "opacity-40" : ""}`}
                      >
                        {p.label}
                      </button>
                    );
                  })}
                </div>
                <div className="flex items-center gap-2">
                  <Input
                    type="date"
                    value={dateFrom}
                    min={vacancyInfo?.date_range?.from?.slice(0, 10)}
                    max={dateTo || vacancyInfo?.date_range?.to?.slice(0, 10)}
                    onChange={(e) => { setDateFrom(e.target.value); if (e.target.value) setMonthsFilter(null); }}
                    className="h-10 rounded-lg dark:[color-scheme:dark]"
                    aria-label="Дата от"
                  />
                  <span className="text-sm text-slate-400">—</span>
                  <Input
                    type="date"
                    value={dateTo}
                    min={dateFrom || vacancyInfo?.date_range?.from?.slice(0, 10)}
                    max={vacancyInfo?.date_range?.to?.slice(0, 10)}
                    onChange={(e) => { setDateTo(e.target.value); if (e.target.value) setMonthsFilter(null); }}
                    className="h-10 rounded-lg dark:[color-scheme:dark]"
                    aria-label="Дата до"
                  />
                </div>
                {vacancyInfo?.date_range && (
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Записи с {fmtDateRU(vacancyInfo.date_range.from)}
                  </p>
                )}
              </div>
            </div>
            </div>

            {/* Active filters */}
            {(applied.experience !== "all" || applied.city !== "all" || applied.search || applied.months !== null || applied.date_from || applied.date_to) && (
              <div className="flex flex-wrap items-center gap-2 px-6 pt-4 border-t border-slate-200 dark:border-slate-800">
                {applied.experience !== "all" && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => { setExperienceFilter("all"); applyFilters({ experience: "all" }); }}
                  >
                    {applied.experience} ✕
                  </Badge>
                )}
                {applied.city !== "all" && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => { setCityFilter("all"); applyFilters({ city: "all" }); }}
                  >
                    {applied.city} ✕
                  </Badge>
                )}
                {applied.search && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => {
                      setSearchQuery("");
                      applyFilters({ search: "" });
                    }}
                  >
                    "{applied.search}" ✕
                  </Badge>
                )}
                {applied.months !== null && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => { setMonthsFilter(null); applyFilters({ months: null }); }}
                  >
                    {applied.months} мес ✕
                  </Badge>
                )}
                {(applied.date_from || applied.date_to) && (
                  <Badge
                    variant="secondary"
                    className="cursor-pointer hover:bg-slate-300 dark:hover:bg-slate-600"
                    onClick={() => { setDateFrom(""); setDateTo(""); applyFilters({ date_from: "", date_to: "" }); }}
                  >
                    {applied.date_from ? fmtDateRU(applied.date_from) : "…"} — {applied.date_to ? fmtDateRU(applied.date_to) : "…"} ✕
                  </Badge>
                )}
              </div>
            )}
              <div className="flex items-center gap-3 p-5 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
                <Button
                  onClick={() => { applyFilters(); setFilterOpen(false); }}
                  disabled={loading}
                  className="h-10 flex-1 bg-blue-700 hover:bg-blue-800 text-white rounded-lg gap-2 whitespace-nowrap"
                >
                  {loading ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Search className="size-4" />
                  )}
                  Показать вакансии{total > 0 ? ` (${total})` : ""}
                </Button>
                <Button
                  onClick={clearFilters}
                  disabled={loading || activeFilterCount === 0}
                  variant="outline"
                  className="h-10 rounded-lg text-slate-600 dark:text-slate-300 gap-2 whitespace-nowrap"
                >
                  <X className="size-4" />
                  Сбросить
                </Button>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

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
                <div className="mt-4 p-4 bg-orange-50 dark:bg-orange-950/30 border border-orange-200 dark:border-orange-800 rounded-lg text-left max-w-lg mx-auto">
                  <p className="text-sm text-orange-800 dark:text-orange-200">
                    <AlertCircle className="size-4 inline mr-1" />
                    <strong>Файл вакансий повреждён.</strong> Файл существует (<code className="text-xs bg-orange-100 dark:bg-orange-950/30 px-1 rounded">{vacancyInfo.file_modified ?? "неизвестно"}</code>), но не может быть прочитан.
                  </p>
                  <p className="text-xs text-orange-700 dark:text-orange-300 mt-2">
                    Попробуйте запустить повторный сбор вакансий – файлы будут перезаписаны.
                  </p>
                </div>
              ) : !vacancyInfo?.file_modified ? (
                <div className="mt-4 p-4 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-lg text-left max-w-lg mx-auto">
                  <p className="text-sm text-amber-800 dark:text-amber-200">
                    <Database className="size-4 inline mr-1" />
                    <strong>Вакансии не собраны.</strong> Нажмите кнопку <strong>«Собрать вакансии»</strong> выше на этой странице.
                  </p>
                  <p className="text-xs text-amber-700 dark:text-amber-300 mt-2">
                    После сбора вакансий данные кэшируются. Если вы уже запускали сбор – проверьте, что бэкенд запущен.
                  </p>
                </div>
              ) : (
                <div className="mt-4 p-4 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-lg text-left max-w-lg mx-auto">
                  <p className="text-sm text-red-800 dark:text-red-200">
                    <AlertCircle className="size-4 inline mr-1" />
                    <strong>Не удалось загрузить данные из файла.</strong> Файл существует, но возникла ошибка при обработке.
                  </p>
                  {vacancyInfo?.load_error && (
                    <p className="text-xs text-red-600 mt-1 font-mono">{vacancyInfo.load_error}</p>
                  )}
                  <p className="text-xs text-red-700 dark:text-red-300 mt-2">
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
              <div className="p-4 bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-lg text-left max-w-lg mx-auto">
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  <Database className="size-4 inline mr-1" />
                  <strong>Если вакансии ещё не собраны</strong> – нажмите кнопку <strong>«Собрать вакансии»</strong> выше на этой странице.
                </p>
                <ul className="mt-2 text-xs text-blue-700 dark:text-blue-300 space-y-1 list-disc list-inside">
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
          <VacancyDetailPanel vacancy={drawerVacancy} onClose={() => setDrawerVacancy(null)} />

          {/* Count + Export */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-between"
          >
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Найдено <span className="font-semibold text-slate-700 dark:text-slate-300">{total}</span> вакансий
              {vacancies.length < total && (
                <span> (показано {vacancies.length})</span>
              )}
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={async () => {
                try {
                  const eq = new URLSearchParams();
                  if (applied.search.trim()) eq.append("search", applied.search.trim());
                  if (applied.experience !== "all") eq.append("experience", applied.experience);
                  if (applied.city !== "all") eq.append("region", applied.city);
                  if (applied.months) eq.append("months", String(applied.months));
                  if (applied.date_from) eq.append("date_from", applied.date_from);
                  if (applied.date_to) eq.append("date_to", applied.date_to);
                  const qs = eq.toString();
                  const r = await fetch(`/api/teacher/export/vacancies${qs ? `?${qs}` : ""}`);
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
            className={`gap-6 ${
              viewMode === "grid"
                ? "columns-1 lg:columns-2 [&>*]:mb-6"
                : "columns-1"
            }`}
          >
            <AnimatePresence mode="popLayout">
              {vacancies.map((vacancy, index) => (
                <motion.div
                  key={vacancy.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: index * 0.05 }}
                  className="h-full break-inside-avoid"
                >
                  <VacancyCard vacancy={vacancy} onOpen={(v) => setDrawerVacancy(v)} />
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
