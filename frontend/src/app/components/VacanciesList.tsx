import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { motion } from "motion/react";
import { VacancyDetailPanel } from "./VacancyDetailPanel";
import {
  Badge,
  Button,
  Card,
  CardBody,
  EmptyState,
  ErrorState,
  PageHeader,
  Skeleton,
  inputClass,
} from "../../design";
import { cn } from "./ui/utils";
import { Search, Filter, Loader2, AlertCircle, ChevronUp, LayoutGrid, List, Database, Rocket, Globe, MapPin, ChevronDown } from "lucide-react";
import { reportError } from "./../../lib/errors";
import { request } from "../../lib/api";
import { useIncrementalList } from "../../lib/useIncrementalList";
import { HH_REGIONS, HH_REGION_MAP, PAGE_SIZE, fmtDateRU } from "./vacancies/constants";
import type {
  AppliedFilters,
  Vacancy,
  VacanciesListProps,
  VacanciesResponse,
  VacancyInfo,
} from "./vacancies/types";
import { MemoVacancyCard } from "./vacancies/VacancyCard";
import { VacancyFiltersDrawer } from "./vacancies/VacancyFiltersDrawer";
import { VacancyPagination } from "./vacancies/VacancyPagination";
import { VacancyStatsBar } from "./vacancies/VacancyStatsBar";

export function VacanciesList({ pipelineStep, pipelineLoading, restartFlag, onStartPipeline, canRunPipeline = true }: VacanciesListProps) {
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
  const [vacancyInfo, setVacancyInfo] = useState<VacancyInfo | null>(null);
  const [showPipelineSetup, setShowPipelineSetup] = useState(false);
  const [, setPipelineRegion] = useState("0");
  const [selectedCities, setSelectedCities] = useState<string[]>([]);
  const [cityQuery, setCityQuery] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);
  const [cityMode, setCityMode] = useState(false);
  const [pipelineProfession, setPipelineProfession] = useState("");
  const [pipelineMaxPagesLocal, setPipelineMaxPagesLocal] = useState(20);
  const [pipelinePeriodLocal, setPipelinePeriodLocal] = useState(30);
  const [showAllMarketInfo, setShowAllMarketInfo] = useState(false);
  const [monthsFilter, setMonthsFilter] = useState<number | null>(null);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [applied, setApplied] = useState<AppliedFilters>({ search: "", experience: "all", city: "all", months: null, date_from: "", date_to: "" });
  const activeFilterCount = [
    experienceFilter !== "all",
    cityFilter !== "all",
    searchQuery.trim() !== "",
    monthsFilter !== null,
    dateFrom !== "" || dateTo !== "",
  ].filter(Boolean).length;
  const handledCompleteRef = useRef(false);
  // 48 на страницу: столько забираем с сервера. Больше окна рендера (PAGE_SIZE
  // = 24), поэтому инкрементальный показ реально включается, а не простаивает.
  const itemsPerPage = 48;

  useEffect(() => {
    loadVacancies();
  }, [currentPage, applied]);

  // Монтирование: защита от setState после размонтирования (уход с вкладки
  // во время запроса) и признак «компонент жив» для обработчиков ниже.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // Информация о базе вакансий не зависит от страницы и фильтров —
  // грузим один раз на монтирование, а не при каждом переключении страницы
  // (обновление после сбора данных делает refreshVacancies).
  useEffect(() => {
    request<VacancyInfo>("/api/vacancies/info")
      .then((info) => { if (mountedRef.current) setVacancyInfo(info); })
      .catch((err) => reportError("app/components/VacanciesList.tsx:info", err));
  }, []);

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

  // Защита от гонок: применяем результат только последнего запроса
  // (быстрые клики по страницам/фильтрам — побеждает тот, кто запрошен
  // последним, а не тот, кто позже разрешился).
  const vacanciesSeqRef = useRef(0);

  const loadVacancies = async () => {
    const seq = ++vacanciesSeqRef.current;
    setLoading(true);
    setError(null);
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

      const data = await request<VacanciesResponse>(`/api/vacancies?${params}`);
      if (seq !== vacanciesSeqRef.current || !mountedRef.current) return;

      setVacancies(data.items);
      setTotal(data.total);

      const cities = Array.from(new Set(data.items.map(v => v.area)))
        .filter(city => city && city !== "Не указано")
        .sort();
      setAvailableCities(cities);
    } catch (err: any) {
      if (seq !== vacanciesSeqRef.current || !mountedRef.current) return;
      setError(err.message);
    } finally {
      if (seq === vacanciesSeqRef.current && mountedRef.current) setLoading(false);
    }
  };

  const refreshVacancies = async () => {
    // Общий счётчик с loadVacancies: результаты не перебивают друг друга.
    const seq = ++vacanciesSeqRef.current;
    try {
      const offset = (currentPage - 1) * itemsPerPage;
      const params = new URLSearchParams({ limit: itemsPerPage.toString(), offset: offset.toString() });
      if (applied.experience && applied.experience !== "all") params.append("experience", applied.experience);
      if (applied.city && applied.city !== "all") params.append("region", applied.city);
      if (applied.search.trim()) params.append("search", applied.search.trim());
      if (applied.months) params.append("months", applied.months.toString());
      if (applied.date_from) params.append("date_from", applied.date_from);
      if (applied.date_to) params.append("date_to", applied.date_to);
      const data = await request<VacanciesResponse>(`/api/vacancies?${params}`);
      if (seq !== vacanciesSeqRef.current || !mountedRef.current) return;
      setVacancies(data.items);
      setTotal(data.total);
    } catch (err) { reportError("app/components/VacanciesList.tsx:265", err); }
    try {
      const info = await request<VacancyInfo>("/api/vacancies/info");
      if (seq !== vacanciesSeqRef.current || !mountedRef.current) return;
      setVacancyInfo(info);
    } catch (err) { reportError("app/components/VacanciesList.tsx:269", err); }
  };

  const applyFilters = (over: Partial<AppliedFilters> = {}) => {
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

  // Отдаваемый сервером список уже отфильтрован/отсортирован под текущие
  // фильтры: мемоизируем производные (склейка городов и массив строк),
  // чтобы каждый ререндер не пересобирал их заново.
  const allCityOptions = useMemo(
    () => [...new Set([...HH_REGIONS, ...availableCities])],
    [availableCities],
  );

  // Стабильный колбэк + useMemo на массив строк: вместе с memo(VacancyCard)
  // исключает перерисовку карточек при ререндерах родителя.
  const openVacancy = useCallback((v: Vacancy) => {
    setDrawerVacancy(v);
  }, []);
  const vacancyRows = useMemo(
    () =>
      vacancies.map((vacancy) => (
        <MemoVacancyCard
          key={vacancy.id}
          vacancy={vacancy}
          variant={viewMode === "list" ? "list" : "grid"}
          onOpen={openVacancy}
        />
      )),
    [vacancies, viewMode, openVacancy],
  );

  // Рисуем не весь массив, а окно: сначала PAGE_SIZE карточек, дальше -
  // порциями через «Показать ещё» или прокрутку к sentinel (IntersectionObserver).
  // При новой выборке (фильтры/поиск/страница) окно само сбрасывается до PAGE_SIZE.
  const { visible, hasMore, loadMore, sentinelRef } = useIncrementalList(vacancies, PAGE_SIZE);
  const visibleRows = useMemo(() => vacancyRows.slice(0, visible), [vacancyRows, visible]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <PageHeader
        eyebrow="Рынок труда"
        title="Вакансии с hh.ru"
        description={
          vacancyInfo ? (
            <span className="flex flex-wrap items-center gap-2">
              {vacancyInfo.date_range && (
                <Badge tone="info">Записи с {fmtDateRU(vacancyInfo.date_range.from)}</Badge>
              )}
              <Badge tone="primary">всего вакансий собрано: {vacancyInfo.count.toLocaleString("ru-RU")}</Badge>
            </span>
          ) : undefined
        }
      />

      {/* Pipeline trigger / settings panel */}
      {canRunPipeline && (showPipelineSetup && (!pipelineStep || pipelineStep.status !== "running") ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <Card className="border-0 bg-muted shadow-sm dark:bg-card">
            <div className="flex items-center justify-between gap-4 border-b border-border px-5 pt-5 pb-3">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary p-2">
                  <Rocket className="size-5 text-primary-foreground" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Настройки сбора вакансий</h3>
                  <p className="text-sm text-muted-foreground">Выберите профессию и города для поиска</p>
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setShowPipelineSetup(false)} title="Свернуть настройки" aria-label="Свернуть настройки" className="size-9 shrink-0 p-0">
                <ChevronUp className="size-4" />
              </Button>
            </div>
            <CardBody className="space-y-5 p-6">
              {/* Profession */}
              <div className="space-y-2">
                <label className="text-sm font-semibold text-foreground dark:text-muted-foreground">
                  Профессия
                </label>
                <input
                  value={pipelineProfession}
                  onChange={(e) => setPipelineProfession(e.target.value)}
                  placeholder="IT-специалист / Data Scientist / Все"
                  disabled={!cityMode}
                  className={inputClass}
                />
                <p className="text-xs text-muted-foreground">{!cityMode ? "Весь рынок - поиск по всем IT-профессиям" : "Оставьте пустым для поиска по всем профессиям"}</p>
              </div>

              {/* Search params */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-foreground dark:text-muted-foreground">
                    Страниц
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={pipelineMaxPagesLocal}
                    onChange={(e) => setPipelineMaxPagesLocal(Math.max(1, Math.min(20, Number(e.target.value) || 20)))}
                    className={inputClass}
                  />
                  <p className="text-xs text-muted-foreground">1-20 (по 100 вакансий на страницу)</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-foreground dark:text-muted-foreground">
                    Период (дней)
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={365}
                    value={pipelinePeriodLocal}
                    onChange={(e) => setPipelinePeriodLocal(Math.max(1, Math.min(365, Number(e.target.value) || 30)))}
                    className={inputClass}
                  />
                  <p className="text-xs text-muted-foreground">1-365, по умолч. 30</p>
                </div>
              </div>

              {/* Region mode toggle */}
              <div className="flex gap-2">
                <Button
                  variant={!cityMode ? "primary" : "secondary"}
                  onClick={() => { setCityMode(false); setSelectedCities([]); setPipelineRegion("0"); setPipelineProfession(""); }}
                  aria-pressed={!cityMode}
                  className="h-10 flex-1 gap-2"
                >
                  <Globe className="size-4" />
                  Весь рынок
                </Button>
                <Button
                  variant={cityMode ? "primary" : "secondary"}
                  onClick={() => { setCityMode(true); if (selectedCities.length === 0) setSelectedCities([...HH_REGIONS]); }}
                  aria-pressed={cityMode}
                  className="h-10 flex-1 gap-2"
                >
                  <MapPin className="size-4" />
                  Выбрать города
                </Button>
              </div>

              {/* City selection */}
              {cityMode && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-semibold text-foreground dark:text-muted-foreground">
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
                  <div className="max-h-48 overflow-y-auto border border-border dark:border-border rounded-lg p-2 grid grid-cols-2 sm:grid-cols-3 gap-1">
                    {HH_REGIONS.map((city) => (
                      <label
                        key={city}
                        className="flex items-center gap-2 px-2 py-1.5 rounded-md text-sm cursor-pointer hover:bg-muted dark:hover:bg-muted select-none"
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
                          className="rounded border-border-strong dark:border-border-strong"
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
                    variant="secondary"
                    onClick={() => setShowAllMarketInfo(!showAllMarketInfo)}
                    className="h-9 w-full gap-2 text-sm"
                  >
                    <ChevronDown className={`size-4 transition-transform ${showAllMarketInfo ? "rotate-180" : ""}`} />
                    Показать список городов и вакансий для сбора
                  </Button>
                  {showAllMarketInfo && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      className="border border-border bg-muted rounded-lg p-4 space-y-3"
                    >
                      <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                        <Globe className="size-4" />
                        Поиск по всему рынку
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Будут собраны вакансии по всем IT-направлениям: Data Scientist, ML Engineer, Python/Java/Fullstack/Frontend/Backend Developer, DevOps, QA, Security, SRE, Mobile Dev, Analyst, Architect, Team Lead, UX/UI Designer, Game Dev и другим
                      </p>
                      <div className="space-y-2">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="text-xs font-medium text-foreground">Города</span>
                          <Badge>{HH_REGIONS.length}</Badge>
                          {selectedCities.length > 0 && (
                            <Badge tone="primary">выбрано: {selectedCities.length}</Badge>
                          )}
                        </div>
                        <input
                          type="text"
                          value={cityQuery}
                          onChange={(e) => setCityQuery(e.target.value)}
                          placeholder="Найти город..."
                          aria-label="Поиск города для сбора"
                          className={cn(inputClass, "h-8 text-xs")}
                        />
                        {(() => {
                          const q = cityQuery.trim().toLowerCase();
                          const filtered = HH_REGIONS.filter((c) => !q || c.toLowerCase().includes(q));
                          if (filtered.length === 0) {
                            return <EmptyState icon={<Search className="size-4" />} title="Ничего не найдено" className="py-6" />;
                          }
                          return (
                            <div
                              role="group"
                              aria-label="Города для сбора"
                              className="max-h-56 overflow-y-auto rounded-lg border border-border bg-card p-2"
                            >
                              <div className="flex flex-wrap gap-1.5 p-1">
                                {filtered.map((c) => {
                                  const on = selectedCities.includes(c);
                                  return (
                                    <Button
                                      key={c}
                                      type="button"
                                      size="sm"
                                      variant={on ? "primary" : "secondary"}
                                      aria-pressed={on}
                                      title={on ? "Убрать из выборки" : "Добавить к выборке"}
                                      onClick={() =>
                                        setSelectedCities((prev) => (on ? prev.filter((x) => x !== c) : [...prev, c]))
                                      }
                                    >
                                      {c}
                                    </Button>
                                  );
                                })}
                              </div>
                            </div>
                          );
                        })()}
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
                  className="flex-1 h-11 gap-2 hover:bg-primary/90 hover:brightness-100"
                >
                  {pipelineLoading ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Rocket className="size-4" />
                  )}
                  Запустить сбор
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => { setShowPipelineSetup(false); setSelectedCities([]); setCityMode(false); setPipelineProfession(""); }}
                  className="h-11"
                >
                  Отмена
                </Button>
              </div>
            </CardBody>
          </Card>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.07 }}
        >
          <Card className="border-0 bg-muted shadow-sm dark:bg-card">
            <CardBody className="p-4">
              <div className="flex flex-col md:flex-row md:flex-wrap items-start md:items-center gap-3">
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className="p-2 bg-primary rounded-lg shrink-0">
                    <Rocket className="size-5 text-primary-foreground" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-foreground dark:text-muted-foreground">
                      Сбор вакансий
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Загрузить актуальные вакансии с hh.ru для всех профилей
                    </p>
                  </div>
                </div>
                <Button
                  onClick={() => setShowPipelineSetup(true)}
                  disabled={pipelineLoading}
                  className="h-9 gap-2 px-6 shadow-xs has-[>svg]:px-6 w-full md:w-auto hover:bg-primary/90 hover:brightness-100"
                  size="sm"
                >
                  {pipelineLoading ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Rocket className="size-4" />
                  )}
                  Собрать вакансии
                </Button>
              </div>
            </CardBody>
          </Card>
        </motion.div>
      ))}

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card>
          <div className="flex items-center justify-between gap-4 border-b border-border/50 bg-muted px-5 pt-5 pb-3 dark:bg-card">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-primary p-2">
                <Filter className="size-5 text-primary-foreground" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-foreground">Фильтры и поиск</h3>
                <p className="text-xs text-muted-foreground">Настройте параметры для поиска вакансий</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setFilterOpen(true)}
                className="gap-2"
              >
                <Filter className="size-4" />
                Фильтры
                {activeFilterCount > 0 && (
                  <span className="ml-1 rounded-full bg-primary px-1.5 py-0.5 text-xs font-semibold text-primary-foreground">
                    {activeFilterCount}
                  </span>
                )}
              </Button>
              <Button
                variant={viewMode === "grid" ? "primary" : "secondary"}
                size="sm"
                onClick={() => setViewMode("grid")}
                aria-label="Отображение: сетка"
                aria-pressed={viewMode === "grid"}
                className="size-9 p-0"
              >
                <LayoutGrid className="size-4" />
              </Button>
              <Button
                variant={viewMode === "list" ? "primary" : "secondary"}
                size="sm"
                onClick={() => setViewMode("list")}
                aria-label="Отображение: список"
                aria-pressed={viewMode === "list"}
                className="size-9 p-0"
              >
                <List className="size-4" />
              </Button>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Фильтры выезжают справа */}
      <VacancyFiltersDrawer
        open={filterOpen}
        onClose={() => setFilterOpen(false)}
        activeFilterCount={activeFilterCount}
        searchQuery={searchQuery}
        onSearchQueryChange={setSearchQuery}
        onSearchKeyPress={handleSearchKeyPress}
        onSearch={handleSearch}
        loading={loading}
        experienceFilter={experienceFilter}
        onExperienceChange={setExperienceFilter}
        cityFilter={cityFilter}
        onCityChange={setCityFilter}
        allCityOptions={allCityOptions}
        monthsFilter={monthsFilter}
        onMonthsChange={setMonthsFilter}
        dateFrom={dateFrom}
        onDateFromChange={setDateFrom}
        dateTo={dateTo}
        onDateToChange={setDateTo}
        vacancyInfo={vacancyInfo}
        applied={applied}
        onApplyFilters={applyFilters}
        total={total}
        onClearFilters={clearFilters}
      />

      {/* Vacancies Grid */}
      {loading ? (
        <div
          role="status"
          aria-live="polite"
          className="grid grid-cols-1 items-stretch gap-4 lg:grid-cols-2"
        >
          <span className="sr-only">Загрузка вакансий...</span>
          {Array.from({ length: 6 }, (_, i) => (
            <Card key={`vacancy-skeleton-${i}`} aria-hidden="true">
              <CardBody className="space-y-3 p-6">
                <div className="flex items-start gap-3">
                  <Skeleton className="size-11 shrink-0 rounded-lg" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-3 w-1/2" />
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <Skeleton className="h-5 w-14 rounded-full" />
                  <Skeleton className="h-5 w-16 rounded-full" />
                  <Skeleton className="h-5 w-12 rounded-full" />
                </div>
                <div className="flex gap-2">
                  <Skeleton className="h-8 flex-1" />
                  <Skeleton className="h-8 w-8" />
                </div>
              </CardBody>
            </Card>
          ))}
        </div>
      ) : error ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="space-y-4">
            <ErrorState
              title="Ошибка загрузки"
              description={error}
              onRetry={loadVacancies}
            />
            {vacancyInfo?.load_error?.startsWith("corrupted:") ? (
              <div className="p-4 rounded-lg border border-warning/30 bg-warning-soft text-left max-w-lg mx-auto">
                <p className="text-sm text-warning">
                  <AlertCircle className="size-4 inline mr-1" />
                  <strong>Файл вакансий повреждён.</strong> Файл существует (<code className="text-xs bg-warning-soft/50 px-1 rounded">{vacancyInfo.file_modified ?? "неизвестно"}</code>), но не может быть прочитан.
                </p>
                <p className="text-xs text-warning/80 mt-2">
                  Попробуйте запустить повторный сбор вакансий - файлы будут перезаписаны.
                </p>
              </div>
            ) : !vacancyInfo?.file_modified ? (
              <div className="p-4 rounded-lg border border-warning/30 bg-warning-soft text-left max-w-lg mx-auto">
                <p className="text-sm text-warning">
                  <Database className="size-4 inline mr-1" />
                  <strong>Вакансии не собраны.</strong> Нажмите кнопку <strong>«Собрать вакансии»</strong> выше на этой странице.
                </p>
                <p className="text-xs text-warning/80 mt-2">
                  После сбора вакансий данные кэшируются. Если вы уже запускали сбор - проверьте, что бэкенд запущен.
                </p>
              </div>
            ) : (
              <div className="p-4 rounded-lg border border-destructive/30 bg-destructive/10 text-left max-w-lg mx-auto">
                <p className="text-sm text-destructive">
                  <AlertCircle className="size-4 inline mr-1" />
                  <strong>Не удалось загрузить данные из файла.</strong> Файл существует, но возникла ошибка при обработке.
                </p>
                {vacancyInfo?.load_error && (
                  <p className="text-xs text-destructive mt-1 font-mono">{vacancyInfo.load_error}</p>
                )}
                <p className="text-xs text-destructive/80 mt-2">
                  Попробуйте перезапустить сервер или запустить повторный сбор.
                </p>
              </div>
            )}
          </div>
        </motion.div>
      ) : vacancies.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="space-y-4">
            <EmptyState
              icon={<Database className="size-6" />}
              title="Вакансии не найдены"
              description="По вашему запросу ничего не найдено. Измените фильтры или запустите сбор заново."
            />
            <div className="p-4 rounded-lg border border-primary/30 bg-primary-soft text-left max-w-lg mx-auto">
              <p className="text-sm text-primary-soft-foreground">
                <Database className="size-4 inline mr-1" />
                <strong>Если вакансии ещё не собраны</strong> - нажмите кнопку <strong>«Собрать вакансии»</strong> выше на этой странице.
              </p>
              <ul className="mt-2 text-xs text-primary-soft-foreground/80 space-y-1 list-disc list-inside">
                <li>После нажатия запустится полный цикл сбора (10-15 минут)</li>
                <li>Прогресс будет отображаться на этой же странице</li>
                <li>После завершения данные обновятся автоматически</li>
              </ul>
            </div>
          </div>
        </motion.div>
      ) : (
        <>
          {/* Count + Export */}
          <VacancyStatsBar total={total} visible={visible} canRunPipeline={canRunPipeline} applied={applied} />

          <div
            className={
              viewMode === "grid"
                ? "grid grid-cols-1 items-stretch gap-4 lg:grid-cols-2"
                : "flex flex-col gap-2"
            }
          >
            {visibleRows}
          </div>

          {/* Инкрементальная догрузка: sentinel ловит прокрутку через
              IntersectionObserver, кнопка - ручной путь (клавиатура/мышь). */}
          {hasMore && (
            <>
              <div ref={sentinelRef} aria-hidden="true" className="h-px w-full" />
              <div className="flex justify-center">
                <Button
                  variant="secondary"
                  onClick={loadMore}
                  aria-label="Показать ещё вакансии"
                  className="gap-2"
                >
                  <ChevronDown className="size-4" />
                  Показать ещё
                  <span className="font-normal text-muted-foreground">
                    ({visible} из {vacancies.length})
                  </span>
                </Button>
              </div>
            </>
          )}

          <VacancyDetailPanel vacancy={drawerVacancy} onClose={() => setDrawerVacancy(null)} />

          {/* Pagination */}
          {totalPages > 1 && (
            <VacancyPagination
              currentPage={currentPage}
              totalPages={totalPages}
              loading={loading}
              onPageChange={setCurrentPage}
            />
          )}
        </>
      )}
    </div>
  );
}
