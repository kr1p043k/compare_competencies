import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Info, Loader2, RotateCcw, Search } from "lucide-react";

import { EmptyState, ErrorState, PageHeader, Skeleton } from "../../design";
import { api } from "../api";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { PaginationBar, PublicationCard, errorText, isAbort } from "./PublicationCard";
import { PublicationDetailDialog } from "./PublicationDetailDialog";
import { type Publication } from "./publicationsFilter";

interface ListResponse {
  items: Publication[]; total: number; page: number; page_size: number; num_pages: number;
}
interface SearchResponse {
  query: string;
  matches: Publication[];
  total: number;
  scanned_pages: number;
  truncated: boolean;
  total_scanned: number;
}
interface TypesResponse {
  types: { id: number; name: string | null; count: number }[];
  exact?: boolean;
  total?: number;
}
interface DeepSearchProgress {
  scanned_pages: number;
  total_pages: number;
  pct: number;
}
interface DeepSearchStatus {
  job_id: string;
  status: "running" | "done" | "error" | "cancelled";
  progress: DeepSearchProgress;
  matches: Publication[];
  total: number;
  total_scanned: number;
  truncated: boolean;
  elapsed_ms: number;
  error: string | null;
}
interface DeepSearchStart {
  job_id: string;
  status: string;
}

export function PublicationsTab() {
  const [query, setQuery] = useState("");
  const [applied, setApplied] = useState("");
  const [typeId, setTypeId] = useState("");
  const [types, setTypes] = useState<TypesResponse["types"]>([]);
  const [typesExact, setTypesExact] = useState(true);
  const [loaded, setLoaded] = useState<Publication[]>([]);
  const [total, setTotal] = useState(0);
  const [numPages, setNumPages] = useState(0);
  const [nextPage, setNextPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [detailId, setDetailId] = useState<number | null>(null);
  // Серверный поиск: у hub нет полнотекстового индекса, поэтому сервер читает
  // первые N страниц корпуса и ищет по ним. Результат приходит вместе с
  // честной метрикой охвата (total_scanned / truncated).
  const [searchData, setSearchData] = useState<SearchResponse | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  // Глубокий поиск — фоновая задача: обычный запрос читает лишь первые
  // страницы корпуса, поэтому явная кнопка запускает скан на сотни страниц
  // с прогрессом и возможностью отмены.
  const [deepJob, setDeepJob] = useState<DeepSearchStatus | null>(null);
  const [deepStarting, setDeepStarting] = useState(false);
  const [deepError, setDeepError] = useState("");

  // Живой поиск по уже загруженным записям: сеть не трогаем.
  useEffect(() => {
    const t = setTimeout(() => setApplied(query.trim().toLowerCase()), 300);
    return () => clearTimeout(t);
  }, [query]);

  // Серверный поиск вместо фильтрации одной загруженной страницы.
  // Раньше запрос искал только среди 20 загруженных записей из 133 902, и
  // любое слово вне первой страницы давало пустую выдачу.
  useEffect(() => {
    if (!applied || applied.length < 2) {
      setSearchData(null);
      setSearchError("");
      return;
    }
    const ctrl = new AbortController();
    setSearching(true);
    setSearchError("");
    const params = new URLSearchParams({ q: applied, max_pages: "10", limit: "200" });
    if (typeId) params.set("type_id", typeId);
    api<SearchResponse>(`/publications/search?${params}`, { signal: ctrl.signal })
      .then((d: SearchResponse) => setSearchData(d))
      .catch((e: unknown) => {
        if (isAbort(e)) return;
        setSearchError(errorText(e));
        setSearchData(null);
      })
      .finally(() => { if (!ctrl.signal.aborted) setSearching(false); });
    return () => ctrl.abort();
  }, [applied, typeId]);

  // Типы публикаций: счётчики точные, поэтому запрашиваем один раз.
  useEffect(() => {
    const ctrl = new AbortController();
    api<TypesResponse>("/publications/types", { signal: ctrl.signal })
      .then((d: TypesResponse) => {
        setTypes(d.types ?? []);
        setTypesExact(d.exact !== false);
      })
      .catch(() => { /* фильтр по типу не критичен */ });
    return () => ctrl.abort();
  }, []);

  // Первая страница каталога: смена типа или сброс начинают накопление заново.
  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    setError("");
    setLoaded([]);
    setPage(1);
    const endpoint = `/publications?page=1&page_size=${pageSize}${typeId ? `&type_id=${typeId}` : ""}`;
    api<ListResponse>(endpoint, { signal: ctrl.signal })
      .then((d: ListResponse) => {
        setLoaded(d.items);
        setTotal(d.total);
        setNumPages(d.num_pages);
        setNextPage(2);
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (isAbort(e)) return;
        setError(errorText(e));
        setLoading(false);
      });
    return () => ctrl.abort();
  }, [typeId, pageSize, reload]);

  const loadMoreCtrl = useRef<AbortController | null>(null);
  // Смена типа или размера страницы отменяет догрузку: иначе запоздавший
  // ответ приклеит записи старого фильтра к новому списку.
  useEffect(() => () => loadMoreCtrl.current?.abort(), [typeId, pageSize]);

  const loadMore = useCallback(() => {
    if (loadingMore || numPages === 0 || nextPage > numPages) return;
    loadMoreCtrl.current?.abort();
    const ctrl = new AbortController();
    loadMoreCtrl.current = ctrl;
    setLoadingMore(true);
    const endpoint = `/publications?page=${nextPage}&page_size=${pageSize}${typeId ? `&type_id=${typeId}` : ""}`;
    api<ListResponse>(endpoint, { signal: ctrl.signal })
      .then((d: ListResponse) => {
        if (ctrl.signal.aborted) return;
        setLoaded((prev) => {
          const seen = new Set(prev.map((p) => p.id));
          return [...prev, ...d.items.filter((p) => !seen.has(p.id))];
        });
        setNextPage((n) => n + 1);
        setTotal(d.total);
        setNumPages(d.num_pages);
      })
      .catch((e: unknown) => {
        if (isAbort(e)) return;
        setError(errorText(e));
      })
      .finally(() => {
        if (loadMoreCtrl.current === ctrl) setLoadingMore(false);
      });
  }, [loadingMore, numPages, nextPage, pageSize, typeId]);

  const filtered = useMemo(
    () => (applied ? (searchData?.matches ?? []) : loaded),
    [loaded, applied, searchData],
  );
  // Ошибку серверного поиска не глотаем: иначе пустой список выглядит как
  // «совпадений нет», хотя причина другая (лимит запросов, hub недоступен).
  useEffect(() => {
    if (searchError) setError(searchError);
  }, [searchError]);
  const canLoadMore = !applied && numPages > 0 && nextPage <= numPages && !loadingMore;

  const resetAll = () => {
    setQuery("");
    setApplied("");
    setTypeId("");
    setPage(1);
    setReload((n) => n + 1);
    setDeepJob(null);
    setDeepError("");
  };

  // Запуск глубокого поиска. Сервер отвечает сразу (202) и сканирует корпус
  // в фоне: полный проход не укладывается в обычный запрос.
  const startDeepSearch = useCallback(async () => {
    const q = applied || query.trim();
    if (q.length < 2) return;
    setDeepStarting(true);
    setDeepError("");
    setDeepJob(null);
    try {
      const params = new URLSearchParams({ q, max_pages: "500" });
      if (typeId) params.set("type_id", typeId);
      const started = await api<DeepSearchStart>(
        `/publications/deep-search?${params}`,
        { method: "POST" },
      );
      setDeepJob({
        job_id: started.job_id,
        status: "running",
        progress: { scanned_pages: 0, total_pages: 0, pct: 0 },
        matches: [],
        total: 0,
        total_scanned: 0,
        truncated: false,
        elapsed_ms: 0,
        error: null,
      });
    } catch (e: unknown) {
      setDeepError(errorText(e));
    } finally {
      setDeepStarting(false);
    }
  }, [applied, query, typeId]);

  const cancelDeepSearch = useCallback(async () => {
    if (!deepJob) return;
    try {
      await api(`/publications/deep-search/${deepJob.job_id}`, { method: "DELETE" });
    } catch {
      // Отмена best-effort: если задача уже завершилась, это не ошибка.
    }
  }, [deepJob]);

  // Опрос статуса, пока задача выполняется.
  useEffect(() => {
    if (!deepJob || deepJob.status !== "running") return;
    const jobId = deepJob.job_id;
    const id = setInterval(() => {
      api<DeepSearchStatus>(`/publications/deep-search/${jobId}`)
        .then((d) => setDeepJob(d))
        .catch((e: unknown) => {
          setDeepError(errorText(e));
          clearInterval(id);
        });
    }, 1500);
    return () => clearInterval(id);
  }, [deepJob]);

  const deepActive = deepJob !== null;
  const shownItems = deepActive ? (deepJob?.matches ?? []) : filtered;
  const shownPages = Math.max(1, Math.ceil(shownItems.length / pageSize));
  const shownSafePage = Math.min(page, shownPages);
  const visibleShown = shownItems.slice((shownSafePage - 1) * pageSize, shownSafePage * pageSize);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Публикации ЮФУ"
        description="Репозиторий hub.sfedu.ru: статьи, учебники, диссертации, патенты и студенческие работы"
      />
      <Card className="border border-gray-200 dark:border-slate-700 shadow-sm overflow-hidden">
        <CardContent className="p-4 sm:p-6 space-y-4">
          <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-gray-400 pointer-events-none" />
              <Input value={query} onChange={(e) => setQuery(e.target.value)}
                placeholder="Фильтр по названию, авторам, журналу, аннотации" aria-label="Фильтр публикаций"
                className="pl-9 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600" />
            </div>
            <Select value={typeId || "all"}
              onValueChange={(v) => setTypeId(v === "all" ? "" : v)}>
              <SelectTrigger className="sm:w-64 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600"
                aria-label="Фильтр по типу публикации">
                <SelectValue placeholder="Все типы" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Все типы</SelectItem>
                {types.map((t) => (
                  <SelectItem key={t.id} value={String(t.id)}>
                    {t.name ?? `Тип ${t.id}`} ({t.count.toLocaleString("ru-RU")})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline" onClick={resetAll}
              className="border-gray-300 dark:border-slate-600 text-gray-700 dark:text-slate-300 hover:bg-gray-50 dark:hover:bg-slate-800">
              <RotateCcw className="size-4" />Сбросить
            </Button>
            {/* Глубокий поиск: обычный поиск читает первые страницы корпуса,
                этот сканирует до 500 страниц в фоне и показывает прогресс. */}
            <Button
              onClick={startDeepSearch}
              disabled={deepStarting || query.trim().length < 2 || deepJob?.status === "running"}
              title="Просканировать значительно больше страниц корпуса (дольше)"
              className="bg-indigo-600 hover:bg-indigo-700 text-white"
            >
              {deepStarting ? <Loader2 className="size-4 animate-spin" /> : <Search className="size-4" />}
              Глубокий поиск
            </Button>
          </div>

          {deepError ? (
            <Alert className="border-red-200 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-200">
              <AlertTitle>Глубокий поиск не запустился</AlertTitle>
              <AlertDescription>{deepError}</AlertDescription>
            </Alert>
          ) : null}

          {deepJob ? (
            <div className="rounded-md border border-indigo-200 dark:border-indigo-800 bg-indigo-50/60 dark:bg-indigo-950/30 p-3 space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-indigo-800 dark:text-indigo-200">
                  {deepJob.status === "running"
                    ? "Глубокий поиск выполняется…"
                    : deepJob.status === "done"
                      ? "Глубокий поиск завершён"
                      : deepJob.status === "cancelled"
                        ? "Глубокий поиск отменён"
                        : "Глубокий поиск завершился с ошибкой"}
                </span>
                <span className="text-xs text-indigo-700 dark:text-indigo-300">
                  просмотрено {deepJob.progress.scanned_pages.toLocaleString("ru-RU")}
                  {deepJob.progress.total_pages > 0
                    ? ` из ${deepJob.progress.total_pages.toLocaleString("ru-RU")}`
                    : ""} страниц
                  {deepJob.progress.pct > 0 ? ` (${deepJob.progress.pct.toFixed(0)}%)` : ""}
                  {" · "}совпадений: {deepJob.total.toLocaleString("ru-RU")}
                </span>
              </div>
              <div className="h-2 w-full rounded bg-indigo-200 dark:bg-indigo-900 overflow-hidden">
                <div
                  className="h-full bg-indigo-600 transition-all"
                  style={{ width: `${Math.min(100, deepJob.progress.pct)}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-[11px] text-indigo-700 dark:text-indigo-300">
                <span>
                  Записей просмотрено: {deepJob.total_scanned.toLocaleString("ru-RU")}
                  {deepJob.truncated ? " (корпус не дочитан до конца)" : ""}
                </span>
                {deepJob.status === "running" ? (
                  <Button variant="outline" size="sm" onClick={cancelDeepSearch}>Отменить</Button>
                ) : (
                  <Button variant="outline" size="sm" onClick={() => { setDeepJob(null); setDeepError(""); }}>
                    Скрыть
                  </Button>
                )}
              </div>
              {deepJob.error ? (
                <p className="text-xs text-red-700 dark:text-red-300">{deepJob.error}</p>
              ) : null}
            </div>
          ) : null}

          <Alert className="border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-200">
            <Info className="size-4" />
            <AlertTitle>Как работает поиск</AlertTitle>
            <AlertDescription className="text-blue-700 dark:text-blue-300">
              <p>
                {applied ? (
                  searching ? (
                    <>Ищем по репозиторию…</>
                  ) : (
                    <>
                      Просмотрено{" "}
                      <strong>{(searchData?.total_scanned ?? 0).toLocaleString("ru-RU")}</strong>{" "}
                      записей из {total.toLocaleString("ru-RU")}
                      {searchData?.truncated ? " (это не весь корпус)" : ""}.
                      Совпадений: <strong>{filtered.length.toLocaleString("ru-RU")}</strong>.
                    </>
                  )
                ) : (
                  <>
                    Показано {loaded.length.toLocaleString("ru-RU")} из{" "}
                    {total.toLocaleString("ru-RU")} записей. Чтобы смотреть дальше,
                    нажмите «Загрузить ещё».
                  </>
                )}
              </p>
              <p className="mt-1 text-[11px] opacity-80">
                У hub.sfedu.ru нет полнотекстового индекса: поиск читает первые страницы
                корпуса, а не весь репозиторий. Уточняйте запрос или выберите тип —
                так совпадений в просмотренной части будет больше.
              </p>
            </AlertDescription>
          </Alert>

          {loading ? (
            <div className="space-y-3" role="status" aria-label="Загрузка публикаций">
              {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28 w-full rounded-xl" />)}
            </div>
          ) : error ? (
            <ErrorState
              title="Не удалось загрузить публикации"
              description={error}
              onRetry={() => setReload((n) => n + 1)}
            />
          ) : loaded.length === 0 ? (
            <EmptyState
              title="Публикаций нет"
              description="Попробуйте выбрать другой тип"
            />
          ) : (
            <div className="space-y-3">
              <p className="text-xs text-gray-500 dark:text-slate-400" aria-live="polite">
                {applied
                  ? `Совпадений: ${filtered.length.toLocaleString("ru-RU")} · просмотрено ${(searchData?.total_scanned ?? 0).toLocaleString("ru-RU")} из ${total.toLocaleString("ru-RU")}`
                  : `Загружено ${loaded.length.toLocaleString("ru-RU")} из ${total.toLocaleString("ru-RU")}`}
                {typesExact ? "" : " · счётчики типов приблизительные"}
              </p>
              {visibleShown.length === 0 ? (
                <EmptyState
                  icon={<Info className="size-8" />}
                  title="Совпадений не найдено"
                  description={`Просмотрено ${(deepActive ? deepJob?.total_scanned ?? 0 : searchData?.total_scanned ?? 0).toLocaleString("ru-RU")} из ${total.toLocaleString("ru-RU")} записей. Уточните запрос или выберите тип публикации.`}
                  action={
                    !deepActive ? (
                      <Button variant="outline" size="sm" onClick={loadMore} disabled={!canLoadMore}>
                        {loadingMore ? <Loader2 className="size-4 animate-spin" /> : null}Загрузить ещё
                      </Button>
                    ) : undefined
                  }
                />
              ) : (
                <>
                  <ul role="list" className="space-y-3">
                    {visibleShown.map((p) => <li key={p.id}><PublicationCard pub={p} onOpen={setDetailId} /></li>)}
                  </ul>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <PaginationBar page={shownSafePage} totalPages={shownPages} pageSize={pageSize}
                      onPage={setPage} onPageSize={(n) => { setPageSize(n); setPage(1); }} />
                    <Button variant="outline" size="sm" onClick={loadMore} disabled={!canLoadMore}
                      className="border-gray-300 dark:border-slate-600">
                      {loadingMore ? <Loader2 className="size-4 animate-spin" /> : null}
                      Загрузить ещё {canLoadMore ? `(${Math.max(1, Math.min(pageSize, total - loaded.length))} шт.)` : ""}
                    </Button>
                  </div>
                </>
              )}
            </div>
          )}
        </CardContent>
      </Card>
      {detailId !== null ? <PublicationDetailDialog id={detailId} onClose={() => setDetailId(null)} /> : null}
    </div>
  );
}
