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
import { matchesQuery, type Publication } from "./publicationsFilter";

interface ListResponse {
  items: Publication[]; total: number; page: number; page_size: number; num_pages: number;
}
interface TypesResponse {
  types: { id: number; name: string | null; count: number }[];
  exact?: boolean;
  total?: number;
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

  // Живой поиск по уже загруженным записям: сеть не трогаем.
  useEffect(() => {
    const t = setTimeout(() => setApplied(query.trim().toLowerCase()), 300);
    return () => clearTimeout(t);
  }, [query]);

  // Типы публикаций: счётчики точные, поэтому запрашиваем один раз.
  useEffect(() => {
    const ctrl = new AbortController();
    api("/publications/types", { signal: ctrl.signal })
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
    api(endpoint, { signal: ctrl.signal })
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
    api(endpoint, { signal: ctrl.signal })
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
    () => (applied ? loaded.filter((p) => matchesQuery(p, applied)) : loaded),
    [loaded, applied],
  );
  const totalFilteredPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalFilteredPages);
  const visibleItems = filtered.slice((safePage - 1) * pageSize, safePage * pageSize);
  const canLoadMore = numPages > 0 && nextPage <= numPages && !loadingMore;

  const resetAll = () => {
    setQuery("");
    setApplied("");
    setTypeId("");
    setPage(1);
    setReload((n) => n + 1);
  };

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
          </div>

          <Alert className="border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-200">
            <Info className="size-4" />
            <AlertTitle>Как работает поиск</AlertTitle>
            <AlertDescription className="text-blue-700 dark:text-blue-300">
              <p>
                У hub.sfedu.ru нет серверного поиска, поэтому фильтр работает по уже загруженным
                записям: <strong>{(applied ? filtered.length : loaded.length).toLocaleString("ru-RU")}</strong>
                {" "}из {total.toLocaleString("ru-RU")}
                {applied ? ` (совпадений среди ${loaded.length.toLocaleString("ru-RU")} загруженных)` : " загруженных"}.
                Чтобы расширить охват, нажмите «Загрузить ещё».
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
                  ? `Совпадений: ${filtered.length.toLocaleString("ru-RU")} среди ${loaded.length.toLocaleString("ru-RU")} загруженных`
                  : `Загружено ${loaded.length.toLocaleString("ru-RU")} из ${total.toLocaleString("ru-RU")}`}
                {typesExact ? "" : " · счётчики типов приблизительные"}
              </p>
              {visibleItems.length === 0 ? (
                <EmptyState
                  icon={<Info className="size-8" />}
                  title="Среди загруженных записей совпадений нет"
                  description={`Просмотрено ${loaded.length.toLocaleString("ru-RU")} из ${total.toLocaleString("ru-RU")} записей. Публикация может быть в незагруженной части каталога.`}
                  action={
                    <Button variant="outline" size="sm" onClick={loadMore} disabled={!canLoadMore}>
                      {loadingMore ? <Loader2 className="size-4 animate-spin" /> : null}Загрузить ещё
                    </Button>
                  }
                />
              ) : (
                <>
                  <ul role="list" className="space-y-3">
                    {visibleItems.map((p) => <li key={p.id}><PublicationCard pub={p} onOpen={setDetailId} /></li>)}
                  </ul>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <PaginationBar page={safePage} totalPages={totalFilteredPages} pageSize={pageSize}
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
