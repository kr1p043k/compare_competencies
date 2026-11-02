import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { Skeleton } from "../../design";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "./ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { AlertCircle, BookOpen, ChevronLeft, ChevronRight, ExternalLink, FileText, Info, Loader2, RotateCcw, Search } from "lucide-react";

interface Publication {
  id: number; title: string | null; authors: string[];
  type_id: number | null; type_name: string | null; signature: string | null;
  doi: string | null; isbn: string | null; track_id: string | null;
  created_at: string | null; year: number | null;
  academic_subject: string[]; education_program: string[];
  files: { url: string; filename: string | null; size: number | null; content_type: string | null }[];
  url: string | null; creator_username: string | null;
}

interface ListResponse {
  items: Publication[]; total: number; page: number; page_size: number; num_pages: number;
}
interface SearchResponse {
  query: string; matches: Publication[]; total: number;
  scanned_pages: number; truncated: boolean; total_scanned: number;
}
interface TypesResponse { types: { id: number; name: string | null; count: number }[] }

const TYPE_BADGE = "bg-blue-100 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300";
const OUTLINE_BADGE = "text-gray-600 dark:text-slate-300 border-gray-300 dark:border-slate-600";
const EXTERNAL_LINK = "text-blue-600 dark:text-blue-400 hover:underline";

const isAbort = (e: unknown): boolean => e instanceof DOMException && e.name === "AbortError";
const RUSSIAN_ERRORS: Record<number, string> = {
  429: "Слишком много запросов к хабу. Подождите минуту и повторите.",
  502: "Сервис хаба отклонил запрос. Попробуйте позже.",
  503: "Сервис хаба недоступен. Попробуйте позже.",
  504: "Сервис хаба не ответил вовремя. Попробуйте позже.",
  422: "Некорректный запрос: проверьте параметры поиска.",
};
function errorText(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e);
  let detail = raw;
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown };
    if (parsed && typeof parsed.detail === "string") detail = parsed.detail;
  } catch { /* тело не JSON — работаем с сырым текстом */ }
  if (/too many requests|rate limit/i.test(detail)) return RUSSIAN_ERRORS[429];
  if (/internal server error/i.test(detail)) return "Внутренняя ошибка сервера. Попробуйте позже.";
  if (/failed to fetch|networkerror|load failed/i.test(detail)) return "Нет связи с сервером. Проверьте подключение.";
  return detail;
}

function formatSize(n: number | null): string {
  if (n === null || n < 0) return "—";
  if (n < 1024) return `${n} Б`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} КБ`;
  return `${(n / (1024 * 1024)).toFixed(1)} МБ`;
}
function formatDate(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleDateString("ru-RU");
}
const doiHref = (doi: string): string => (/^https?:\/\//i.test(doi) ? doi : `https://doi.org/${doi}`);
function PublicationCard({ pub, onOpen }: { pub: Publication; onOpen: (id: number) => void }) {
  const subjects = pub.academic_subject.slice(0, 3);
  const more = pub.academic_subject.length - subjects.length;
  return (
    <Card className="border border-gray-200 dark:border-slate-700 shadow-sm">
      <CardContent className="p-4 sm:p-5 space-y-2.5">
        <div className="flex flex-wrap items-center gap-2">
          {pub.type_name ? <Badge className={TYPE_BADGE}>{pub.type_name}</Badge> : null}
          {pub.year ? <span className="text-sm text-gray-500 dark:text-slate-400">{pub.year}</span> : null}
          <span className="ml-auto flex items-center gap-3 text-xs">
            {pub.doi ? <a href={doiHref(pub.doi)} target="_blank" rel="noreferrer" className={EXTERNAL_LINK}>DOI</a> : null}
            {pub.url ? (
              <a href={pub.url} target="_blank" rel="noreferrer" className={`inline-flex items-center gap-1 ${EXTERNAL_LINK}`}>
                Открыть в хабе<ExternalLink className="size-3.5" />
              </a>
            ) : null}
          </span>
        </div>
        <button type="button" onClick={() => onOpen(pub.id)}
          className="block text-left text-sm sm:text-base font-semibold text-gray-900 dark:text-slate-100 hover:text-blue-700 dark:hover:text-blue-300 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded">
          {pub.title ?? "Публикация без названия"}
        </button>
        {pub.authors.length > 0 ? <p className="text-sm text-gray-600 dark:text-slate-400">{pub.authors.join(", ")}</p> : null}
        {pub.academic_subject.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {subjects.map((s) => <Badge key={s} variant="outline" className={OUTLINE_BADGE}>{s}</Badge>)}
            {more > 0 ? (
              <Badge variant="outline" className="text-gray-500 dark:text-slate-400 border-gray-300 dark:border-slate-600">+{more}</Badge>
            ) : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function PaginationBar({ page, totalPages, pageSize, onPage, onPageSize }: {
  page: number; totalPages: number; pageSize: number; onPage: (p: number) => void; onPageSize: (n: number) => void;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => onPage(page - 1)} disabled={page <= 1}
          aria-label="Предыдущая страница" className="border-gray-300 dark:border-slate-600">
          <ChevronLeft className="size-4" />Назад
        </Button>
        <span className="text-sm text-gray-600 dark:text-slate-400" aria-live="polite">Стр. {page} из {totalPages}</span>
        <Button variant="outline" size="sm" onClick={() => onPage(page + 1)} disabled={page >= totalPages}
          aria-label="Следующая страница" className="border-gray-300 dark:border-slate-600">
          Вперёд<ChevronRight className="size-4" />
        </Button>
      </div>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-slate-400">
        На странице
        <Select value={String(pageSize)} onValueChange={(v) => onPageSize(Number(v))}>
          <SelectTrigger className="w-20 h-8 bg-white dark:bg-slate-950" aria-label="Размер страницы"><SelectValue /></SelectTrigger>
          <SelectContent>
            {[10, 20, 50].map((n) => <SelectItem key={n} value={String(n)}>{n}</SelectItem>)}
          </SelectContent>
        </Select>
      </label>
    </div>
  );
}

function DetailBody({ detail }: { detail: Publication }) {
  const rows: [string, string | null][] = [
    ["Год", detail.year ? String(detail.year) : null], ["Дата добавления", formatDate(detail.created_at)],
    ["Тип", detail.type_name], ["Подпись", detail.signature],
    ["ISBN", detail.isbn], ["Автор в хабе", detail.creator_username],
  ];
  const tagSections: [string, string[]][] = [
    ["Дисциплины", detail.academic_subject], ["Образовательные программы", detail.education_program],
  ];
  return (
    <div className="space-y-4">
      {detail.type_name ? <Badge className={TYPE_BADGE}>{detail.type_name}</Badge> : null}
      <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3">
        {rows.map(([label, value]) => value ? (
          <div key={label}>
            <dt className="text-xs text-gray-500 dark:text-slate-400">{label}</dt>
            <dd className="mt-0.5 text-sm break-words">{value}</dd>
          </div>
        ) : null)}
        {detail.doi ? (
          <div className="sm:col-span-2">
            <dt className="text-xs text-gray-500 dark:text-slate-400">DOI</dt>
            <dd className="mt-0.5 text-sm"><a href={doiHref(detail.doi)} target="_blank" rel="noreferrer" className={`break-all ${EXTERNAL_LINK}`}>{detail.doi}</a></dd>
          </div>
        ) : null}
      </dl>
      {tagSections.map(([label, list]) => list.length > 0 ? (
        <div key={label}>
          <h3 className="text-sm font-medium text-gray-900 dark:text-slate-100">{label}</h3>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {list.map((s) => <Badge key={s} variant="outline" className={OUTLINE_BADGE}>{s}</Badge>)}
          </div>
        </div>
      ) : null)}
      <div className="h-px bg-gray-200 dark:bg-slate-700" role="separator" />
      <div>
        <h3 className="text-sm font-medium text-gray-900 dark:text-slate-100">Файлы{detail.files.length > 0 ? ` (${detail.files.length})` : ""}</h3>
        {detail.files.length === 0 ? (
          <p className="mt-1.5 text-sm text-gray-500 dark:text-slate-400">Файлов нет</p>
        ) : (
          <ul role="list" className="mt-1.5 space-y-2">
            {detail.files.map((f) => (
              <li key={f.url} className="flex items-center justify-between gap-3 rounded-lg border border-gray-200 dark:border-slate-700 px-3 py-2">
                <div className="flex items-center gap-2 min-w-0">
                  <FileText className="size-4 shrink-0 text-gray-400" />
                  <span className="truncate text-sm" title={f.filename ?? undefined}>{f.filename ?? "Файл"}</span>
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  <span className="text-xs text-gray-500 dark:text-slate-400">{formatSize(f.size)}</span>
                  <a href={f.url} target="_blank" rel="noreferrer" className={`inline-flex items-center gap-1 text-xs ${EXTERNAL_LINK}`}><ExternalLink className="size-3.5" />Скачать</a>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
      {detail.url ? (
        <Button asChild variant="outline" size="sm" className="border-gray-300 dark:border-slate-600">
          <a href={detail.url} target="_blank" rel="noreferrer"><ExternalLink className="size-3.5" />Открыть в хабе</a>
        </Button>
      ) : null}
    </div>
  );
}
function PublicationDetailDialog({ id, onClose }: { id: number; onClose: () => void }) {
  const [detail, setDetail] = useState<Publication | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    setError("");
    setDetail(null);
    api(`/publications/${id}`, { signal: ctrl.signal })
      .then((d: Publication) => { setDetail(d); setLoading(false); })
      .catch((e: unknown) => {
        if (isAbort(e)) return;
        setError(errorText(e)); setLoading(false);
      });
    return () => ctrl.abort();
  }, [id, reload]);

  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="sm:max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="pr-8 text-left">{detail?.title ?? "Публикация"}</DialogTitle>
          <DialogDescription>
            {detail && detail.authors.length > 0 ? detail.authors.join(", ") : "Карточка публикации хаба"}
          </DialogDescription>
        </DialogHeader>
        {loading ? (
          <div className="space-y-3" role="status" aria-label="Загрузка публикации">
            <Skeleton className="h-6 w-3/4" /><Skeleton className="h-4 w-1/2" /><Skeleton className="h-20 w-full" /><Skeleton className="h-20 w-full" />
          </div>
        ) : error ? (
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertTitle>Не удалось загрузить публикацию</AlertTitle>
            <AlertDescription>
              <p>{error}</p>
              <Button variant="outline" size="sm" onClick={() => setReload((n) => n + 1)} className="mt-2">Повторить</Button>
            </AlertDescription>
          </Alert>
        ) : detail ? <DetailBody detail={detail} /> : null}
      </DialogContent>
    </Dialog>
  );
}

export function PublicationsTab() {
  const [query, setQuery] = useState("");
  const [applied, setApplied] = useState("");
  const [typeId, setTypeId] = useState("");
  const [types, setTypes] = useState<TypesResponse["types"]>([]);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [items, setItems] = useState<Publication[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [searchMeta, setSearchMeta] = useState<{ pages: number; scanned: number; truncated: boolean } | null>(null);
  const [searchAll, setSearchAll] = useState<Publication[] | null>(null);
  const [detailId, setDetailId] = useState<number | null>(null);

  const isSearch = applied !== "";
  const SEARCH_LIMIT = 200;

  const applySearch = useCallback((q: string) => {
    const t = q.trim();
    if (t.length === 1) return; // сервер требует минимум 2 символа
    setApplied(t); setPage(1); setReload((n) => n + 1);
  }, []);

  // Живой поиск: применяем запрос через 400 мс после ввода.
  useEffect(() => {
    if (query.trim() === applied) return;
    const t = setTimeout(() => applySearch(query), 400);
    return () => clearTimeout(t);
  }, [query, applied, applySearch]);

  // Типы публикаций — один раз при монтировании.
  useEffect(() => {
    const ctrl = new AbortController();
    api("/publications/types", { signal: ctrl.signal })
      .then((d: TypesResponse) => setTypes(d.types ?? []))
      .catch(() => { /* фильтр по типу не критичен */ });
    return () => ctrl.abort();
  }, []);

  // Каталог: страница за страницей с сервера.
  useEffect(() => {
    if (isSearch) return;
    const ctrl = new AbortController();
    setLoading(true);
    setError("");
    const endpoint = `/publications?page=${page}&page_size=${pageSize}${typeId ? `&type_id=${typeId}` : ""}`;
    api(endpoint, { signal: ctrl.signal })
      .then((d: ListResponse) => {
        setItems(d.items);
        setTotal(d.total);
        setTotalPages(Math.max(1, d.num_pages));
        setSearchMeta(null);
        if (d.items.length === 0 && page > d.num_pages && d.num_pages > 0) setPage(d.num_pages);
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (isAbort(e)) return;
        setError(errorText(e)); setLoading(false);
      });
    return () => ctrl.abort();
  }, [isSearch, page, pageSize, typeId, reload]);

  // Поиск: один скан на запрос, листаем результат локально (иначе каждое
  // перелистывание = новый дорогой скан + риск 429).
  useEffect(() => {
    if (!isSearch) { setSearchAll(null); return; }
    const ctrl = new AbortController();
    setLoading(true);
    setError("");
    const endpoint = `/publications/search?q=${encodeURIComponent(applied)}&limit=${SEARCH_LIMIT}`
      + `${typeId ? `&type_id=${typeId}` : ""}`;
    api(endpoint, { signal: ctrl.signal })
      .then((d: SearchResponse) => {
        setSearchAll(d.matches);
        setTotal(d.total);
        setTotalPages(Math.max(1, Math.ceil(d.total / pageSize)));
        setSearchMeta({ pages: d.scanned_pages, scanned: d.total_scanned, truncated: d.truncated });
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (isAbort(e)) return;
        setError(errorText(e)); setLoading(false); setSearchAll(null);
      });
    return () => ctrl.abort();
  }, [isSearch, applied, typeId, reload]);

  const visibleItems = isSearch && searchAll
    ? searchAll.slice((page - 1) * pageSize, page * pageSize)
    : items;
  const visiblePages = isSearch && searchAll
    ? Math.max(1, Math.ceil(searchAll.length / pageSize))
    : totalPages;

  const resetAll = () => {
    setQuery(""); setTypeId(""); setPage(1); setApplied(""); setReload((n) => n + 1);
  };

  return (
    <Card className="border border-gray-200 dark:border-slate-700 shadow-sm overflow-hidden">
      <CardHeader className="border-b border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-900">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-9 h-9 bg-blue-600 rounded-lg shrink-0">
            <BookOpen className="size-5 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <CardTitle className="text-lg font-semibold text-gray-900 dark:text-slate-100">Публикации ЮФУ</CardTitle>
            <CardDescription className="text-sm text-gray-600 dark:text-slate-400">
              Научные публикации Южного федерального университета из репозитория hub.sfedu.ru
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 sm:p-6 space-y-4">
        <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-gray-400 pointer-events-none" />
            <Input value={query} onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") applySearch(query); }}
              placeholder="Поиск по названию, авторам, дисциплине (от 2 символов)" aria-label="Поиск публикаций"
              className="pl-9 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600" />
          </div>
          <Select value={typeId || "all"} disabled={isSearch}
            onValueChange={(v) => { setTypeId(v === "all" ? "" : v); setPage(1); setReload((n) => n + 1); }}>
            <SelectTrigger className="sm:w-56 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600"
              aria-label="Фильтр по типу публикации"
              title={isSearch ? "Фильтр типа действует только в режиме каталога" : undefined}>
              <SelectValue placeholder="Все типы" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Все типы</SelectItem>
              {types.map((t) => (
                <SelectItem key={t.id} value={String(t.id)}>{t.name ?? `Тип ${t.id}`} ({t.count})</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button onClick={() => applySearch(query)} disabled={loading || query.trim().length === 1} className="bg-blue-600 hover:bg-blue-700 text-white">
            {loading && isSearch ? <Loader2 className="size-4 animate-spin" /> : <Search className="size-4" />}Найти
          </Button>
          <Button variant="outline" onClick={resetAll}
            className="border-gray-300 dark:border-slate-600 text-gray-700 dark:text-slate-300 hover:bg-gray-50 dark:hover:bg-slate-800">
            <RotateCcw className="size-4" />Сбросить
          </Button>
        </div>

        <Alert className="border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-200">
          <Info className="size-4" />
          <AlertTitle>Ограничение поиска</AlertTitle>
          <AlertDescription className="text-blue-700 dark:text-blue-300">
            <p>
              Поиск идёт по последним{searchMeta ? ` ${searchMeta.pages}` : ""} страницам каталога хаба
              (серверного поиска у hub.sfedu.ru нет): это порядка
              {searchMeta ? ` ${searchMeta.scanned}` : " 1000"} свежих записей из 133 902, поэтому выдача неполная.
            </p>
            {searchMeta?.truncated ? (
              <p>Просмотрено страниц: {searchMeta.pages} (записей: {searchMeta.scanned}); более старые страницы каталога не сканировались.</p>
            ) : null}
          </AlertDescription>
        </Alert>

        {loading ? (
          <div className="space-y-3" role="status" aria-label="Загрузка публикаций">
            {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28 w-full rounded-xl" />)}
          </div>
        ) : error ? (
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertTitle>Не удалось загрузить публикации</AlertTitle>
            <AlertDescription>
              <p>{error}</p>
              <Button variant="outline" size="sm" onClick={() => setReload((n) => n + 1)} className="mt-2">Повторить</Button>
            </AlertDescription>
          </Alert>
        ) : visibleItems.length === 0 ? (
          <div className="py-12 text-center">
            <p className="text-sm text-gray-500 dark:text-slate-400">Ничего не найдено</p>
            {isSearch ? <p className="mt-1 text-xs text-gray-400 dark:text-slate-500">Попробуйте изменить запрос или сбросить фильтры</p> : null}
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-xs text-gray-500 dark:text-slate-400">{isSearch ? `Найдено: ${total}` : `Всего публикаций: ${total}`}</p>
            <ul role="list" className="space-y-3">
              {visibleItems.map((p) => <li key={p.id}><PublicationCard pub={p} onOpen={setDetailId} /></li>)}
            </ul>
            <PaginationBar page={page} totalPages={visiblePages} pageSize={pageSize} onPage={setPage}
              onPageSize={(n) => { setPageSize(n); setPage(1); }} />
          </div>
        )}
      </CardContent>
      {detailId !== null ? <PublicationDetailDialog id={detailId} onClose={() => setDetailId(null)} /> : null}
    </Card>
  );
}
