/**
 * Статьи по программе из Course Trends API ЮФУ.
 *
 * Сервис ЮФУ отдаёт предрассчитанные данные по программе (`pid`), включая
 * статьи по курсам. Бэкенд нормализует произвольную структуру ответа в
 * стабильный формат (см. src/infrastructure/course_trends_normalize.py),
 * поэтому здесь нет привязки к именам полей сервиса.
 *
 * Состояния: нет входа через хаб (419), сервис недоступен (502), пустая
 * выдача (не ошибка), загрузка. Все они показываются явно: молчаливый пустой
 * список выглядел бы как «статей нет», хотя причина другая.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Card, CardContent, CardHeader, CardTitle, CardDescription,
} from "./ui/card";
import { BookOpen, ExternalLink, Search, RefreshCw, AlertCircle, Loader2 } from "lucide-react";
import { request } from "../../lib/api";
import { reportError } from "../../lib/errors";
import { PageHeader } from "./PageHeader";

type Article = {
  title: string;
  abstract: string;
  year: number | null;
  authors: string[];
  url: string;
  doi: string | null;
  citations: number | null;
  source: string;
  course: string;
};

type ArticlesResponse = {
  pid: string;
  total: number;
  limit: number;
  offset: number;
  items: Article[];
  source: string;
};

type Program = { pid: string; label: string; n_courses?: number; avg_coverage?: number };

function bearerHeaders(): Record<string, string> {
  // Сессия в HttpOnly-cookie: токена в JS больше нет, браузер приложит сам.
  return {};
}

async function getJson<T>(path: string): Promise<T> {
  const res = await request<Response>(path, { raw: true, headers: bearerHeaders() });
  if (!res.ok) {
    let detail = `Ошибка (${res.status})`;
    try {
      const j = await res.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch (err) {
      reportError("app/components/ArticlesPanel.tsx:60", err);
    }
    const error = new Error(detail) as Error & { status?: number };
    error.status = res.status;
    throw error;
  }
  return res.json() as Promise<T>;
}

const PAGE = 50;

export function ArticlesPanel() {
  const [programs, setPrograms] = useState<Program[]>([]);
  const [pid, setPid] = useState<string>("");
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [data, setData] = useState<ArticlesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [needHub, setNeedHub] = useState(false);
  const debounceRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    getJson<{ programs?: Program[] } | Program[]>("/api/academic/programs")
      .then((raw) => {
        if (cancelled) return;
        const list = Array.isArray(raw) ? raw : (raw.programs ?? []);
        const normalized = list
          .filter((p): p is Program => !!p && typeof (p as Program).pid === "string")
          .map((p) => ({ ...p, label: p.label || p.pid }));
        setPrograms(normalized);
        if (normalized.length && !pid) setPid(normalized[0].pid);
      })
      .catch((e) => {
        if (cancelled) return;
        const status = (e as { status?: number }).status;
        setNeedHub(status === 419 || status === 403);
        setError(e.message);
      });
    return () => { cancelled = true; };
    // pid намеренно не в зависимостях: список программ грузим один раз.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (debounceRef.current !== null) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => setDebounced(query.trim()), 350);
    return () => {
      if (debounceRef.current !== null) window.clearTimeout(debounceRef.current);
    };
  }, [query]);

  const load = useCallback(async (offset: number) => {
    if (!pid) return;
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        limit: String(PAGE),
        offset: String(offset),
      });
      if (debounced) params.set("query", debounced);
      const next = await getJson<ArticlesResponse>(
        `/api/academic/programs/${encodeURIComponent(pid)}/articles?${params}`,
      );
      setData((prev) => (offset > 0 && prev
        ? { ...next, items: [...prev.items, ...next.items] }
        : next));
      setNeedHub(false);
    } catch (e) {
      const err = e as Error & { status?: number };
      setNeedHub(err.status === 419 || err.status === 403);
      setError(err.message || "Не удалось загрузить статьи");
    } finally {
      setLoading(false);
    }
  }, [pid, debounced]);

  useEffect(() => {
    if (!pid) return;
    setData(null);
    void load(0);
  }, [pid, debounced, load]);

  const items = data?.items ?? [];
  const hasMore = !!data && data.offset + data.items.length < data.total;

  const years = useMemo(() => {
    const set = new Set<number>();
    for (const a of items) if (a.year) set.add(a.year);
    return Array.from(set).sort((a, b) => b - a);
  }, [items]);

  return (
    <div className="space-y-4">
      <PageHeader
        title="Научные статьи"
        description="Публикации по дисциплинам программы из сервиса ЮФУ"
      />

      {needHub && (
        <Card>
          <CardContent className="flex items-start gap-2 p-4 text-sm">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <div>
              <div className="font-medium">Нужен вход через hub.sfedu.ru</div>
              <div className="text-muted-foreground">
                Сервис ЮФУ выдаёт статьи только по токену хаба. Войдите через
                hub.sfedu.ru и откройте раздел снова.
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Программа и поиск</CardTitle>
          <CardDescription>
            {data ? `${data.total} публикаций найдено` : "Выберите программу"}
            {data?.source ? ` · источник: ${data.source}` : ""}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3 items-end">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-muted-foreground">Программа</span>
            <select
              value={pid}
              onChange={(e) => setPid(e.target.value)}
              className="h-9 rounded-md border px-2 bg-background min-w-[260px]"
              aria-label="Программа"
            >
              {programs.length === 0 && <option value="">нет доступных программ</option>}
              {programs.map((p) => (
                <option key={p.pid} value={p.pid}>{p.label}</option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-sm flex-1 min-w-[240px]">
            <span className="text-muted-foreground">Поиск по теме, авторам, аннотации</span>
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="например: графовые нейросети"
                className="h-9 w-full rounded-md border pl-8 pr-2 bg-background"
                aria-label="Поиск статей"
              />
            </div>
          </label>

          <button
            onClick={() => void load(0)}
            disabled={loading || !pid}
            className="h-9 px-3 rounded-md border inline-flex items-center gap-1.5 text-sm cursor-pointer disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Обновить
          </button>
        </CardContent>
      </Card>

      {error && !needHub && (
        <Card>
          <CardContent className="flex items-start gap-2 p-4 text-sm">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </CardContent>
        </Card>
      )}

      {!loading && !error && data && items.length === 0 && (
        <Card>
          <CardContent className="flex items-start gap-2 p-6 text-sm">
            <BookOpen className="h-4 w-4 mt-0.5 shrink-0" />
            <div>
              <div className="font-medium">Публикаций не найдено</div>
              <div className="text-muted-foreground">
                {debounced
                  ? `По запросу «${debounced}» ничего нет. Попробуйте более общую формулировку.`
                  : "Для этой программы сервис не вернул статей."}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {items.length > 0 && (
        <>
          <div className="text-xs text-muted-foreground flex flex-wrap gap-3">
            <span>Показано {items.length} из {data?.total ?? 0}</span>
            {years.length > 0 && (
              <span>Годы: {years[years.length - 1]}–{years[0]}</span>
            )}
          </div>

          <div className="space-y-3">
            {items.map((a, i) => (
              <Card key={`${a.doi ?? a.title}-${i}`}>
                <CardContent className="p-4 space-y-2">
                  <div className="flex items-start justify-between gap-3">
                    <div className="font-medium leading-snug">
                      {a.url ? (
                        <a href={a.url} target="_blank" rel="noopener noreferrer"
                           className="hover:underline inline-flex items-center gap-1">
                          {a.title}
                          <ExternalLink className="h-3.5 w-3.5 shrink-0 opacity-60" />
                        </a>
                      ) : a.title}
                    </div>
                    <div className="flex gap-1.5 shrink-0 text-xs">
                      {a.year && (
                        <span className="rounded border px-1.5 py-0.5">{a.year}</span>
                      )}
                      {typeof a.citations === "number" && (
                        <span className="rounded border px-1.5 py-0.5" title="Цитирований">
                          {a.citations} цит.
                        </span>
                      )}
                    </div>
                  </div>

                  {a.authors.length > 0 && (
                    <div className="text-xs text-muted-foreground">
                      {a.authors.slice(0, 6).join(", ")}
                      {a.authors.length > 6 ? ` и др. (${a.authors.length})` : ""}
                    </div>
                  )}

                  {a.abstract && (
                    <p className="text-sm text-muted-foreground line-clamp-4">{a.abstract}</p>
                  )}

                  <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                    {a.course && <span className="rounded bg-muted px-1.5 py-0.5">{a.course}</span>}
                    {a.source && <span>{a.source}</span>}
                    {a.doi && <span className="font-mono">doi:{a.doi}</span>}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {hasMore && (
            <div className="flex justify-center pt-2">
              <button
                onClick={() => void load(data!.offset + data!.items.length)}
                disabled={loading}
                className="px-4 py-2 rounded-md border text-sm cursor-pointer disabled:opacity-50"
              >
                {loading ? "Загрузка…" : `Показать ещё (ещё ${data!.total - items.length})`}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default ArticlesPanel;
