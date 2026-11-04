import { ChevronLeft, ChevronRight, ExternalLink } from "lucide-react";

import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { resolveHubLink, shortenAbstract, type Publication } from "./publicationsFilter";

/**
 * Общие константы и помощники вкладки «Публикации»: их используют карточка,
 * диалог деталей и контейнер списка, поэтому держим их в одном модуле.
 */
export const TYPE_BADGE = "bg-blue-100 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300";
export const OUTLINE_BADGE = "text-gray-600 dark:text-slate-300 border-gray-300 dark:border-slate-600";
export const EXTERNAL_LINK = "text-blue-600 dark:text-blue-400 hover:underline";

export const isAbort = (e: unknown): boolean =>
  e instanceof DOMException && e.name === "AbortError";

const RUSSIAN_ERRORS: Record<number, string> = {
  429: "Слишком много запросов к хабу. Подождите минуту и повторите.",
  502: "Сервис хаба отклонил запрос. Попробуйте позже.",
  503: "Сервис хаба недоступен. Попробуйте позже.",
  504: "Сервис хаба не ответил вовремя. Попробуйте позже.",
  422: "Некорректный запрос: проверьте параметры поиска.",
};

export function errorText(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e);
  let detail = raw;
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown };
    if (parsed && typeof parsed.detail === "string") detail = parsed.detail;
  } catch { /* тело не JSON - работаем с сырым текстом */ }
  if (/too many requests|rate limit/i.test(detail)) return RUSSIAN_ERRORS[429];
  if (/internal server error/i.test(detail)) return "Внутренняя ошибка сервера. Попробуйте позже.";
  if (/failed to fetch|networkerror|load failed/i.test(detail)) return "Нет связи с сервером. Проверьте подключение.";
  return detail;
}

export const doiHref = (doi: string): string =>
  /^https?:\/\//i.test(doi) ? doi : `https://doi.org/${doi}`;

export function PaginationBar({ page, totalPages, pageSize, onPage, onPageSize }: {
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

export function PublicationCard({ pub, onOpen }: { pub: Publication; onOpen: (id: number) => void }) {
  const subjects = pub.academic_subject.slice(0, 3);
  const more = pub.academic_subject.length - subjects.length;
  const year = pub.year ?? pub.publishing_year;
  const hubLink = resolveHubLink(pub);
  const abstract = shortenAbstract(pub.abstract);
  return (
    <Card className="border border-gray-200 dark:border-slate-700 shadow-sm">
      <CardContent className="p-4 sm:p-5 space-y-2.5">
        <div className="flex flex-wrap items-center gap-2">
          {pub.type_name ? <Badge className={TYPE_BADGE}>{pub.type_name}</Badge> : null}
          {year ? <span className="text-sm text-gray-500 dark:text-slate-400">{year}</span> : null}
          {pub.journal ? <span className="text-xs text-gray-500 dark:text-slate-400 truncate max-w-[18rem]" title={pub.journal}>{pub.journal}</span> : null}
          <span className="ml-auto flex items-center gap-3 text-xs">
            {pub.doi ? <a href={doiHref(pub.doi)} target="_blank" rel="noreferrer" className={EXTERNAL_LINK}>DOI</a> : null}
            {hubLink ? (
              <a href={hubLink} target="_blank" rel="noreferrer" className={`inline-flex items-center gap-1 ${EXTERNAL_LINK}`}>
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
        {abstract ? <p className="text-sm text-gray-500 dark:text-slate-400">{abstract}</p> : null}
        {pub.keywords.length > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            {pub.keywords.slice(0, 5).map((k) => (
              <Badge key={k} variant="outline" className="text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800">{k}</Badge>
            ))}
          </div>
        ) : null}
        {subjects.length > 0 ? (
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
