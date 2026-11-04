import { useEffect, useState } from "react";
import { ExternalLink, FileText } from "lucide-react";

import { ErrorState, Skeleton } from "../../design";
import { api } from "../api";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "./ui/dialog";
import { EXTERNAL_LINK, OUTLINE_BADGE, TYPE_BADGE, doiHref, errorText, isAbort } from "./PublicationCard";
import { resolveHubLink, type Publication } from "./publicationsFilter";

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

function DetailBody({ detail }: { detail: Publication }) {
  const rows: [string, string | null][] = [
    ["Год", detail.year ? String(detail.year) : (detail.publishing_year ? String(detail.publishing_year) : null)],
    ["Дата добавления", formatDate(detail.created_at)],
    ["Тип", detail.type_name],
    ["Журнал / издание", detail.journal],
    ["Издатель", detail.publisher],
    ["Страницы", detail.pages],
    ["Язык", detail.language],
    ["ISBN", detail.isbn],
    ["Автор в хабе", detail.creator_username],
    ["Подпись", detail.signature],
  ];
  const tagSections: [string, string[]][] = [
    ["Ключевые слова", detail.keywords],
    ["Дисциплины", detail.academic_subject],
    ["Образовательные программы", detail.education_program],
  ];
  const hubLink = resolveHubLink(detail);
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
      {detail.abstract ? (
        <div>
          <h3 className="text-sm font-medium text-gray-900 dark:text-slate-100">Аннотация</h3>
          <p className="mt-1.5 text-sm text-gray-700 dark:text-slate-300 whitespace-pre-line">{detail.abstract}</p>
        </div>
      ) : null}
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
      {hubLink ? (
        <Button asChild variant="outline" size="sm" className="border-gray-300 dark:border-slate-600">
          <a href={hubLink} target="_blank" rel="noreferrer"><ExternalLink className="size-3.5" />Открыть в хабе</a>
        </Button>
      ) : null}
    </div>
  );
}

export function PublicationDetailDialog({ id, onClose }: { id: number; onClose: () => void }) {
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
          <ErrorState
            title="Не удалось загрузить публикацию"
            description={error}
            onRetry={() => setReload((n) => n + 1)}
          />
        ) : detail ? <DetailBody detail={detail} /> : null}
      </DialogContent>
    </Dialog>
  );
}
