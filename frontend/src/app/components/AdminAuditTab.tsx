/**
 * Вкладка «Аудит» админ-консоли.
 *
 * Журнал действий администраторов: GET /api/admin/audit?action=&actor=&target_type=
 * &since=&until=&limit=&offset=. Действия для подсказок берутся из
 * GET /api/admin/audit/actions. Клик по строке или кнопке «Подробнее» открывает
 * карточку события, «Экспорт CSV» выгружает текущую страницу в файл.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { request, saveBlob } from "../../lib/api";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  Field,
  inputClass,
  Skeleton,
  TableFrame,
  tableHeadClass,
  tableRowClass,
} from "../../design/primitives";
import { fmtDateTime, roleLabel, roleTone } from "./adminRoles";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { ChevronLeft, ChevronRight, Download, RefreshCw } from "lucide-react";

const SKELETON_ROWS = 8;
const selectClass =
  "h-9 px-3 rounded-lg border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-950 text-sm";

interface AuditEntry {
  id: string;
  actor_email: string;
  actor_role: string;
  action: string;
  target_type: string;
  target_id: string | number | null;
  target: string;
  meta: unknown;
  ip: string;
  request_id: string | null;
  created_at: string;
}

interface AuditPage {
  entries: AuditEntry[];
  total: number;
  limit: number;
  offset: number;
}

/** datetime-local -> ISO 8601 (UTC) или пустая строка, если поле не заполнено. */
/** Локальное значение для input[type=datetime-local] (без секунд и TZ). */
function toLocalInput(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`;
}

/** Быстрые периоды: последние N часов (7 д = 168 ч, 30 д = 720 ч). */
const PERIOD_PRESETS: { label: string; hours: number }[] = [
  { label: "1 ч", hours: 1 },
  { label: "24 ч", hours: 24 },
  { label: "7 д", hours: 168 },
  { label: "30 д", hours: 720 },
];

function toIso(localValue: string): string {
  if (!localValue) return "";
  const d = new Date(localValue);
  return isNaN(d.getTime()) ? "" : d.toISOString();
}

function formatMeta(meta: unknown): string {
  if (meta === null || meta === undefined) return "{}";
  if (typeof meta === "string") return meta;
  try {
    return JSON.stringify(meta, null, 2);
  } catch {
    return String(meta);
  }
}

function csvCell(value: unknown): string {
  const s = value === null || value === undefined ? "" : String(value);
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function AdminAuditTab() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [phase, setPhase] = useState<"loading" | "error" | "ready">("loading");
  const [fetchError, setFetchError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const requestIdRef = useRef(0);

  const [action, setAction] = useState("");
  const [actions, setActions] = useState<string[]>([]);
  const [actor, setActor] = useState("");
  const [actorQ, setActorQ] = useState("");
  const [targetType, setTargetType] = useState("");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [limit, setLimit] = useState(50);
  const [offset, setOffset] = useState(0);

  const [selected, setSelected] = useState<AuditEntry | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setActorQ(actor.trim());
      setOffset(0);
    }, 300);
    return () => clearTimeout(timer);
  }, [actor]);

  const loadAudit = useCallback(async () => {
    const reqId = ++requestIdRef.current;
    setPhase("loading");
    setFetchError("");
    const params = new URLSearchParams();
    if (action.trim()) params.set("action", action.trim());
    if (actorQ) params.set("actor", actorQ);
    if (targetType.trim()) params.set("target_type", targetType.trim());
    const sinceIso = toIso(since);
    if (sinceIso) params.set("since", sinceIso);
    const untilIso = toIso(until);
    if (untilIso) params.set("until", untilIso);
    params.set("limit", String(limit));
    params.set("offset", String(offset));
    try {
      const d = await request<AuditPage>(`/api/admin/audit?${params.toString()}`);
      if (reqId !== requestIdRef.current) return;
      const list: AuditEntry[] = Array.isArray(d?.entries) ? d.entries : [];
      const totalCount =
        typeof d?.total === "number" && d.total >= 0 ? d.total : list.length;
      setEntries(list);
      setTotal(totalCount);
      setPhase("ready");
      if (list.length === 0 && totalCount > 0 && offset > 0) {
        setOffset(Math.max(0, Math.floor((totalCount - 1) / limit) * limit));
      }
    } catch (e) {
      if (reqId !== requestIdRef.current) return;
      setFetchError(e instanceof Error ? e.message : "Не удалось загрузить журнал");
      setPhase("error");
    }
  }, [action, actorQ, targetType, since, until, limit, offset, reloadKey]);

  useEffect(() => {
    void loadAudit();
  }, [loadAudit]);

  // список возможных действий для подсказок в фильтре
  useEffect(() => {
    request<{ actions?: string[]; total?: number }>("/api/admin/audit/actions")
      .then((d) => setActions(Array.isArray(d?.actions) ? d.actions : []))
      .catch(() => setActions([]));
  }, []);

  const bumpReload = () => setReloadKey((k) => k + 1);

  /** Быстрый период: since = сейчас - hours, until пуст (до сейчас). null = всё время. */
  const applyPeriod = (hours: number | null) => {
    setSince(hours === null ? "" : toLocalInput(new Date(Date.now() - hours * 3600_000)));
    setUntil("");
    setOffset(0);
  };

  const resetFilters = () => {
    setAction("");
    setActor("");
    setTargetType("");
    setSince("");
    setUntil("");
    setOffset(0);
  };

  const exportCsv = () => {
    const header = [
      "время",
      "актор",
      "роль",
      "действие",
      "тип объекта",
      "id объекта",
      "объект",
      "ip",
      "request_id",
    ];
    const lines = [header.map(csvCell).join(",")];
    for (const e of entries) {
      lines.push(
        [
          fmtDateTime(e.created_at),
          e.actor_email,
          e.actor_role,
          e.action,
          e.target_type,
          e.target_id === null || e.target_id === undefined ? "" : String(e.target_id),
          e.target,
          e.ip,
          e.request_id,
        ]
          .map(csvCell)
          .join(","),
      );
    }
    const csv = "\uFEFF" + lines.join("\r\n");
    saveBlob(new Blob([csv], { type: "text/csv;charset=utf-8" }), `audit-${new Date().toISOString().slice(0, 10)}.csv`);
  };

  const first = total === 0 ? 0 : offset + 1;
  const last = Math.min(offset + limit, total);
  const hasFilters =
    action !== "" || actorQ !== "" || targetType !== "" || since !== "" || until !== "";
  const showSkeleton = phase === "loading" && entries.length === 0;
  const hasResultRows = showSkeleton || entries.length > 0;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Журнал аудита"
          description="Кто, что и когда делал в системе. Фильтры применяются на сервере."
          actions={
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1" role="group" aria-label="Быстрый период">
                {PERIOD_PRESETS.map((p) => (
                  <button
                    key={p.label}
                    type="button"
                    className="rounded-md border border-border px-2 py-1 text-xs text-foreground hover:bg-muted"
                    onClick={() => applyPeriod(p.hours)}
                  >
                    {p.label}
                  </button>
                ))}
                <button
                  type="button"
                  className="rounded-md border border-border px-2 py-1 text-xs text-foreground hover:bg-muted"
                  onClick={() => applyPeriod(null)}
                >
                  Всё
                </button>
              </div>
              <Button
                size="sm"
                variant="secondary"
                onClick={exportCsv}
                disabled={entries.length === 0}
                iconLeft={<Download className="size-4" />}
              >
                Экспорт CSV
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={bumpReload}
                loading={phase === "loading"}
                iconLeft={<RefreshCw className="size-4" />}
              >
                Обновить
              </Button>
            </div>
          }
        />
        <CardBody className="space-y-4">
          {/* фильтры */}
          <div className="flex flex-wrap items-end gap-3">
            <Field
              label="Действие"
              htmlFor="audit-action"
              hint="Выберите из списка или введите префикс"
              className="w-60 max-w-full"
            >
              <input
                id="audit-action"
                className={inputClass}
                list="audit-actions-list"
                placeholder="например user.patch"
                value={action}
                onChange={(e) => {
                  setAction(e.target.value);
                  setOffset(0);
                }}
                autoComplete="off"
              />
              <datalist id="audit-actions-list">
                {actions.map((a) => (
                  <option key={a} value={a} />
                ))}
              </datalist>
            </Field>
            <Field label="Актор (email)" htmlFor="audit-actor" className="w-60 max-w-full">
              <input
                id="audit-actor"
                className={inputClass}
                value={actor}
                onChange={(e) => setActor(e.target.value)}
                autoComplete="off"
              />
            </Field>
            <Field label="Тип объекта" htmlFor="audit-target-type" className="w-44 max-w-full">
              <input
                id="audit-target-type"
                className={inputClass}
                value={targetType}
                onChange={(e) => {
                  setTargetType(e.target.value);
                  setOffset(0);
                }}
                autoComplete="off"
              />
            </Field>
            <Field label="С" htmlFor="audit-since">
              <input
                id="audit-since"
                type="datetime-local"
                className={inputClass}
                value={since}
                onChange={(e) => {
                  setSince(e.target.value);
                  setOffset(0);
                }}
              />
            </Field>
            <Field label="По" htmlFor="audit-until">
              <input
                id="audit-until"
                type="datetime-local"
                className={inputClass}
                value={until}
                onChange={(e) => {
                  setUntil(e.target.value);
                  setOffset(0);
                }}
              />
            </Field>
            <Field label="На странице" htmlFor="audit-limit">
              <select
                id="audit-limit"
                className={selectClass}
                value={String(limit)}
                onChange={(e) => {
                  setLimit(Number(e.target.value) || 50);
                  setOffset(0);
                }}
              >
                <option value="50">50</option>
                <option value="100">100</option>
                <option value="200">200</option>
              </select>
            </Field>
          </div>

          {phase === "error" && entries.length === 0 && (
            <ErrorState
              title="Не удалось загрузить журнал аудита"
              description={fetchError}
              onRetry={loadAudit}
            />
          )}
          {phase === "error" && entries.length > 0 && (
            <div
              role="alert"
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300"
            >
              <span>{fetchError}</span>
              <Button size="sm" variant="secondary" onClick={loadAudit}>
                Повторить
              </Button>
            </div>
          )}

          {phase === "ready" && entries.length === 0 && (
            <EmptyState
              title={hasFilters ? "События не найдены" : "Журнал пуст"}
              description={
                hasFilters
                  ? "Попробуйте изменить условия или сбросить фильтры."
                  : "События появятся после первых действий в системе."
              }
              action={
                hasFilters ? (
                  <Button variant="secondary" size="sm" onClick={resetFilters}>
                    Сбросить фильтры
                  </Button>
                ) : undefined
              }
            />
          )}

          {hasResultRows && (
            <TableFrame>
              <table className="w-full text-sm">
                <thead className={tableHeadClass}>
                  <tr>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Время</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Актор</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Действие</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Объект</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">Детали</th>
                    <th scope="col" className="px-3 py-2.5 text-left font-semibold">IP</th>
                    <th scope="col" className="px-3 py-2.5 text-right font-semibold">
                      <span className="sr-only">Подробнее</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {showSkeleton
                    ? Array.from({ length: SKELETON_ROWS }).map((_, i) => (
                        <tr key={i} className={tableRowClass}>
                          {Array.from({ length: 7 }).map((__, j) => (
                            <td key={j} className="px-3 py-3">
                              <Skeleton className="h-4 w-20" />
                            </td>
                          ))}
                        </tr>
                      ))
                    : entries.map((e) => (
                        <tr
                          key={e.id}
                          className={`${tableRowClass} cursor-pointer`}
                          onClick={() => setSelected(e)}
                        >
                          <td className="whitespace-nowrap px-3 py-2 text-xs text-muted-foreground">
                            {fmtDateTime(e.created_at)}
                          </td>
                          <td className="px-3 py-2">
                            <span className="block text-xs text-foreground">{e.actor_email}</span>
                            <Badge tone={roleTone(e.actor_role)}>{roleLabel(e.actor_role)}</Badge>
                          </td>
                          <td className="px-3 py-2">
                            <span className="font-mono text-xs text-foreground">{e.action}</span>
                          </td>
                          <td className="px-3 py-2">
                            <span className="font-mono text-xs text-muted-foreground">
                              {e.target_type}:{e.target_id ?? "-"}
                            </span>
                          </td>
                          <td
                            className="max-w-[14rem] truncate px-3 py-2 text-xs text-muted-foreground"
                            title={e.target || undefined}
                          >
                            {e.target || "-"}
                          </td>
                          <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-muted-foreground">
                            {e.ip || "-"}
                          </td>
                          <td className="px-3 py-2 text-right">
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={(event) => {
                                event.stopPropagation();
                                setSelected(e);
                              }}
                            >
                              Подробнее
                            </Button>
                          </td>
                        </tr>
                      ))}
                </tbody>
              </table>
            </TableFrame>
          )}

          {total > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-xs text-muted-foreground" aria-live="polite">
                Показано {first}-{last} из {total}
              </p>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={offset === 0 || phase === "loading"}
                  onClick={() => setOffset(Math.max(0, offset - limit))}
                >
                  <ChevronLeft className="size-4" /> Назад
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={offset + limit >= total || phase === "loading"}
                  onClick={() => setOffset(offset + limit)}
                >
                  Вперёд <ChevronRight className="size-4" />
                </Button>
              </div>
            </div>
          )}
        </CardBody>
      </Card>

      {/* Полная карточка события */}
      <Dialog open={selected !== null} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-xl">
          <DialogHeader>
            <DialogTitle className="font-mono text-base">{selected?.action}</DialogTitle>
            <DialogDescription>
              {selected ? fmtDateTime(selected.created_at) : ""}
            </DialogDescription>
          </DialogHeader>
          {selected && (
            <div className="space-y-3">
              <dl className="grid grid-cols-1 gap-x-4 gap-y-1.5 text-sm sm:grid-cols-2">
                <dt className="text-xs text-muted-foreground">Актор</dt>
                <dd className="flex items-center gap-2 break-all text-foreground">
                  {selected.actor_email}
                  <Badge tone={roleTone(selected.actor_role)}>{roleLabel(selected.actor_role)}</Badge>
                </dd>
                <dt className="text-xs text-muted-foreground">Объект</dt>
                <dd className="break-all font-mono text-xs text-foreground">
                  {selected.target_type}:{selected.target_id ?? "-"}
                </dd>
                <dt className="text-xs text-muted-foreground">Детали</dt>
                <dd className="break-all text-foreground">{selected.target || "-"}</dd>
                <dt className="text-xs text-muted-foreground">IP</dt>
                <dd className="font-mono text-xs text-foreground">{selected.ip || "-"}</dd>
                <dt className="text-xs text-muted-foreground">Request ID</dt>
                <dd className="break-all font-mono text-xs text-foreground">
                  {selected.request_id || "-"}
                </dd>
              </dl>
              <div>
                <div className="mb-1 text-xs font-semibold text-muted-foreground">Meta</div>
                <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-muted/40 p-3 text-xs font-mono text-foreground">
                  {formatMeta(selected.meta)}
                </pre>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
