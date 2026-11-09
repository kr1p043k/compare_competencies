import { useState, useEffect, useCallback, useMemo, memo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { BarChart3, Activity, TrendingUp, AlertCircle, RefreshCw, Server, Cpu, Bell, Newspaper, Trash2, Plus, CheckCheck, ExternalLink, Mail } from "lucide-react";
import { request } from "../../lib/api";
import { reportError } from "./../../lib/errors";
import { PageHeader } from "./PageHeader";
import { Skeleton } from "../../design";

interface MetricSample {
  name: string;
  labels: Record<string, string>;
  value: number;
}

interface MetricFamily {
  name: string;
  type: string;
  samples: MetricSample[];
}

interface Subscription {
  id: string;
  topic: string;
  source: string;
  telegram_chat_id: string | null;
  email: string | null;
  is_active: boolean;
  last_checked_at: string | null;
  created_at: string;
}

interface NotificationItem {
  id: string;
  subscription_id: string | null;
  title: string;
  body: string;
  article_url: string | null;
  article_source: string | null;
  severity?: string;
  is_read: boolean;
  created_at: string;
}

interface GrafanaDashboard {
  uid: string;
  title: string;
  url?: string | null;
}

interface GrafanaStatus {
  enabled: boolean;
  reachable: boolean;
  reason?: string | null;
  dashboards: GrafanaDashboard[];
}

function MonitoringTabBase() {
  const [metrics, setMetrics] = useState<Record<string, MetricFamily> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [health, setHealth] = useState<{
    latency_p95_by_endpoint?: Record<string, number | null>;
    errors_1h?: { "4xx"?: number; "5xx"?: number };
    sessions_active?: number | null;
    recent_runs?: Array<{ action?: string; status?: string; started_at?: string; completed_at?: string; error_message?: string | null }>;
    freshness?: { vacancies?: number; date_from?: string | null; date_to?: string | null };
  } | null>(null);

  const [grafana, setGrafana] = useState<GrafanaStatus | null>(null);

  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [subsLoading, setSubsLoading] = useState(false);
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [notifsLoading, setNotifsLoading] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  const [newTopic, setNewTopic] = useState("");
  const [newSource, setNewSource] = useState("openalex+arxiv");
  const [newTelegramChatId, setNewTelegramChatId] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [subError, setSubError] = useState<string | null>(null);
  const [notifError, setNotifError] = useState<string | null>(null);
  const [notifFilter, setNotifFilter] = useState<"all" | "unread">("all");

  const loadMetrics = useCallback(async () => {
    try {
      const data = await request<Record<string, MetricFamily>>("/api/admin/monitoring");
      setMetrics(data);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
    try {
      setHealth(await request<NonNullable<typeof health>>("/api/admin/monitoring/summary"));
    } catch (err) { reportError("app/components/MonitoringTab.tsx:88", err); }
    try {
      setGrafana(await request<GrafanaStatus>("/api/admin/monitoring/grafana"));
    } catch (err) { reportError("app/components/MonitoringTab.tsx:grafana", err); }
  }, []);

  const loadSubscriptions = useCallback(async () => {
    setSubsLoading(true);
    setSubError(null);
    try {
      const data = await request<{ subscriptions?: Subscription[] }>("/api/subscriptions");
      setSubscriptions(data.subscriptions ?? []);
    } catch (e: any) {
      setSubError(e.message ?? "Ошибка сети");
    } finally {
      setSubsLoading(false);
    }
  }, []);

  const loadNotifications = useCallback(async () => {
    setNotifsLoading(true);
    setNotifError(null);
    try {
      const query = notifFilter === "unread" ? "/api/notifications?limit=50&unread_only=true" : "/api/notifications?limit=50";
      const data = await request<{ notifications?: NotificationItem[] }>(query);
      setNotifications(data.notifications ?? []);
    } catch (e: any) {
      setNotifError(e.message ?? "Ошибка сети");
    } finally {
      setNotifsLoading(false);
    }
  }, [notifFilter]);

  const loadUnreadCount = useCallback(async () => {
    try {
      const data = await request<{ unread_count?: number }>("/api/notifications/unread-count");
      setUnreadCount(data.unread_count ?? 0);
    } catch (err) { reportError("app/components/MonitoringTab.tsx:137", err); }
  }, []);

  useEffect(() => {
    loadMetrics();
    loadSubscriptions();
    loadNotifications();
    loadUnreadCount();
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      loadMetrics();
      loadSubscriptions();
      loadNotifications();
      loadUnreadCount();
    }, 15000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadMetrics, loadSubscriptions, loadNotifications, loadUnreadCount]);

  const findMetric = (name: string): MetricSample[] => {
    if (!metrics || !metrics[name]) return [];
    return metrics[name].samples;
  };

  // Чистый derive из metrics: пересчёт только при смене метрик (15с-поллинг),
  // а не на каждый ререндер компонента (тик, ввод форм, фильтр уведомлений).
  const pipelineCountSamples = useMemo(
    () =>
      findMetric("pipeline_stage_duration_seconds").filter(
        (s) => s.name.endsWith("_count")
      ),
    [metrics]
  );
  const pipelineRunCount = useMemo(
    () => pipelineCountSamples.reduce((s, m) => s + m.value, 0),
    [pipelineCountSamples]
  );

  const createSubscription = async () => {
    if (!newTopic.trim()) return;
    setCreating(true);
    setCreateError(null);
    try {
      await request<Subscription>("/api/subscriptions", {
        method: "POST",
        body: {
          topic: newTopic.trim(),
          source: newSource,
          telegram_chat_id: newTelegramChatId.trim() || null,
        },
      });
      setNewTopic("");
      setNewTelegramChatId("");
      await loadSubscriptions();
    } catch (e: any) {
      setCreateError(e.message ?? "Ошибка сети");
    } finally {
      setCreating(false);
    }
  };

  const deleteSubscription = async (id: string) => {
    try {
      await request<void>(`/api/subscriptions/${id}`, { method: "DELETE" });
      setSubscriptions((prev) => prev.filter((s) => s.id !== id));
    } catch (e: any) {
      setSubError(e.message ?? "Ошибка сети");
    }
  };

  const markRead = async (id: string) => {
    try {
      await request<void>(`/api/notifications/${id}/read`, { method: "POST" });
      setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, is_read: true } : n)));
      setUnreadCount((prev) => Math.max(0, prev - 1));
    } catch (err) { reportError("app/components/MonitoringTab.tsx:224", err); }
  };

  const pipelineErrors = findMetric("pipeline_errors");
  const ltrMetrics = findMetric("ltr_model_metric");
  const recsGenerated = findMetric("recommendations_generated");
  const apiReqCount = findMetric("api_requests");

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Activity}
        title="Мониторинг"
        description="Состояние сервисов, подписки и уведомления."
      />
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted-foreground">Метрики системы, подписки на источники и уведомления</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} className="rounded" />
            Автообновление
          </label>
          <Button variant="outline" size="sm" onClick={() => { loadMetrics(); loadSubscriptions(); loadNotifications(); loadUnreadCount(); }} disabled={loading}>
            <RefreshCw className={`size-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Обновить
          </Button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-destructive bg-destructive/10 p-3 rounded-lg">
          <AlertCircle className="size-5" />
          <span>{error}</span>
        </div>
      )}

      {/* System metrics */}
      {!loading && metrics && (
        <>
          {health && (
            <Card className="border-primary/30">
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2"><Activity className="size-4" />Здоровье за час</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div>
                    <div className="text-xs text-muted-foreground">Сессий активно</div>
                    <div className="text-2xl font-bold">{health.sessions_active ?? "-"}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Ошибки 4xx / 5xx (час)</div>
                    <div className="text-2xl font-bold">
                      <span className="text-warning">{health.errors_1h?.["4xx"] ?? 0}</span>
                      {" / "}
                      <span className="text-destructive">{health.errors_1h?.["5xx"] ?? 0}</span>
                    </div>
                  </div>
                  <div className="col-span-2">
                    <div className="text-xs text-muted-foreground">Вакансий (свежесть)</div>
                    <div className="text-2xl font-bold">
                      {(health.freshness?.vacancies ?? 0).toLocaleString("ru-RU")}{" "}
                      <span className="text-xs font-normal text-muted-foreground">
                        {health.freshness?.date_from || "-"} - {health.freshness?.date_to || "-"}
                      </span>
                    </div>
                  </div>
                </div>
                {Object.keys(health.latency_p95_by_endpoint || {}).length > 0 && (
                  <div className="mb-4">
                    <div className="text-xs font-medium text-muted-foreground mb-1">p95 latency, топ эндпоинтов (с)</div>
                    {Object.entries(health.latency_p95_by_endpoint || {}).map(([ep, v]) => (
                      <div key={ep} className="flex items-center gap-2 text-xs py-0.5">
                        <span className="font-mono truncate flex-1 text-foreground">{ep}</span>
                        <span className="font-mono font-semibold">{v === null ? "-" : v.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                )}
                {(health.recent_runs || []).length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-muted-foreground mb-1">Последние прогоны</div>
                    <div className="space-y-1">
                      {(health.recent_runs || []).slice(0, 5).map((run, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <Badge variant={run.status === "completed" ? "default" : "destructive"} className="text-xs">
                            {run.status}
                          </Badge>
                          <span className="font-mono text-foreground">{run.action}</span>
                          <span className="text-subtle-foreground ml-auto">
                            {run.started_at ? new Date(run.started_at).toLocaleString("ru-RU") : "-"}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2"><Activity className="size-4" />Запуски пайплайна</CardDescription>
              </CardHeader>
              <CardContent><div className="text-2xl font-bold">{pipelineRunCount}</div></CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2"><AlertCircle className="size-4" />Ошибки пайплайна</CardDescription>
              </CardHeader>
              <CardContent><div className="text-2xl font-bold text-destructive">{pipelineErrors.reduce((s, m) => s + m.value, 0)}</div></CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2"><TrendingUp className="size-4" />Рекомендации</CardDescription>
              </CardHeader>
              <CardContent><div className="text-2xl font-bold text-success">{recsGenerated.reduce((s, m) => s + m.value, 0).toFixed(0)}</div></CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2"><Server className="size-4" />Запросы API</CardDescription>
              </CardHeader>
              <CardContent><div className="text-2xl font-bold text-primary">{apiReqCount.reduce((s, m) => s + m.value, 0).toFixed(0)}</div></CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg"><BarChart3 className="size-5" />Этапы пайплайна</CardTitle>
              <CardDescription>Длительность этапов пайплайна</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {(["data_collection", "quality_scoring", "skill_extraction", "weight_cleaning", "level_building", "cluster_training", "model_training", "gap_analysis"] as const).map((stage) => {
                  const countSamples = findMetric("pipeline_stage_duration_seconds").filter(s => s.name.endsWith("_count") && s.labels.stage === stage);
                  const sumSamples = findMetric("pipeline_stage_duration_seconds").filter(s => s.name.endsWith("_sum") && s.labels.stage === stage);
                  const count = countSamples.reduce((s, m) => s + m.value, 0);
                  const total = sumSamples.reduce((s, m) => s + m.value, 0);
                  const avg = count > 0 ? total / count : 0;
                  return (
                    <div key={stage} className="flex items-center justify-between py-2 border-b border-border last:border-0">
                      <span className="text-sm font-medium text-foreground capitalize">{stage.replace(/_/g, " ")}</span>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-muted-foreground">{count}x</span>
                        <span className="font-mono text-foreground">{avg.toFixed(1)}s avg</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {ltrMetrics.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg"><Cpu className="size-5" />LTR Model</CardTitle>
                <CardDescription>Метрики качества LTR-модели</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4">
                  {ltrMetrics.map((m) => (
                    <div key={m.labels.metric} className="text-center p-4 bg-muted rounded-lg">
                      <div className="text-xs text-muted-foreground uppercase">{m.labels.metric}</div>
                      <div className="text-xl font-bold text-foreground">{m.value.toFixed(4)}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {pipelineErrors.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg text-destructive"><AlertCircle className="size-5" />Ошибки пайплайна по этапам</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {pipelineErrors.map((m, i) => (
                    <div key={i} className="flex items-center justify-between py-1">
                      <Badge variant="outline" className="bg-destructive/10 text-destructive border-destructive/30">{m.labels.stage}</Badge>
                      <span className="font-mono text-destructive">{m.value}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          <details className="text-sm text-muted-foreground">
            <summary className="cursor-pointer hover:text-foreground">Сырые метрики</summary>
            <pre className="mt-2 p-4 bg-muted rounded-lg overflow-x-auto text-xs">
              {JSON.stringify(metrics, null, 2)}
            </pre>
          </details>
        </>
      )}

      {loading && !metrics && (
        <div role="status" aria-live="polite" className="space-y-4">
          <span className="sr-only">Загрузка метрик...</span>
          <Card className="border-primary/30">
            <CardHeader className="pb-2">
              <Skeleton className="h-4 w-32" />
            </CardHeader>
            <CardContent>
              <div className="mb-4 grid grid-cols-2 gap-4 md:grid-cols-4">
                {Array.from({ length: 4 }, (_, i) => (
                  <div key={`health-${i}`}>
                    <Skeleton className="mb-1.5 h-3 w-24" />
                    <Skeleton className="h-7 w-16" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
            {Array.from({ length: 4 }, (_, i) => (
              <Card key={`stat-${i}`}>
                <CardHeader className="pb-2">
                  <Skeleton className="h-4 w-28" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-7 w-12" />
                </CardContent>
              </Card>
            ))}
          </div>
          <Card>
            <CardHeader>
              <Skeleton className="mb-1 h-5 w-40" />
            </CardHeader>
            <CardContent className="space-y-3">
              {Array.from({ length: 6 }, (_, i) => (
                <div
                  key={`stage-${i}`}
                  className="flex items-center justify-between border-b border-border py-2 last:border-0"
                >
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-4 w-20" />
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {/* ─── Subscriptions ───────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Newspaper className="size-5" />
            Подписки на источники
          </CardTitle>
          <CardDescription>Управление подписками на публикации из OpenAlex и arXiv</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium text-foreground mb-1">Тема поиска</label>
              <Input value={newTopic} onChange={(e) => setNewTopic(e.target.value)} placeholder="Например: machine learning" />
            </div>
            <div className="w-40">
              <label className="block text-sm font-medium text-foreground mb-1">Источник</label>
              <Select value={newSource} onValueChange={setNewSource}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="openalex+arxiv">OpenAlex + arXiv</SelectItem>
                  <SelectItem value="openalex">OpenAlex</SelectItem>
                  <SelectItem value="arxiv">arXiv</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1 min-w-[180px]">
              <label className="block text-sm font-medium text-foreground mb-1">Telegram Chat ID (опционально)</label>
              <Input value={newTelegramChatId} onChange={(e) => setNewTelegramChatId(e.target.value)} placeholder="Например: 123456789" />
            </div>
            <Button onClick={createSubscription} disabled={creating || !newTopic.trim()}>
              <Plus className="size-4 mr-2" />
              Добавить
            </Button>
          </div>

          {createError && (
            <div className="flex items-center gap-2 text-destructive bg-destructive/10 p-2 rounded-lg text-sm">
              <AlertCircle className="size-4 shrink-0" />
              <span>{createError}</span>
            </div>
          )}

          {subError && (
            <div className="flex items-center gap-2 text-destructive bg-destructive/10 p-2 rounded-lg text-sm">
              <AlertCircle className="size-4 shrink-0" />
              <span>{subError}</span>
            </div>
          )}

          <div className="space-y-2">
            {subsLoading && <p className="text-sm text-muted-foreground">Загрузка подписок...</p>}
            {!subsLoading && subscriptions.length === 0 && (
              <p className="text-sm text-subtle-foreground py-4 text-center">Нет активных подписок. Добавьте тему для отслеживания.</p>
            )}
            {subscriptions.map((sub) => (
              <div key={sub.id} className="flex items-center justify-between py-2 px-3 bg-muted rounded-lg">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-foreground">{sub.topic}</span>
                    <Badge variant="outline" className="text-xs">{sub.source}</Badge>
                    {sub.telegram_chat_id && <Badge variant="secondary" className="text-xs"><Mail className="size-3 mr-1" />Telegram</Badge>}
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Создана: {new Date(sub.created_at).toLocaleDateString("ru-RU")}
                    {sub.last_checked_at && ` · Проверена: ${new Date(sub.last_checked_at).toLocaleString("ru-RU")}`}
                  </p>
                </div>
                <Button variant="ghost" size="icon" onClick={() => deleteSubscription(sub.id)} className="text-destructive hover:text-destructive/80">
                  <Trash2 className="size-4" />
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* ─── Notifications ────────────────────────────────── */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bell className="size-5" />
              <CardTitle className="text-lg">Уведомления</CardTitle>
              {unreadCount > 0 && (
                <Badge className="bg-primary">{unreadCount} новых</Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button variant={notifFilter === "all" ? "default" : "outline"} size="sm" onClick={() => setNotifFilter("all")}>
                Все
              </Button>
              <Button variant={notifFilter === "unread" ? "default" : "outline"} size="sm" onClick={() => setNotifFilter("unread")}>
                Непрочитанные
              </Button>
              <Button variant="outline" size="sm" onClick={() => { loadNotifications(); loadUnreadCount(); }} disabled={notifsLoading}>
                <RefreshCw className={`size-3 mr-1 ${notifsLoading ? "animate-spin" : ""}`} />
                Обновить
              </Button>
            </div>
          </div>
          <CardDescription>Ошибки, предупреждения и публикации по отслеживаемым темам</CardDescription>
        </CardHeader>
        <CardContent>
          {notifError && (
            <div className="flex items-center gap-2 text-destructive bg-destructive/10 p-2 rounded-lg text-sm mb-3">
              <AlertCircle className="size-4 shrink-0" />
              <span>{notifError}</span>
            </div>
          )}
          {notifsLoading && <p className="text-sm text-muted-foreground">Загрузка...</p>}
          {!notifsLoading && !notifError && notifications.length === 0 && (
            <p className="text-sm text-subtle-foreground py-4 text-center">Уведомлений пока нет. Создайте подписку, чтобы начать отслеживание.</p>
          )}
          <div className="space-y-2">
            {notifications.map((n) => (
              <div key={n.id} className={`p-3 rounded-lg border ${n.is_read ? "bg-card border-border" : "bg-primary-soft border-primary/30"}`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-medium text-foreground truncate">{n.title}</span>
                      {n.severity === "error" && (
                        <Badge variant="destructive" className="shrink-0"><AlertCircle className="size-3 mr-1" />ошибка</Badge>
                      )}
                      {n.severity === "warning" && (
                        <Badge variant="outline" className="bg-warning-soft text-warning border-warning/30 shrink-0">предупреждение</Badge>
                      )}
                      {n.article_source && <Badge variant="outline" className="text-xs shrink-0">{n.article_source}</Badge>}
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{n.body}</p>
                    <p className="text-xs text-subtle-foreground mt-1">{new Date(n.created_at).toLocaleString("ru-RU")}</p>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    {n.article_url && (
                      <a href={n.article_url} target="_blank" rel="noopener noreferrer">
                        <Button variant="ghost" size="icon" className="size-8">
                          <ExternalLink className="size-4" />
                        </Button>
                      </a>
                    )}
                    {!n.is_read && (
                      <Button variant="ghost" size="icon" className="size-8 text-primary" onClick={() => markRead(n.id)}>
                        <CheckCheck className="size-4" />
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* ─── Grafana ───────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <BarChart3 className="size-5" />
            Grafana
          </CardTitle>
          <CardDescription>Дашборды из Grafana (GRAFANA_URL / GRAFANA_USER / GRAFANA_PASSWORD)</CardDescription>
        </CardHeader>
        <CardContent>
          {grafana === null ? (
            <Skeleton className="h-20 w-full" />
          ) : !grafana.enabled ? (
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <AlertCircle className="size-4 shrink-0" />
              <span>Grafana не сконфигурирована. Задайте GRAFANA_URL (и GRAFANA_USER / GRAFANA_PASSWORD) в окружении.</span>
            </div>
          ) : !grafana.reachable ? (
            <div className="flex items-center gap-3 text-sm text-warning">
              <AlertCircle className="size-4 shrink-0" />
              <span>{grafana.reason || "Grafana недоступна."}</span>
            </div>
          ) : grafana.dashboards.length === 0 ? (
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <BarChart3 className="size-4 shrink-0" />
              <span>Дашборды не найдены.</span>
            </div>
          ) : (
            <div className="divide-y divide-border">
              {grafana.dashboards.map((d) => (
                <div key={d.uid} className="flex items-center justify-between gap-3 py-2 first:pt-0 last:pb-0">
                  <span className="text-sm font-medium text-foreground truncate">{d.title}</span>
                  {d.url ? (
                    <a href={d.url} target="_blank" rel="noopener noreferrer">
                      <Button variant="outline" size="sm">
                        <ExternalLink className="size-4 mr-2" />
                        Открыть
                      </Button>
                    </a>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export const MonitoringTab = memo(MonitoringTabBase);

