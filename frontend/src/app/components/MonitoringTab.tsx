import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { BarChart3, Activity, TrendingUp, AlertCircle, RefreshCw, Server, Cpu, Bell, Newspaper, Trash2, Plus, CheckCheck, ExternalLink, Mail } from "lucide-react";
import { apiFetch } from "../../lib/auth";

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
  subscription_id: string;
  title: string;
  body: string;
  article_url: string | null;
  article_source: string | null;
  is_read: boolean;
  created_at: string;
}

export function MonitoringTab() {
  const [metrics, setMetrics] = useState<Record<string, MetricFamily> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [subsLoading, setSubsLoading] = useState(false);
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [notifsLoading, setNotifsLoading] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  const [newTopic, setNewTopic] = useState("");
  const [newSource, setNewSource] = useState("openalex+arxiv");
  const [newTelegramChatId, setNewTelegramChatId] = useState("");
  const [creating, setCreating] = useState(false);
  const [notifFilter, setNotifFilter] = useState<"all" | "unread">("all");

  const loadMetrics = useCallback(async () => {
    try {
      const r = await apiFetch("/api/admin/monitoring");
      if (!r.ok) throw new Error("Failed to load metrics");
      const data = await r.json();
      setMetrics(data);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadSubscriptions = useCallback(async () => {
    setSubsLoading(true);
    try {
      const r = await apiFetch("/api/subscriptions");
      if (r.ok) {
        const data = await r.json();
        setSubscriptions(data.subscriptions ?? []);
      }
    } finally {
      setSubsLoading(false);
    }
  }, []);

  const loadNotifications = useCallback(async () => {
    setNotifsLoading(true);
    try {
      const query = notifFilter === "unread" ? "/api/notifications?limit=50&unread_only=true" : "/api/notifications?limit=50";
      const r = await apiFetch(query);
      if (r.ok) {
        const data = await r.json();
        setNotifications(data.notifications ?? []);
      }
    } finally {
      setNotifsLoading(false);
    }
  }, [notifFilter]);

  const loadUnreadCount = useCallback(async () => {
    try {
      const r = await apiFetch("/api/notifications/unread-count");
      if (r.ok) {
        const data = await r.json();
        setUnreadCount(data.unread_count);
      }
    } catch {}
  }, []);

  useEffect(() => {
    loadMetrics();
    loadSubscriptions();
    loadNotifications();
    loadUnreadCount();
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      loadMetrics();
      loadUnreadCount();
    }, 15000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadMetrics, loadSubscriptions, loadNotifications, loadUnreadCount]);

  const findMetric = (name: string): MetricSample[] => {
    if (!metrics || !metrics[name]) return [];
    return metrics[name].samples;
  };

  const sumMetric = (name: string): number => {
    return findMetric(name).reduce((s, m) => s + m.value, 0);
  };

  const createSubscription = async () => {
    if (!newTopic.trim()) return;
    setCreating(true);
    try {
      const r = await apiFetch("/api/subscriptions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: newTopic.trim(),
          source: newSource,
          telegram_chat_id: newTelegramChatId.trim() || null,
        }),
      });
      if (r.ok) {
        setNewTopic("");
        setNewTelegramChatId("");
        await loadSubscriptions();
      }
    } finally {
      setCreating(false);
    }
  };

  const deleteSubscription = async (id: string) => {
    try {
      const r = await apiFetch(`/api/subscriptions/${id}`, { method: "DELETE" });
      if (r.ok) {
        setSubscriptions((prev) => prev.filter((s) => s.id !== id));
      }
    } catch {}
  };

  const markRead = async (id: string) => {
    try {
      const r = await apiFetch(`/api/notifications/${id}/read`, { method: "POST" });
      if (r.ok) {
        setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, is_read: true } : n)));
        setUnreadCount((prev) => Math.max(0, prev - 1));
      }
    } catch {}
  };

  const pipelineErrors = findMetric("pipeline_errors_total");
  const ltrMetrics = findMetric("ltr_model_metric");
  const recsGenerated = findMetric("recommendations_generated_total");
  const apiReqCount = findMetric("api_requests_total");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Мониторинг</h2>
          <p className="text-sm text-gray-500">Метрики системы, подписки на источники и уведомления</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-gray-600">
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
        <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg">
          <AlertCircle className="size-5" />
          <span>{error}</span>
        </div>
      )}

      {/* System metrics */}
      {!loading && metrics && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2"><Activity className="size-4" />Pipeline runs</CardDescription>
              </CardHeader>
              <CardContent><div className="text-2xl font-bold">{sumMetric("pipeline_stage_duration_seconds_count")}</div></CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2"><AlertCircle className="size-4" />Pipeline errors</CardDescription>
              </CardHeader>
              <CardContent><div className="text-2xl font-bold text-red-600">{pipelineErrors.reduce((s, m) => s + m.value, 0)}</div></CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2"><TrendingUp className="size-4" />Recommendations</CardDescription>
              </CardHeader>
              <CardContent><div className="text-2xl font-bold text-green-600">{recsGenerated.reduce((s, m) => s + m.value, 0).toFixed(0)}</div></CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription className="flex items-center gap-2"><Server className="size-4" />API requests</CardDescription>
              </CardHeader>
              <CardContent><div className="text-2xl font-bold text-blue-600">{apiReqCount.reduce((s, m) => s + m.value, 0).toFixed(0)}</div></CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg"><BarChart3 className="size-5" />Pipeline stages</CardTitle>
              <CardDescription>Длительность этапов пайплайна</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {(["data_collection", "quality_scoring", "skill_extraction", "weight_cleaning", "level_building", "cluster_training", "model_training", "gap_analysis"] as const).map((stage) => {
                  const countSamples = findMetric("pipeline_stage_duration_seconds_count").filter(s => s.labels.stage === stage);
                  const sumSamples = findMetric("pipeline_stage_duration_seconds_sum").filter(s => s.labels.stage === stage);
                  const count = countSamples.reduce((s, m) => s + m.value, 0);
                  const total = sumSamples.reduce((s, m) => s + m.value, 0);
                  const avg = count > 0 ? total / count : 0;
                  return (
                    <div key={stage} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
                      <span className="text-sm font-medium text-gray-700 capitalize">{stage.replace(/_/g, " ")}</span>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-gray-500">{count}x</span>
                        <span className="font-mono text-gray-700">{avg.toFixed(1)}s avg</span>
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
                    <div key={m.labels.metric} className="text-center p-4 bg-gray-50 rounded-lg">
                      <div className="text-xs text-gray-500 uppercase">{m.labels.metric}</div>
                      <div className="text-xl font-bold text-gray-900">{m.value.toFixed(4)}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {pipelineErrors.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg text-red-700"><AlertCircle className="size-5" />Pipeline errors by stage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {pipelineErrors.map((m, i) => (
                    <div key={i} className="flex items-center justify-between py-1">
                      <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">{m.labels.stage}</Badge>
                      <span className="font-mono text-red-600">{m.value}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          <details className="text-sm text-gray-500">
            <summary className="cursor-pointer hover:text-gray-700">Сырые метрики</summary>
            <pre className="mt-2 p-4 bg-gray-50 rounded-lg overflow-x-auto text-xs">
              {JSON.stringify(metrics, null, 2)}
            </pre>
          </details>
        </>
      )}

      {loading && !metrics && (
        <div className="text-center py-12 text-gray-500">Загрузка метрик...</div>
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
              <label className="block text-sm font-medium text-gray-700 mb-1">Тема поиска</label>
              <Input value={newTopic} onChange={(e) => setNewTopic(e.target.value)} placeholder="Например: machine learning" />
            </div>
            <div className="w-40">
              <label className="block text-sm font-medium text-gray-700 mb-1">Источник</label>
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
              <label className="block text-sm font-medium text-gray-700 mb-1">Telegram Chat ID (опционально)</label>
              <Input value={newTelegramChatId} onChange={(e) => setNewTelegramChatId(e.target.value)} placeholder="Например: 123456789" />
            </div>
            <Button onClick={createSubscription} disabled={creating || !newTopic.trim()}>
              <Plus className="size-4 mr-2" />
              Добавить
            </Button>
          </div>

          <div className="space-y-2">
            {subsLoading && <p className="text-sm text-gray-500">Загрузка подписок...</p>}
            {!subsLoading && subscriptions.length === 0 && (
              <p className="text-sm text-gray-400 py-4 text-center">Нет активных подписок. Добавьте тему для отслеживания.</p>
            )}
            {subscriptions.map((sub) => (
              <div key={sub.id} className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-900">{sub.topic}</span>
                    <Badge variant="outline" className="text-xs">{sub.source}</Badge>
                    {sub.telegram_chat_id && <Badge variant="secondary" className="text-xs"><Mail className="size-3 mr-1" />Telegram</Badge>}
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5">
                    Создана: {new Date(sub.created_at).toLocaleDateString("ru-RU")}
                    {sub.last_checked_at && ` · Проверена: ${new Date(sub.last_checked_at).toLocaleString("ru-RU")}`}
                  </p>
                </div>
                <Button variant="ghost" size="icon" onClick={() => deleteSubscription(sub.id)} className="text-red-500 hover:text-red-700">
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
                <Badge className="bg-blue-600">{unreadCount} новых</Badge>
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
          <CardDescription>Новые публикации по отслеживаемым темам</CardDescription>
        </CardHeader>
        <CardContent>
          {notifsLoading && <p className="text-sm text-gray-500">Загрузка...</p>}
          {!notifsLoading && notifications.length === 0 && (
            <p className="text-sm text-gray-400 py-4 text-center">Уведомлений пока нет. Создайте подписку, чтобы начать отслеживание.</p>
          )}
          <div className="space-y-2">
            {notifications.map((n) => (
              <div key={n.id} className={`p-3 rounded-lg border ${n.is_read ? "bg-white border-gray-200" : "bg-blue-50 border-blue-200"}`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-900 truncate">{n.title}</span>
                      {n.article_source && <Badge variant="outline" className="text-xs shrink-0">{n.article_source}</Badge>}
                    </div>
                    <p className="text-xs text-gray-500 mt-1 line-clamp-2">{n.body}</p>
                    <p className="text-xs text-gray-400 mt-1">{new Date(n.created_at).toLocaleString("ru-RU")}</p>
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
                      <Button variant="ghost" size="icon" className="size-8 text-blue-600" onClick={() => markRead(n.id)}>
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
    </div>
  );
}


