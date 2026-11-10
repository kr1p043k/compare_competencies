import { useState, useEffect, useRef } from "react";
import { FileUp } from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";
import {
  AlertCircle, RefreshCw, Users, FileText, Database,
  Upload, Brain, BookOpen, Download, Clock, Activity, Shield,
} from "lucide-react";
import { logAction } from "../../lib/auth";
import { reportError } from "./../../lib/errors";
import { ApiError, request } from "../../lib/api";
import { AdminSecurity } from "./AdminSecurity";
import { AdminAuditTab } from "./AdminAuditTab";
import { AdminUsersTab } from "./AdminUsersTab";
import { cn } from "./ui/utils";
import {
  Badge,
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  inputClass,
  PageHeader,
  Progress,
  StatCard,
  TableFrame,
  tableHeadClass,
  tableRowClass,
} from "../../design";

import { AdminLlmCostCard } from "./AdminLlmCostCard";

export function AdminDashboard() {
  const RPD_STAGE_LABELS: Record<string, string> = {
    saved: "файл сохранён",
    parse: "разбор PDF и извлечение компетенций",
    collect: "скачивание аннотаций с Yandex Disk",
    merge: "слияние аннотаций в KRM",
    seed: "запись направления в базу данных",
    analysis: "пересчёт анализа направления",
    done: "готово",
    error: "ошибка",
  };
  const rpdStageLabel = (stage?: string): string => {
    if (!stage) return "...";
    return RPD_STAGE_LABELS[stage] ?? stage;
  };
  const [users, setUsers] = useState<any[]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState("users");
  const [sched, setSched] = useState<any>(null);
  const [schedLoading, setSchedLoading] = useState(false);
  const [schedMsg, setSchedMsg] = useState("");
  const [schedInterval, setSchedInterval] = useState("12");
  const [schedBusy, setSchedBusy] = useState(false);
  const [health, setHealth] = useState<any>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthMsg, setHealthMsg] = useState("");
  const loadHealth = async () => {
    setHealthLoading(true);
    setHealthMsg("");
    try {
      const d = await request<Record<string, unknown>>("/health");
      setHealth(d);
      setHealthMsg("Сервис отвечает");
    } catch (e: any) {
      setHealthMsg(e?.message || "Ошибка проверки");
    } finally {
      setHealthLoading(false);
    }
  };
  const loadSched = async () => {
    setSchedLoading(true);
    try {
      const d = await request<any>("/api/admin/scheduler/status");
      setSched(d);
      if (d && d.collect_interval_hours != null) setSchedInterval(String(d.collect_interval_hours));
    } catch (e: any) {
      setSchedMsg(e?.message || "Ошибка загрузки");
    } finally {
      setSchedLoading(false);
    }
  };
  const saveSched = async (patch: any) => {
    setSchedBusy(true);
    setSchedMsg("");
    try {
      const d = await request<any>("/api/admin/scheduler/settings", { method: "PUT", body: patch });
      setSched(d);
      if (d && d.collect_interval_hours != null) setSchedInterval(String(d.collect_interval_hours));
      setSchedMsg("Сохранено");
    } catch (e: any) {
      setSchedMsg(e?.message || "Ошибка сохранения");
    } finally {
      setSchedBusy(false);
    }
  };
  const runSchedJob = async (job: string, label: string) => {
    if (!window.confirm(`Запустить "${label}" вне расписания?`)) return;
    setSchedBusy(true);
    setSchedMsg("");
    try {
      await request<any>(`/api/admin/scheduler/${job}/now`, { method: "POST" });
      setSchedMsg("Запущено");
      setTimeout(loadSched, 2000);
    } catch (e: any) {
      setSchedMsg(e?.message || "Ошибка запуска");
    } finally {
      setSchedBusy(false);
    }
  };
  useEffect(() => { if (tab === "sched") loadSched(); }, [tab]);
  const [sugList, setSugList] = useState<any[]>([]);
  const [sugLoading, setSugLoading] = useState(false);
  const [sugMsg, setSugMsg] = useState("");
  const [sugCats, setSugCats] = useState<Record<string, string>>({});
  const [taxCats, setTaxCats] = useState<{ id: string; label: string }[]>([]);
  const loadSug = async () => {
    setSugLoading(true);
    setSugMsg("");
    try {
      const d = await request<any>("/api/admin/skills/suggestions?status=pending");
      setSugList(d.suggestions || []);
      if (taxCats.length === 0) {
        const t = await request<any>("/api/taxonomy/categories").catch(() => null);
        if (t?.categories) setTaxCats(t.categories.map((c: any) => ({ id: c.id, label: c.label })));
      }
    } catch (e: any) {
      setSugMsg(e?.message || "Ошибка загрузки");
    } finally {
      setSugLoading(false);
    }
  };
  useEffect(() => { if (tab === "skills") loadSug(); }, [tab]);
  const decideSug = async (id: string, approve: boolean) => {
    const cat = sugCats[id] || "";
    if (approve && !cat) { setSugMsg("Выберите категорию для одобрения"); return; }
    try {
      await request<any>(`/api/admin/skills/suggestions/${id}/${approve ? "approve" : "reject"}`, {
        method: "POST",
        body: { category: cat },
      });
      setSugList((prev) => prev.filter((s) => s.id !== id));
      setSugMsg(approve ? `Принят в таксономию (${cat})` : "Отклонён");
    } catch (e: any) {
      setSugMsg("Ошибка: " + (e?.message || e));
    }
  };
  const fmtTs = (ts: any) => {
    if (ts == null) return "-";
    const d = new Date(Number(ts) * 1000);
    return isNaN(d.getTime()) ? "-" : d.toLocaleString("ru-RU");
  };
  const [logFilter, setLogFilter] = useState("all");
  const [seedLoading, setSeedLoading] = useState(false);
  const [embLoading, setEmbLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [backupLoading, setBackupLoading] = useState(false);
  const [backupMsg, setBackupMsg] = useState("");
  const [extLoading, setExtLoading] = useState(false);
  const [seedMsg, setSeedMsg] = useState("");
  const [embMsg, setEmbMsg] = useState("");
  const [exportMsg, setExportMsg] = useState("");
  const [extMsg, setExtMsg] = useState("");
  const [impMsg, setImpMsg] = useState("");
  const [uncategorized, setUncategorized] = useState<any[]>([]);
  const [uncatLoading, setUncatLoading] = useState(false);
  const [uncatMsg, setUncatMsg] = useState("");
  const [uncatSelections, setUncatSelections] = useState<Record<string, string>>({});
  const [linkSkill, setLinkSkill] = useState<string | null>(null);
  const [compQuery, setCompQuery] = useState("");
  const [compResults, setCompResults] = useState<any[]>([]);
  const [compLoading, setCompLoading] = useState(false);
  const [linkKsa, setLinkKsa] = useState("skills");
  const [linkCompId, setLinkCompId] = useState("");
  const [linkMsg, setLinkMsg] = useState("");
  const [linkSaving, setLinkSaving] = useState(false);

  const searchCompetencies = async (q: string) => {
    setCompQuery(q);
    if (!q.trim()) { setCompResults([]); return; }
    setCompLoading(true);
    try {
      const d = await request<any>(
        `/api/teacher/zun/competencies/search?q=${encodeURIComponent(q)}&dir_code=09.03.02&limit=20`,
      );
      setCompResults(d.items || []);
    } catch (e: any) {
      setLinkMsg("Поиск: " + e.message);
    } finally {
      setCompLoading(false);
    }
  };

  const saveSkillLink = async () => {
    if (!linkSkill || !linkCompId) { setLinkMsg("Выберите компетенцию из поиска"); return; }
    setLinkSaving(true); setLinkMsg("");
    try {
      const d = await request<any>(`/api/teacher/zun/competencies/${linkCompId}/skills`, {
        method: "POST",
        body: { skill_name: linkSkill, ksa_type: linkKsa },
      });
      setLinkMsg(`Привязано: ${d.skill_name} → explicit. Учтётся в следующем teacher-анализе.`);
      setLinkSkill(null); setCompQuery(""); setCompResults([]); setLinkCompId("");
    } catch (e: any) {
      setLinkMsg("Ошибка: " + e.message);
    } finally {
      setLinkSaving(false);
    }
  };
  const [directions, setDirections] = useState<any[]>([]);
  const [importJson, setImportJson] = useState("");
  const [rpdDir, setRpdDir] = useState("09.03.02");
  const [rpdFile, setRpdFile] = useState<File | null>(null);
  const rpdInputRef = useRef<HTMLInputElement | null>(null);
  const [rpdUploading, setRpdUploading] = useState(false);
  const [rpdCollecting, setRpdCollecting] = useState(false);
  const [rpdYandexUrl, setRpdYandexUrl] = useState("");
  const [collectCooldown, setCollectCooldown] = useState(0);
  const [rpdRunId, setRpdRunId] = useState<string | null>(null);
  const [rpdStatus, setRpdStatus] = useState<any>(null);
  const [rpdMsg, setRpdMsg] = useState("");
  const [rpdSources, setRpdSources] = useState<{ yandex_covered: string[] }>({ yandex_covered: [] });

  const loadData = async () => {
    setLoading(true); setError(null);
    try {
      const [uData, lData, dData] = await Promise.all([
        request<any>("/api/admin/users"),
        request<any>("/api/admin/logs?limit=200"),
        request<any>("/api/admin/directions"),
      ]);
      setUsers(uData.users || []);
      setLogs(lData.logs || []);
      setDirections(Array.isArray(dData) ? dData : []);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadData(); }, []);

  // Возобновление поллинга после переключения вкладок (табы размонтируют компонент).
  useEffect(() => {
    let stored: string | null = null;
    try { stored = sessionStorage.getItem("rpdRunId"); } catch (err) { reportError("app/components/AdminDashboard.tsx:258", err); }
    if (stored && !rpdRunId) {
      setRpdRunId(stored);
      setRpdCollecting(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (collectCooldown <= 0) return;
    const t = setTimeout(() => setCollectCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => clearTimeout(t);
  }, [collectCooldown]);

  useEffect(() => {
    request<any>("/api/teacher/rpd/sources")
      .then((d) => { if (d) setRpdSources(d); })
      .catch((err) => reportError("app/components/AdminDashboard.tsx", err));
  }, []);

  useEffect(() => {
    if (!rpdRunId) return;
    let cancelled = false;
    let misses = 0;
    const poll = () => {
      request<any>(`/api/teacher/rpd/status/${rpdRunId}`)
        .catch(() => null)
        .then((s) => {
          if (cancelled) return;
          if (!s) {
            // Разовый сбой сети/рестарт сервера - не умираем молча, ждём.
            misses += 1;
            if (misses < 10) {
              setRpdMsg(`Сервер перезапускается, жду статус... (${misses})`);
              setTimeout(poll, 5000);
              return;
            }
            setRpdUploading(false);
            setRpdCollecting(false);
            setRpdMsg("Ошибка: статус задачи недоступен. Проверьте сервер и повторите.");
            return;
          }
          misses = 0;
          setRpdStatus(s);
          if (s.status === "completed" || s.status === "failed" || s.status === "cancelled") {
            setRpdUploading(false);
            setRpdCollecting(false);
            try { sessionStorage.removeItem("rpdRunId"); } catch (err) { reportError("app/components/AdminDashboard.tsx:306", err); }
            setRpdMsg(s.status === "completed" ? "Готово" : s.status === "cancelled" ? "Отменено пользователем" : `Ошибка: ${s.error || "unknown"}`);
          } else {
            setTimeout(poll, 3000);
          }
        })
        .catch(() => {
          if (cancelled) return;
          misses += 1;
          if (misses < 10) {
            setTimeout(poll, 5000);
          } else {
            setRpdUploading(false);
            setRpdCollecting(false);
            setRpdMsg("Ошибка: нет связи с сервером. Проверьте сервер и повторите.");
          }
        });
    };
    poll();
    return () => { cancelled = true; };
  }, [rpdRunId]);

  async function uploadRpd(file?: File | null) {
    const f = file ?? rpdFile;
    if (!f || rpdUploading) return;
    setRpdUploading(true); setRpdMsg("Загрузка...");
    const fd = new FormData();
    fd.append("file", f);
    fd.append("dir_code", rpdDir);
    try {
      const d = await request<{ run_id?: string }>("/api/teacher/rpd/upload", {
        method: "POST",
        body: fd,
      });
      if (!d.run_id) throw new Error("Сервер не вернул ID задачи");
      try { sessionStorage.setItem("rpdRunId", d.run_id); } catch (err) { reportError("app/components/AdminDashboard.tsx:341", err); }
      setRpdRunId(d.run_id);
      setRpdMsg("Обработка запущена...");
      setRpdFile(null);
    } catch (e: any) {
      setRpdMsg("Ошибка: " + e.message);
      setRpdUploading(false);
    }
  }

  async function collectRpd() {
    if (collectCooldown > 0) return;
    if (rpdCollecting) {
      // Повторный клик во время сбора = отмена.
      if (!rpdRunId) {
        setRpdCollecting(false);
        setRpdMsg("");
        return;
      }
      try {
        setRpdMsg("Останавливаю сбор...");
        await request<void>(`/api/teacher/rpd/cancel/${rpdRunId}`, { method: "POST" });
      } catch {
        setRpdCollecting(false);
        setRpdMsg("");
      }
      return;
    }
    setRpdCollecting(true); setRpdMsg("Сбор аннотаций с Yandex Disk...");
    try {
      const form = new URLSearchParams({ dir_code: rpdDir });
      if (rpdYandexUrl.trim()) form.set("public_url", rpdYandexUrl.trim());
      let d: { run_id?: string };
      try {
        d = await request<{ run_id?: string }>("/api/teacher/rpd/collect", {
          method: "POST",
          body: form,
        });
      } catch (err) {
        if (err instanceof ApiError && err.isRateLimited) {
          setCollectCooldown(30);
          throw new Error("Слишком частые запросы - кнопка заблокирована на 30 секунд");
        }
        throw err;
      }
      if (!d.run_id) throw new Error("Сервер не вернул ID задачи");
      try { sessionStorage.setItem("rpdRunId", d.run_id); } catch (err) { reportError("app/components/AdminDashboard.tsx:383", err); }
      setRpdRunId(d.run_id);
      setRpdMsg("Сбор и обработка запущены...");
    } catch (e: any) {
      setRpdMsg("Ошибка: " + e.message);
      setRpdCollecting(false);
    }
  }

  const filteredLogs = logFilter === "all" ? logs : logs.filter((l) => l.user === logFilter);

  const callAction = async (url: string, body: any, setMsg: (s: string) => void, setLoad: (b: boolean) => void) => {
    setLoad(true); setMsg(""); logAction(url);
    try {
      const d = await request<any>(url, { method: "POST", body });
      setMsg(d?.message || d?.status || "Done");
    } catch (e: any) {
      setMsg("Error: " + e.message);
    } finally {
      setLoad(false);
    }
  };

  const callExport = async () => {
    logAction("/api/admin/export/db");
    setExportLoading(true); setExportMsg("");
    try {
      const d = await request<any>("/api/admin/export/db");
      setExportMsg(d?.message || "Done");
    } catch (e: any) { setExportMsg("Error: " + e.message); }
    finally { setExportLoading(false); }
  };

  const importStudents = async () => {
    setImpMsg("");
    try {
      const data = JSON.parse(importJson);
      const d = await request<any>("/api/admin/students/import", {
        method: "POST",
        body: data,
      });
      setImpMsg(`Imported: ${d.imported || "?"} students`);
    } catch (e: any) {
      setImpMsg("Error: " + e.message);
    }
  };

  const loadUncategorized = async () => {
    setUncatLoading(true); setUncatMsg("");
    try {
      const d = await request<any>("/api/admin/skills/uncategorized");
      setUncategorized(d.items || []);
      const sel: Record<string, string> = {};
      (d.items || []).forEach((it: any) => { sel[it.skill] = it.category; });
      setUncatSelections(sel);
    } catch (e: any) {
      setUncatMsg("Ошибка: " + e.message);
    } finally {
      setUncatLoading(false);
    }
  };

  const applyCategorization = async () => {
    setUncatMsg("");
    const assignments = uncategorized.map((it) => ({
      skill: it.skill,
      category: uncatSelections[it.skill] || it.category,
    }));
    try {
      const d = await request<any>("/api/admin/skills/categorize", {
        method: "POST",
        body: { assignments },
      });
      setUncatMsg(`Распределено навыков: ${d.added}`);
      await loadUncategorized();
    } catch (e: any) {
      setUncatMsg("Ошибка: " + e.message);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Shield}
        title="Администрирование"
        description="Пользователи, планировщик, импорт данных и категории навыков."
      />
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">Управление системой</p>
        <Button variant="secondary" size="sm" onClick={loadData} disabled={loading}
          iconLeft={<RefreshCw className={cn("size-4", loading && "animate-spin")} />}>
          Обновить
        </Button>
      </div>

      {error && (
        <Callout role="alert" tone="danger" icon={<AlertCircle className="size-4" aria-hidden="true" />}>
          {error}
        </Callout>
      )}

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="flex-wrap">
          <TabsTrigger value="users"><Users className="size-4 mr-2" />Пользователи</TabsTrigger>
          <TabsTrigger value="logs"><FileText className="size-4 mr-2" />Логи</TabsTrigger>
          <TabsTrigger value="db"><Database className="size-4 mr-2" />БД</TabsTrigger>
          <TabsTrigger value="import"><Upload className="size-4 mr-2" />Импорт</TabsTrigger>
          <TabsTrigger value="skills"><Brain className="size-4 mr-2" />Навыки</TabsTrigger>
          <TabsTrigger value="sched"><Clock className="size-4 mr-2" />Фоновые задачи</TabsTrigger>
          <TabsTrigger value="health"><Activity className="size-4 mr-2" />Проверка</TabsTrigger>
          <TabsTrigger value="audit"><FileText className="size-4 mr-2" />Аудит</TabsTrigger>
          <TabsTrigger value="security"><Shield className="size-4 mr-2" />Безопасность</TabsTrigger>
        </TabsList>

        {/* ── Users tab ── */}
        <TabsContent value="users" className="space-y-4">
          <AdminUsersTab />
        </TabsContent>

        <TabsContent value="logs" className="space-y-4">
          <Card>
            <CardHeader
              title="История запросов"
              actions={
                <select value={logFilter} onChange={(e) => setLogFilter(e.target.value)}
                  className={cn(inputClass, "h-9 w-48")}>
                  <option value="all">Все пользователи</option>
                  {users.map((u) => <option key={u.username} value={u.username}>{u.name}</option>)}
                </select>
              }
            />
            <CardBody>
              <TableFrame className="max-h-96 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className={cn(tableHeadClass, "sticky top-0 bg-card")}>
                      <th className="pb-2 text-left font-semibold">Время</th>
                      <th className="pb-2 text-left font-semibold">Пользователь</th>
                      <th className="pb-2 text-left font-semibold">Метод</th>
                      <th className="pb-2 text-left font-semibold">Путь</th>
                      <th className="pb-2 text-left font-semibold">Действие/Детали</th>
                      <th className="pb-2 text-right font-semibold">Статус</th>
                      <th className="pb-2 text-right font-semibold">мс</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredLogs.toReversed().slice(0, 200).map((l, i) => (
                      <tr key={i} className={tableRowClass}>
                        <td className="py-1.5 text-xs text-muted-foreground whitespace-nowrap">{new Date(l.timestamp).toLocaleTimeString()}</td>
                        <td className="py-1.5 text-foreground">{l.user}</td>
                        <td className="py-1.5">
                          <Badge
                            tone={
                              l.method === "GET" ? "success" :
                              l.method === "POST" ? "primary" :
                              l.method === "AUDIT" ? "info" : "neutral"
                            }
                            className="font-mono"
                          >{l.method}</Badge>
                        </td>
                        <td className="py-1.5 text-xs text-muted-foreground max-w-xs truncate">{l.path}</td>
                        <td className="py-1.5 text-xs text-muted-foreground max-w-md truncate" title={l.detail || `${l.method} ${l.path}`}>{l.detail || `${l.method} ${l.path}`}</td>
                        <td className="py-1.5 text-right">
                          <span className={cn("text-xs font-mono",
                            l.status < 300 ? "text-success" : l.status < 400 ? "text-warning" : "text-destructive"
                          )}>{l.status}</span>
                        </td>
                        <td className="py-1.5 text-right text-xs text-muted-foreground">{l.duration_ms.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableFrame>
            </CardBody>
          </Card>
        </TabsContent>

        {/* ── DB tab ── */}
        <TabsContent value="db" className="space-y-4">
          <Card>
            <CardHeader
              title={
                <span className="flex items-center gap-2">
                  <Database className="size-4" aria-hidden="true" />
                  Управление БД
                </span>
              }
            />
            <CardBody className="space-y-4">
              <div className="flex items-center gap-4">
                <Button onClick={() => callAction("/api/admin/db/seed", { drop: false }, setSeedMsg, setSeedLoading)} disabled={seedLoading}>
                  {seedLoading ? "..." : "Заполнить БД"}
                </Button>
                <Button variant="secondary" onClick={() => callAction("/api/admin/db/seed", { drop: true }, setSeedMsg, setSeedLoading)} disabled={seedLoading}>
                  Drop + Seed
                </Button>
                {seedMsg && <span className="text-sm text-muted-foreground">{seedMsg}</span>}
              </div>
              <div className="flex items-center gap-4">
                <Button onClick={() => callAction("/api/admin/embeddings/generate", { force: false }, setEmbMsg, setEmbLoading)} disabled={embLoading} iconLeft={<Brain className="size-4" />}>
                  {embLoading ? "..." : "Сгенерировать эмбеддинги"}
                </Button>
                <Button variant="secondary" onClick={() => callAction("/api/admin/embeddings/generate", { force: true }, setEmbMsg, setEmbLoading)} disabled={embLoading}>
                  Перегенерировать принудительно
                </Button>
                {embMsg && <span className="text-sm text-muted-foreground">{embMsg}</span>}
              </div>
              <div className="flex items-center gap-4">
                <Button onClick={callExport} disabled={exportLoading} iconLeft={<FileText className="size-4" />}>
                  {exportLoading ? "..." : "Экспорт БД в JSON"}
                </Button>
                {exportMsg && <span className="text-sm text-muted-foreground">{exportMsg}</span>}
              </div>
              <div className="flex items-center gap-4">
                <Button onClick={() => callAction("/api/admin/db/backup", {}, setBackupMsg, setBackupLoading)} disabled={backupLoading}>
                  {backupLoading ? "..." : "Бэкап БД (pg_dump)"}
                </Button>
                {backupMsg && <span className="text-sm text-muted-foreground">{backupMsg}</span>}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHeader
              title={
                <span className="flex items-center gap-2">
                  <FileUp className="size-4" aria-hidden="true" />
                  Подгрузка РПД
                </span>
              }
            />
            <CardBody className="space-y-3">
              <div className="flex items-center gap-3">
                <select
                  value={rpdDir}
                  onChange={(e) => setRpdDir(e.target.value)}
                  className={cn(inputClass, "h-9 w-auto font-mono")}
                >
                  {directions.map((d: any) => (
                    <option key={d.dir_code} value={d.dir_code}>{d.dir_code}</option>
                  ))}
                </select>
                <input
                  ref={rpdInputRef}
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0] || null;
                    setRpdFile(f);
                    e.target.value = "";
                    if (f) uploadRpd(f);
                  }}
                />
              </div>
              <div className="flex items-center gap-3">
                <Button onClick={() => rpdInputRef.current?.click()} disabled={rpdUploading} iconLeft={<Upload className="size-4" />}>
                  {rpdUploading ? "Обработка..." : "Загрузить PDF и обработать"}
                </Button>
                {rpdSources.yandex_covered?.includes(rpdDir) && (
                  <Button onClick={collectRpd} disabled={collectCooldown > 0} variant={rpdCollecting ? "danger" : "secondary"}>
                    {rpdCollecting ? "Отмена" : collectCooldown > 0 ? `Подождите ${collectCooldown} сек` : "Собрать с Yandex Disk"}
                  </Button>
                )}
              </div>
              <div className="flex items-center gap-3">
                <input
                  type="text"
                  value={rpdYandexUrl}
                  onChange={(e) => setRpdYandexUrl(e.target.value)}
                  placeholder="Ссылка на папку Yandex Disk (https://disk.360.yandex.ru/d/...)"
                  className={cn(inputClass, "h-9 flex-1")}
                />
                <Button onClick={collectRpd} disabled={rpdCollecting || collectCooldown > 0 || !rpdYandexUrl.trim()} iconLeft={<Download className="size-4" />}>
                  {rpdCollecting ? "Сбор..." : "Загрузить по ссылке"}
                </Button>
              </div>
              {rpdStatus && (rpdStatus.status === "running" || rpdStatus.status === "started") && (
                <div>
                  <Progress value={rpdStatus.stats?.progress ?? 10} label="Обработка РПД" />
                  <p className="mt-1 text-sm text-warning">
                    Этап: {rpdStageLabel(rpdStatus.stats?.stage)} · {rpdStatus.stats?.progress ?? 10}%
                  </p>
                </div>
              )}
              {rpdMsg && <p className="text-sm text-muted-foreground">{rpdMsg}</p>}
            </CardBody>
          </Card>
        </TabsContent>

        {/* ── Import tab ── */}
        <TabsContent value="import" className="space-y-4">
          <Card>
            <CardHeader
              title={
                <span className="flex items-center gap-2">
                  <Upload className="size-4" aria-hidden="true" />
                  Импорт студентов
                </span>
              }
            />
            <CardBody className="space-y-3">
              <p className="text-sm text-muted-foreground">JSON-массив студентов:</p>
              <textarea
                value={importJson}
                onChange={(e) => setImportJson(e.target.value)}
                rows={8}
                className={cn(inputClass, "h-auto p-3 font-mono")}
                placeholder={`[
  {"full_name":"Иванов Иван","group_name":"ИСИТ-31","skills":"python,sql"}
]`}
              />
              <Button onClick={importStudents} disabled={!importJson}>Импортировать</Button>
              {impMsg && <p className="text-sm text-success">{impMsg}</p>}
            </CardBody>
          </Card>
        </TabsContent>

        {/* ── Skills tab ── */}
        <TabsContent value="skills" className="space-y-4">
          <Card>
            <CardHeader
              title="Предложения от преподавателей"
              actions={
                <Button variant="secondary" size="sm" onClick={loadSug} disabled={sugLoading} iconLeft={<RefreshCw className={cn("size-4", sugLoading && "animate-spin")} />}>
                  Обновить
                </Button>
              }
            />
            <CardBody className="space-y-3">
              {sugMsg && <p className="text-sm text-muted-foreground">{sugMsg}</p>}
              {!sugLoading && sugList.length === 0 && (
                <p className="text-sm text-success">Очередь пуста.</p>
              )}
              {sugList.map((s) => (
                <div key={s.id} className="flex flex-col sm:flex-row sm:items-center gap-2 rounded-lg border border-border p-3">
                  <div className="flex-1 min-w-0">
                    <span className="font-medium text-sm text-foreground">{s.skill}</span>
                    <span className="ml-2 text-xs text-muted-foreground">
                      от {s.created_by || "?"} · {s.created_at || ""}
                      {s.category_hint ? ` · подсказка: ${s.category_hint}` : ""}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <select
                      value={sugCats[s.id] || ""}
                      onChange={(e) => setSugCats((p) => ({ ...p, [s.id]: e.target.value }))}
                      className={cn(inputClass, "h-9 w-auto px-2")}
                    >
                      <option value="">Категория...</option>
                      {taxCats.map((c) => (
                        <option key={c.id} value={c.id}>{c.label}</option>
                      ))}
                    </select>
                    <Button size="sm" variant="success" onClick={() => decideSug(s.id, true)}>
                      Принять
                    </Button>
                    <Button size="sm" variant="secondary" onClick={() => decideSug(s.id, false)}>
                      Отклонить
                    </Button>
                  </div>
                </div>
              ))}
            </CardBody>
          </Card>
          <Card>
            <CardHeader
              title={
                <span className="flex items-center gap-2">
                  <Brain className="size-4" aria-hidden="true" />
                  Расширение таксономии
                </span>
              }
            />
            <CardBody className="space-y-3">
              <p className="text-sm text-muted-foreground">Анализ вакансий и добавление новых навыков в it_skills.json</p>
              <Button onClick={() => callAction("/api/admin/skills/extend", { yes: true }, setExtMsg, setExtLoading)} disabled={extLoading}>
                {extLoading ? "..." : "Анализировать и добавить"}
              </Button>
              {extMsg && <p className="text-sm text-success">{extMsg}</p>}
            </CardBody>
          </Card>

          <Card>
            <CardHeader
              title={
                <span className="flex items-center gap-2">
                  <BookOpen className="size-4" aria-hidden="true" />
                  Некатегоризованные навыки
                </span>
              }
              actions={
                <Button variant="secondary" size="sm" onClick={loadUncategorized} disabled={uncatLoading} iconLeft={<RefreshCw className={cn("size-4", uncatLoading && "animate-spin")} />}>
                  Проверить
                </Button>
              }
            />
            <CardBody className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Навыки из it_skills, не имеющие категории в таксономии. Предложена категория по эмбеддинг-сходству.
              </p>
              {uncatLoading && <p className="text-sm text-muted-foreground">Загрузка...</p>}
              {!uncatLoading && uncategorized.length === 0 && (
                <p className="text-sm text-success">Все навыки категоризованы.</p>
              )}
              {!uncatLoading && uncategorized.length > 0 && (
                <>
                  <TableFrame className="max-h-80 overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className={cn(tableHeadClass, "sticky top-0 bg-card")}>
                          <th className="px-3 py-2 text-left">Навык</th>
                          <th className="px-3 py-2 text-left">Категория</th>
                          <th className="px-3 py-2 text-right">Score</th>
                          <th className="px-3 py-2 text-right">В БД</th>
                        </tr>
                      </thead>
                      <tbody>
                        {uncategorized.map((it) => (
                          <tr key={it.skill} className={tableRowClass}>
                            <td className="px-3 py-1.5 font-mono text-xs text-foreground">
                              {it.skill}
                              {it.manual && <Badge tone="neutral" className="ml-2">manual</Badge>}
                            </td>
                            <td className="px-3 py-1.5">
                              <select
                                value={uncatSelections[it.skill] || it.category}
                                onChange={(e) => setUncatSelections((prev) => ({ ...prev, [it.skill]: e.target.value }))}
                                className={cn(inputClass, "h-8 w-auto px-2 text-xs")}
                              >
                                {["programming_languages","frameworks","databases","devops","cloud","data_science","ml_advanced","frontend","mobile","testing_qa","security","llm_ai","enterprise","gis","embedded","game_dev","management","soft_skills","mathematics","methodologies_concepts","business_tools","methodologies","abstract_concepts"].map((c) => (
                                  <option key={c} value={c}>{c}</option>
                                ))}
                              </select>
                            </td>
                            <td className="px-3 py-1.5 text-right text-xs text-muted-foreground">{it.score.toFixed(3)}</td>
                            <td className="px-3 py-1.5 text-right">
                              <Button
                                variant="secondary"
                                size="sm"
                                className="h-7 text-xs"
                                onClick={() => { setLinkSkill(it.skill); setLinkMsg(""); setCompQuery(""); setCompResults([]); setLinkCompId(""); }}
                              >
                                В комп.
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </TableFrame>
                  {linkSkill && (
                    <div className="rounded-lg border border-success/30 bg-success-soft p-3 space-y-2">
                      <div className="text-sm font-medium text-foreground">
                        Привязать <span className="font-mono text-success">{linkSkill}</span> к компетенции (запись в БД, match=exact)
                      </div>
                      <div className="flex flex-col md:flex-row gap-2">
                        <input
                          value={compQuery}
                          onChange={(e) => searchCompetencies(e.target.value)}
                          placeholder="Поиск компетенции: код или дисциплина..."
                          className={cn(inputClass, "h-9")}
                        />
                        <select
                          value={linkKsa}
                          onChange={(e) => setLinkKsa(e.target.value)}
                          className={cn(inputClass, "h-9 w-auto px-2 text-xs")}
                        >
                          {["knowledge", "abilities", "skills", "flat"].map((k) => (
                            <option key={k} value={k}>{k}</option>
                          ))}
                        </select>
                      </div>
                      {compLoading && <p className="text-xs text-muted-foreground">Поиск...</p>}
                      {compResults.length > 0 && (
                        <div className="max-h-40 overflow-y-auto rounded-lg border border-border bg-card text-sm">
                          {compResults.map((c: any) => (
                            <button
                              key={c.id}
                              type="button"
                              onClick={() => setLinkCompId(c.id)}
                              className={cn(
                                "w-full text-left px-3 py-1.5 cursor-pointer transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                                linkCompId === c.id
                                  ? "bg-success-soft font-medium"
                                  : "hover:bg-success-soft",
                              )}
                            >
                              <span className="font-mono text-xs text-success">{c.code}</span>
                              <span className="ml-2 text-muted-foreground">{c.discipline_name}</span>
                            </button>
                          ))}
                        </div>
                      )}
                      <div className="flex items-center gap-2">
                        <Button onClick={saveSkillLink} disabled={linkSaving || !linkCompId} size="sm" variant="success">
                          {linkSaving ? "..." : "Сохранить в БД"}
                        </Button>
                        <Button variant="secondary" size="sm" onClick={() => setLinkSkill(null)}>Отмена</Button>
                        {linkMsg && <span className="text-xs text-muted-foreground">{linkMsg}</span>}
                      </div>
                    </div>
                  )}
                  <div className="flex items-center gap-3">
                    <Button onClick={applyCategorization}>Применить категории</Button>
                    {uncatMsg && <span className="text-sm text-success">{uncatMsg}</span>}
                  </div>
                </>
              )}
            </CardBody>
          </Card>
        </TabsContent>
        <TabsContent value="sched" className="space-y-4">
          <Card>
            <CardHeader
              title={
                <span className="flex items-center gap-2">
                  <Clock className="size-4" aria-hidden="true" />
                  Планировщик
                </span>
              }
            />
            <CardBody className="space-y-4">
              <div className="flex items-center gap-3 flex-wrap">
                <Button
                  onClick={() => saveSched({ collector_enabled: !(sched && sched.collector_enabled) })}
                  disabled={schedBusy}
                  variant={sched && sched.collector_enabled ? "primary" : "secondary"}
                >
                  {sched && sched.collector_enabled ? "Сбор: включён" : "Сбор: выключен"}
                </Button>
                <Button
                  onClick={() => saveSched({ daily_gap_enabled: !(sched && sched.daily_gap_enabled) })}
                  disabled={schedBusy}
                  variant={sched && sched.daily_gap_enabled ? "primary" : "secondary"}
                >
                  {sched && sched.daily_gap_enabled ? "Ночной gap: включён" : "Ночной gap: выключен"}
                </Button>
                <span className="text-xs text-muted-foreground">gap бежит цепочкой после ночного сбора</span>
              </div>
              <div className="flex items-center gap-3 flex-wrap">
                <span className="text-sm text-muted-foreground">Интервал сбора, ч:</span>
                <input
                  type="number"
                  min={1}
                  max={72}
                  value={schedInterval}
                  onChange={(e) => setSchedInterval(e.target.value)}
                  className={cn(inputClass, "h-9 w-24")}
                />
                <Button variant="secondary" size="sm" onClick={() => saveSched({ collect_interval_hours: Number(schedInterval) })} disabled={schedBusy}>Сохранить</Button>
                <Button variant="secondary" size="sm" onClick={loadSched} disabled={schedLoading}>Обновить статус</Button>
              </div>
              {sched && (
                <div className="text-sm text-muted-foreground space-y-1">
                  <div>Последний сбор: {fmtTs(sched.last_collect_ts)} · Последний gap: {sched.last_gap_date || "-"}</div>
                  <div>Следующий сбор: {fmtTs(sched.next_collect_ts)} · Занят: {sched.busy || "нет"} · Конвейер активен: {sched.pipeline_running ? "да" : "нет"}</div>
                </div>
              )}
              {schedMsg && <span className="text-sm text-muted-foreground">{schedMsg}</span>}
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Запуск вне расписания" />
            <CardBody className="space-y-3">
              <div className="flex items-center gap-3 flex-wrap">
                <Button onClick={() => runSchedJob("collect", "сбор вакансий")} disabled={schedBusy}>Собрать сейчас</Button>
                <Button variant="secondary" onClick={() => runSchedJob("gap", "gap-анализ")} disabled={schedBusy}>Gap сейчас</Button>
              </div>
              <p className="text-xs text-muted-foreground">Ручные запуски конвейера блокируются кодом 409, пока планировщик занят.</p>
            </CardBody>
          </Card>
        </TabsContent>

        {/* ── Проверка состояния (только админ; с вкладки «Данные» убрана) ── */}
        <TabsContent value="health" className="space-y-4">
          <AdminLlmCostCard />
          <Card>
            <CardHeader
              title={
                <span className="flex items-center gap-2">
                  <Activity className="size-4" aria-hidden="true" />
                  Состояние сервиса
                </span>
              }
            />
            <CardBody className="space-y-3">
              <Button variant="secondary" onClick={loadHealth} disabled={healthLoading}>
                {healthLoading ? "Проверяю..." : "Проверить состояние"}
              </Button>
              {healthMsg && <span className="text-sm text-muted-foreground">{healthMsg}</span>}
              {health && (
                <div className="space-y-3">
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    <StatCard label="Статус" value={<Badge tone={health.status === "ok" ? "success" : "danger"}>{String(health.status ?? "-")}</Badge>} />
                    <StatCard label="Версия" value={String(health.version ?? "-")} />
                    <StatCard label="API (api_ready)" value={<Badge tone={health.api_ready ? "success" : "warning"}>{health.api_ready ? "готов" : "не готов"}</Badge>} />
                    <StatCard label="Оценщик (evaluator)" value={<Badge tone={health.evaluator ? "success" : "warning"}>{health.evaluator ? "готов" : "не готов"}</Badge>} />
                    <StatCard label="Рекомендации (recommendation_engine)" value={<Badge tone={health.recommendation_engine ? "success" : "warning"}>{health.recommendation_engine ? "готов" : "не готов"}</Badge>} />
                    <StatCard label="Модель LTR (ltr_model_version)" value={String(health.ltr_model_version ?? "-")} />
                  </div>
                  <div>
                    <div className="mb-2 text-sm font-medium text-foreground">Версии кластеров (clusters_versions)</div>
                    <TableFrame>
                      <table className="w-full text-sm">
                        <thead>
                          <tr className={tableHeadClass}>
                            <th className="px-3 py-2.5 text-left">Уровень</th>
                            <th className="px-3 py-2.5 text-left">Версия</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(Object.entries(health.clusters_versions ?? {}) as [string, string | null][]).map(([lvl, ver]) => (
                            <tr key={lvl} className={tableRowClass}>
                              <td className="px-3 py-2 font-mono text-xs text-foreground">{lvl}</td>
                              <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{ver || "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </TableFrame>
                  </div>
                </div>
              )}
            </CardBody>
          </Card>
        </TabsContent>
        <TabsContent value="audit" className="space-y-4">
          <AdminAuditTab />
        </TabsContent>
        <TabsContent value="security" className="space-y-4">
          <AdminSecurity />
        </TabsContent>
      </Tabs>
    </div>
  );
}
