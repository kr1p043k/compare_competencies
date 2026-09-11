import { useState, useEffect, useRef } from "react";
import { api } from "../api";
import { authHeaders } from "../../lib/auth";
import { AnalysisPanel } from "./AnalysisPanel";
import CompetencyTrendsPanel from "./CompetencyTrendsPanel";

const API = "/api/teacher";

function plural(n: number, one: string, few: string, many: string): string {
  const m10 = n % 10;
  const m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

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

function rpdStageLabel(stage?: string): string {
  if (!stage) return "...";
  return RPD_STAGE_LABELS[stage] ?? stage;
}

type Direction = {
  dir_code: string;
  name: string;
  profile: string;
  disciplines_count: number;
};

type DirectionAnalysis = {
  direction: string;
  direction_name: string;
  profile: string;
  total_disciplines: number;
  average_coverage: number;
  average_quality_coverage?: number;
  coverage_level: string;
  total_gaps_across_all: number;
  top_cross_discipline_gaps: { skill: string; disciplines: number }[];
  top_emerging_across_all: { skill: string; frequency: number }[];
  recommendations: { type: string; priority: string; message: string }[];
  trends: { rising: any[]; declining: any[] };
  disciplines: { name: string; coverage_ratio: number; weighted_coverage?: number; coverage_level: string; gaps: number; emerging: number }[];
  generated_at: string;
};

type Discipline = {
  id: string;
  name: string;
  competencies_count: number;
  skills_count: number;
  knowledge_count?: number;
  abilities_count?: number;
  semester?: number | null;
  course?: number | null;
};

interface KsaItem {
  id: string;
  text: string;
}

type Competency = {
  id: string;
  code: string;
  name: string;
  skills: string[];
  ksa?: {
    knowledge: KsaItem[];
    abilities: KsaItem[];
    skills: KsaItem[];
  };
};

type DisciplineDetail = {
  id: string;
  name: string;
  competencies: Competency[];
};

type Recommendation = {
  id: number;
  discipline_id: string;
  competency_id: string;
  suggestion: string;
  suggestion_type: string;
};

type Stats = {
  total_disciplines: number;
  total_competencies: number;
  total_skills: number;
};

export function TeacherDashboard() {
  const [disciplines, setDisciplines] = useState<Discipline[]>([]);
  const [selected, setSelected] = useState<DisciplineDetail | null>(null);

  const [stats, setStats] = useState<Stats | null>(null);
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [suggestion, setSuggestion] = useState("");
  const [recType, setRecType] = useState("modify");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [directions, setDirections] = useState<Direction[]>([]);
  const [selectedDir, setSelectedDir] = useState("09.03.02");
  const [analysis, setAnalysis] = useState<DirectionAnalysis | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<"coverage" | "trends">("coverage");
  const [showAddForm, setShowAddForm] = useState(false);
  const [seedMsg, setSeedMsg] = useState("");
  const [seedLoading, setSeedLoading] = useState(false);
  const [zunForm, setZunForm] = useState<{ compId: string; ksaType: string; text: string } | null>(null);
  const [zunMsg, setZunMsg] = useState("");
  const [zunSaving, setZunSaving] = useState(false);
  const [ksaEditing, setKsaEditing] = useState<{ id: string; text: string } | null>(null);
  const [selectedCompetency, setSelectedCompetency] = useState("");
  const [runLoading, setRunLoading] = useState(false);
  const [runMsg, setRunMsg] = useState("");

  const [rpdSources, setRpdSources] = useState<{ yandex_covered: string[] }>({ yandex_covered: [] });
  const [rpdFile, setRpdFile] = useState<File | null>(null);
  const rpdInputRef = useRef<HTMLInputElement | null>(null);
  const [rpdUploading, setRpdUploading] = useState(false);
  const [rpdCollecting, setRpdCollecting] = useState(false);
  const [rpdYandexUrl, setRpdYandexUrl] = useState("");
  const [rpdRun, setRpdRun] = useState<{ run_id: string } | null>(null);
  const [rpdStatus, setRpdStatus] = useState<any>(null);
  const [rpdMsg, setRpdMsg] = useState("");
  const [collectCooldown, setCollectCooldown] = useState(0);
  const runAnalysisBtnRef = useRef<HTMLButtonElement | null>(null);

  // Кнопка "Запустить анализ" из пустого состояния деталки дисциплины.
  useEffect(() => {
    const handler = () => { runAnalysisBtnRef.current?.click(); };
    window.addEventListener("run-direction-analysis", handler);
    return () => window.removeEventListener("run-direction-analysis", handler);
  }, []);

  useEffect(() => {
    Promise.all([
      api("/teacher/krm/recommendations"),
      api("/teacher/krm/directions"),
    ]).then(([r, dirs]) => {
      setRecs(r as Recommendation[]);
      const list = (Array.isArray(dirs) ? dirs : [dirs]) as Direction[];
      setDirections(list);
      if (list.length > 0 && !list.some((d) => d.dir_code === selectedDir)) {
        setSelectedDir(list[0].dir_code);
      }
      setLoading(false);
    }).catch((e) => {
      console.error("TeacherDashboard init failed", e);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selectedDir) return;
    api(`/teacher/krm/disciplines?dir_code=${selectedDir}`)
      .then((d) => setDisciplines(d as Discipline[]))
      .catch(() => setDisciplines([]));
    api(`/teacher/krm/stats?dir_code=${selectedDir}`)
      .then(setStats)
      .catch(() => {});
    api(`/teacher/analysis?dir_code=${selectedDir}`)
      .then(setAnalysis)
      .catch(() => setAnalysis(null));
  }, [selectedDir]);

  useEffect(() => {
    if (collectCooldown <= 0) return;
    const t = setTimeout(() => setCollectCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => clearTimeout(t);
  }, [collectCooldown]);

  useEffect(() => {
    api("/teacher/rpd/sources")
      .then(setRpdSources)
      .catch(() => {});
  }, []);

  // Возобновление поллинга после переключения вкладок (табы размонтируют компонент).
  useEffect(() => {
    let stored: string | null = null;
    try { stored = sessionStorage.getItem("rpdRunId"); } catch {}
    if (stored) {
      setRpdRun({ run_id: stored });
      setRpdCollecting(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!rpdRun?.run_id) return;
    let cancelled = false;
    let misses = 0;
    const poll = () => {
      api(`/teacher/rpd/status/${rpdRun.run_id}`)
        .then((s) => {
          if (cancelled) return;
          misses = 0;
          setRpdStatus(s);
          if (s?.status === "completed" || s?.status === "failed" || s?.status === "cancelled") {
            setRpdUploading(false);
            setRpdCollecting(false);
            try { sessionStorage.removeItem("rpdRunId"); } catch {}
            setRpdMsg(s?.status === "completed" ? "Готово" : s?.status === "cancelled" ? "Отменено пользователем" : `Ошибка: ${s?.error || "unknown"}`);
            if (s?.status === "completed") {
              api(`/teacher/analysis?dir_code=${selectedDir}`)
                .then(setAnalysis)
                .catch(() => {});
            }
          } else {
            setTimeout(poll, 3000);
          }
        })
        .catch(() => {
          if (cancelled) return;
          // Разовый сбой сети/рестарт сервера — ждём, а не висим молча.
          misses += 1;
          if (misses < 10) {
            setRpdMsg(`Сервер перезапускается, жду статус... (${misses})`);
            setTimeout(poll, 5000);
          } else {
            setRpdUploading(false);
            setRpdCollecting(false);
            setRpdMsg("Ошибка: статус задачи недоступен. Проверьте сервер и повторите.");
          }
        });
    };
    poll();
    return () => { cancelled = true; };
  }, [rpdRun?.run_id]);

  async function uploadRpd(file?: File | null) {
    const f = file ?? rpdFile;
    if (!f || rpdUploading) return;
    setRpdUploading(true); setRpdMsg("Загрузка...");
    const fd = new FormData();
    fd.append("file", f);
    fd.append("dir_code", selectedDir);
    try {
      const res = await fetch(`/api/teacher/rpd/upload`, {
        method: "POST",
        headers: { ...authHeaders() },
        body: fd,
      });
      if (res.status === 429) throw new Error("Слишком частые запросы — подождите минуту и повторите");
      const data = await res.json().catch(() => ({} as any));
      if (!res.ok) throw new Error(data.detail || res.statusText);
      if (!data.run_id) throw new Error("Сервер не вернул ID задачи");
      try { sessionStorage.setItem("rpdRunId", data.run_id); } catch {}
      setRpdRun({ run_id: data.run_id });
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
      if (!rpdRun?.run_id) {
        setRpdCollecting(false);
        setRpdMsg("");
        return;
      }
      try {
        setRpdMsg("Останавливаю сбор...");
        await fetch(`/api/teacher/rpd/cancel/${rpdRun.run_id}`, {
          method: "POST",
          headers: { ...authHeaders() },
        });
      } catch {
        setRpdCollecting(false);
        setRpdMsg("");
      }
      return;
    }
    setRpdCollecting(true); setRpdMsg("Сбор аннотаций с Yandex Disk...");
    try {
      const res = await fetch(`/api/teacher/rpd/collect`, {
        method: "POST",
        headers: {
          ...authHeaders(),
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `dir_code=${encodeURIComponent(selectedDir)}${rpdYandexUrl.trim() ? `&public_url=${encodeURIComponent(rpdYandexUrl.trim())}` : ""}`,
      });
      if (res.status === 429) {
        setCollectCooldown(30);
        throw new Error("Слишком частые запросы — кнопка заблокирована на 30 секунд");
      }
      const data = await res.json().catch(() => ({} as any));
      if (!res.ok) throw new Error(data.detail || res.statusText);
      if (!data.run_id) throw new Error("Сервер не вернул ID задачи");
      try { sessionStorage.setItem("rpdRunId", data.run_id); } catch {}
      setRpdRun({ run_id: data.run_id });
      setRpdMsg("Сбор и обработка запущены...");
    } catch (e: any) {
      setRpdMsg("Ошибка: " + e.message);
      setRpdCollecting(false);
    }
  }

  async function addZunEntry() {
    if (!selected || !zunForm || !zunForm.text.trim() || zunSaving) return;
    setZunSaving(true);
    setZunMsg("");
    try {
      await api(`/teacher/zun/competencies/${zunForm.compId}/entries`, {
        method: "POST",
        body: JSON.stringify({ ksa_type: zunForm.ksaType, text: zunForm.text.trim() }),
      });
      setZunForm(null);
      setZunMsg("ЗУН добавлен. Учтётся при следующем пересчёте анализа.");
      await loadDiscipline(selected.name);
    } catch (e: any) {
      setZunMsg("Ошибка: " + (e.message || "не удалось сохранить"));
    } finally {
      setZunSaving(false);
    }
  }

  async function saveKsaEdit() {
    if (!selected || !ksaEditing || !ksaEditing.text.trim() || zunSaving) return;
    setZunSaving(true);
    setZunMsg("");
    try {
      await api(`/teacher/zun/entries/${ksaEditing.id}`, {
        method: "PATCH",
        body: JSON.stringify({ text: ksaEditing.text.trim() }),
      });
      setKsaEditing(null);
      setZunMsg("Пункт обновлён. Учтётся при следующем пересчёте анализа.");
      await loadDiscipline(selected.name);
    } catch (e: any) {
      setZunMsg("Ошибка: " + (e.message || "не удалось сохранить"));
    } finally {
      setZunSaving(false);
    }
  }

  async function delKsa(id: string) {
    if (!selected || zunSaving) return;
    if (!window.confirm("Удалить пункт?")) return;
    setZunSaving(true);
    setZunMsg("");
    try {
      await api(`/teacher/zun/entries/${id}`, { method: "DELETE" });
      setZunMsg("Пункт удалён. Учтётся при следующем пересчёте анализа.");
      await loadDiscipline(selected.name);
    } catch (e: any) {
      setZunMsg("Ошибка: " + (e.message || "не удалось удалить"));
    } finally {
      setZunSaving(false);
    }
  }

  async function seedAutoRecs() {
    setSeedLoading(true);
    setSeedMsg("");
    try {
      const r = await api(`/teacher/krm/recommendations/seed/auto?dir_code=${selectedDir}`, { method: "POST" });
      const data = await api("/teacher/krm/recommendations");
      setRecs(data);
      setSeedMsg(`Добавлено ${r.seeded} авторекомендаций (ручные не тронуты).`);
    } catch (e: any) {
      setSeedMsg("Ошибка: " + (e.message || "не удалось заполнить"));
    } finally {
      setSeedLoading(false);
    }
  }

  async function loadDiscipline(name: string) {
    try {
      const data = await api(`/teacher/krm/disciplines/${encodeURIComponent(name)}?dir_code=${selectedDir}`);
      setSelected(data as DisciplineDetail);
      setShowAnalysis(false);
    } catch {}
  }

  async function addRec() {
    if (!selected || !suggestion.trim()) return;
    try {
      const compId = recType === "add" ? "" : selectedCompetency;
      const resp = await api("/teacher/krm/recommendations", {
        method: "POST",
        body: JSON.stringify({
          discipline_id: selected.name,
          competency_id: compId || null,
          suggestion: suggestion.trim(),
          suggestion_type: recType,
        }),
      });
      setRecs((prev) => [
        ...prev,
        {
          id: resp.id,
          discipline_id: selected.name,
          competency_id: compId || "",
          suggestion: suggestion.trim(),
          suggestion_type: recType,
        },
      ]);
      setSuggestion("");
    } catch {}
  }

  async function deleteRec(id: number) {
    try {
      await api(`/teacher/krm/recommendations/${id}`, { method: "DELETE" });
      setRecs((prev) => prev.filter((r) => r.id !== id));
    } catch {}
  }

  if (loading) {
    return (
      <div style={{ padding: "40px", fontFamily: "system-ui, sans-serif" }}>
        Loading...
      </div>
    );
  }

  const filtered = disciplines.filter((d) =>
    d.name.toLowerCase().includes(search.toLowerCase())
  );

  const containerStyle: React.CSSProperties = {
    display: "flex",
    height: "calc(100vh - 100px)",
    fontFamily: "system-ui, -apple-system, sans-serif",
    color: "#111827",
    background: "#fff",
  };

  const sidebarStyle: React.CSSProperties = {
    width: 380,
    minWidth: 380,
    borderRight: "1px solid #e5e7eb",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  };

  const mainStyle: React.CSSProperties = {
    flex: 1,
    overflow: "auto",
    padding: "24px",
  };

  const card: React.CSSProperties = {
    background: "#fff",
    borderRadius: 8,
    border: "1px solid #e5e7eb",
    padding: 14,
    marginBottom: 10,
    boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
  };

  function covColor(cov: number) {
    return cov > 0.5 ? "#059669" : cov > 0.2 ? "#d97706" : "#dc2626";
  }

  return (
    <div style={containerStyle}>
      <div style={sidebarStyle}>
          <div style={{ padding: "16px", borderBottom: "1px solid #e5e7eb" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h1 style={{ fontSize: 18, margin: 0, color: "#111827" }}>
                KRM Teacher
              </h1>
              {stats && (
                  <div style={{ fontSize: 11, color: "#6b7280", textAlign: "right" }}>
                    <div>{(stats as any).total_disciplines ?? (stats as any).total_reports ?? 0} disc</div>
                    <div>{(stats as any).total_competencies ?? Object.keys((stats as any).by_profession || {}).length} comp</div>
                    <div>{((stats as any).total_skills ?? 0).toLocaleString()} skills</div>
                  </div>
                )}
          </div>

          {/* Direction selector */}
          <select
            value={selectedDir}
            onChange={(e) => setSelectedDir(e.target.value)}
            style={{
              width: "100%",
              marginTop: 10,
              padding: "6px 8px",
              background: "#fff",
              border: "1px solid #e5e7eb",
              borderRadius: 6,
              color: "#111827",
              fontSize: 12,
              outline: "none",
            }}
          >
            {directions.map((d, idx) => (
              <option key={`${d.dir_code}-${idx}`} value={d.dir_code}>{d.dir_code} - {d.name}</option>
            ))}
          </select>

          {/* Run analysis button */}
          <button
            ref={runAnalysisBtnRef}
            onClick={async () => {
              if (runLoading) return;
              setRunLoading(true);
              setRunMsg("Запуск анализа...");
              try {
                const started: any = await api(`/teacher/krm/run-analysis?dir_code=${selectedDir}`, { method: "POST" });
                const runId = started?.run_id;
                const t0 = Date.now();
                if (!runId) throw new Error("Сервер не вернул ID задачи");
                for (let i = 0; i < 360; i++) {
                  await new Promise((res) => setTimeout(res, 5000));
                  const spent = Math.round((Date.now() - t0) / 1000);
                  let s: any = null;
                  try {
                    s = await api(`/teacher/rpd/status/${runId}`);
                  } catch {
                    setRunMsg(`Этап: пересчёт анализа… нет связи (${spent} c)`);
                    continue;
                  }
                  if (!s) continue;
                  if (s.status === "completed") {
                    const a: any = await api(`/teacher/analysis?dir_code=${selectedDir}`).catch(() => null);
                    if (a) setAnalysis(a);
                    setRunMsg("Готово");
                    break;
                  }
                  if (s.status === "failed" || s.status === "cancelled") {
                    setRunMsg(s.status === "cancelled" ? "Отменено пользователем" : `Ошибка анализа: ${s.error || "unknown"}`);
                    break;
                  }
                  setRunMsg(`Этап: пересчёт анализа (60 дисциплин)… ${spent} c`);
                  if (i === 359) setRunMsg("Превышено ожидание — проверьте результат позже");
                }
              } catch (e: any) {
                setRunMsg(`Ошибка запуска: ${e?.message || e}`);
              } finally {
                setRunLoading(false);
              }
            }}
            disabled={runLoading}
            style={{
              width: "100%",
              marginTop: 10,
              padding: "8px 12px",
              background: runLoading ? "#9ca3af" : "#7c3aed",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: runLoading ? "default" : "pointer",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            {runLoading ? "Анализ запущен..." : "Запустить анализ"}
          </button>
            {runMsg && (
              <div style={{ fontSize: 12, color: "#b45309", marginTop: 6 }}>{runMsg}</div>
            )}

          {/* RPD upload block */}
          <div
            style={{
              marginTop: 12,
              padding: "10px 12px",
              background: "#fffbeb",
              border: "1px solid #fde68a",
              borderRadius: 6,
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 700, color: "#92400e", marginBottom: 8 }}>
              Подгрузка РПД
            </div>
            <input
              ref={rpdInputRef}
              type="file"
              accept=".pdf"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0] || null;
                setRpdFile(f);
                e.target.value = "";
                if (f) uploadRpd(f);
              }}
            />
            <button
              onClick={() => rpdInputRef.current?.click()}
              disabled={rpdUploading}
              style={{
                width: "100%",
                padding: "8px 12px",
                background: rpdUploading ? "#9ca3af" : "#7c3aed",
                color: "#fff",
                border: "none",
                borderRadius: 6,
                cursor: rpdUploading ? "default" : "pointer",
                fontSize: 12,
                fontWeight: 600,
              }}
            >
              {rpdUploading ? "Обработка..." : "Загрузить PDF и обработать"}
            </button>
            {rpdSources.yandex_covered?.includes(selectedDir) && (
              <button
                onClick={collectRpd}
                disabled={collectCooldown > 0}
                style={{
                  width: "100%",
                  marginTop: 8,
                  padding: "8px 12px",
                  background: rpdCollecting ? "#dc2626" : collectCooldown > 0 ? "#9ca3af" : "#7c3aed",
                  color: "#fff",
                  border: "none",
                  borderRadius: 6,
                  cursor: collectCooldown > 0 ? "default" : "pointer",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                {rpdCollecting ? "Отмена" : collectCooldown > 0 ? `Подождите ${collectCooldown} сек` : "Собрать с Yandex Disk"}
              </button>
            )}
            <div style={{ display: "flex", flexDirection: "column", gap: 6, borderTop: "1px dashed #fde68a", paddingTop: 8, marginTop: 8 }}>
              <input
                type="text"
                value={rpdYandexUrl}
                onChange={(e) => setRpdYandexUrl(e.target.value)}
                placeholder="Ссылка на папку Yandex Disk (https://disk.360.yandex.ru/d/...)"
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  border: "1px solid #fde68a",
                  borderRadius: 6,
                  fontSize: 12,
                  boxSizing: "border-box",
                }}
              />
              <button
                onClick={collectRpd}
                disabled={rpdCollecting || collectCooldown > 0 || !rpdYandexUrl.trim()}
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  background: rpdCollecting || collectCooldown > 0 || !rpdYandexUrl.trim() ? "#9ca3af" : "#0d9488",
                  color: "#fff",
                  border: "none",
                  borderRadius: 6,
                  cursor: rpdCollecting || collectCooldown > 0 || !rpdYandexUrl.trim() ? "default" : "pointer",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                {rpdCollecting ? "Сбор..." : "Загрузить по ссылке"}
              </button>
            </div>
            {rpdStatus && (rpdStatus.status === "completed" || rpdStatus.status === "failed") && (
              <div
                style={{
                  marginTop: 8,
                  fontSize: 11,
                  color: rpdStatus.status === "completed" ? "#15803d" : "#b91c1c",
                }}
              >
                {rpdMsg}
                {rpdStatus.stats?.disciplines != null && rpdStatus.status === "completed" && (
                  <span> — {rpdStatus.stats.disciplines} дисциплин</span>
                )}
              </div>
            )}
            {(rpdUploading || rpdCollecting) && rpdStatus && (rpdStatus.status === "running" || rpdStatus.status === "started") && (
              <div style={{ marginTop: 10 }}>
                <div
                  style={{
                    height: 8,
                    background: "#fde68a",
                    borderRadius: 4,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${rpdStatus.stats?.progress ?? 10}%`,
                      background: "#0d9488",
                      transition: "width 0.5s ease",
                    }}
                  />
                </div>
                <div style={{ marginTop: 6, fontSize: 11, color: "#92400e" }}>
                  Этап: {rpdStageLabel(rpdStatus.stats?.stage)} · {rpdStatus.stats?.progress ?? 10}%
                </div>
              </div>
            )}
          </div>

          {/* Analysis summary button */}
          {analysis && (
            <div
              onClick={() => setShowAnalysis(!showAnalysis)}
              style={{
                marginTop: 10,
                padding: "8px 12px",
                background: "#f9fafb",
                border: `1px solid ${covColor(analysis.average_coverage)}`,
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 12,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "#7c3aed", fontWeight: 600 }}>Анализ</span>
                <span style={{ color: covColor(analysis.average_coverage), fontWeight: 700 }}>
                  {(analysis.average_coverage * 100).toFixed(1)}%
                  {analysis.average_quality_coverage != null && (
                    <span style={{ display: "block", fontSize: 12, color: covColor(analysis.average_quality_coverage), marginTop: 2 }}>
                      Качество: {(analysis.average_quality_coverage * 100).toFixed(1)}%
                    </span>
                  )}
                </span>
              </div>
              <div style={{ color: "#6b7280", marginTop: 2 }}>
                {analysis.total_disciplines} {plural(analysis.total_disciplines, "дисциплина", "дисциплины", "дисциплин")}, {analysis.total_gaps_across_all} {plural(analysis.total_gaps_across_all, "пробел", "пробела", "пробелов")}
              </div>
            </div>
          )}

          <input
            placeholder="Поиск дисциплин..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              width: "100%",
              marginTop: 10,
              padding: "8px 12px",
              background: "#fff",
              border: "1px solid #e5e7eb",
              borderRadius: 6,
              color: "#111827",
              fontSize: 13,
              outline: "none",
              boxSizing: "border-box",
            }}
          />
        </div>
        <div style={{ flex: 1, overflow: "auto" }}>
          {filtered.map((d) => {
            const discAnalysis = analysis?.disciplines.find((a) => a.name === d.name);
            return (
              <div
                key={d.name}
                onClick={() => loadDiscipline(d.name)}
                style={{
                  padding: "10px 16px",
                  cursor: "pointer",
                  borderBottom: "1px solid #e5e7eb",
                  background:
                    selected?.name === d.name ? "#eef2ff" : "transparent",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#7c3aed" }}>
                    {d.name}
                  </div>
                  {discAnalysis && (
                    <span style={{
                      fontSize: 11,
                      fontWeight: 700,
                      color: covColor(discAnalysis.coverage_ratio),
                    }}>
                      {(discAnalysis.coverage_ratio * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 11, color: "#6b7280", marginTop: 2 }}>
                  {d.course != null && `${d.course} курс · `}
                  {d.competencies_count} {plural(d.competencies_count, "комп.", "комп.", "комп.")} · {d.skills_count} {plural(d.skills_count, "навык", "навыка", "навыков")}
                  {d.abilities_count != null && ` · ${d.abilities_count} ${plural(d.abilities_count, "умение", "умения", "умений")}`}
                  {d.knowledge_count != null && ` · ${d.knowledge_count} ${plural(d.knowledge_count, "знание", "знания", "знаний")}`}
                  {discAnalysis && ` / ${discAnalysis.gaps} ${plural(discAnalysis.gaps, "пробел", "пробела", "пробелов")}`}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div style={mainStyle}>
        {!selected && !showAnalysis && (
          <div style={{ textAlign: "center", marginTop: 80, color: "#9ca3af", fontSize: 14 }}>
            Выберите дисциплину или откройте анализ
          </div>
        )}

        {/* Always visible mode toggle */}
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          <button
            onClick={() => setAnalysisMode("coverage")}
            style={{
              padding: "6px 14px",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 12,
              fontWeight: analysisMode === "coverage" ? 700 : 400,
              background: analysisMode === "coverage" ? "#7c3aed" : "#f9fafb",
              color: analysisMode === "coverage" ? "#fff" : "#7c3aed",
            }}
          >
            Coverage
          </button>
          <button
            onClick={() => setAnalysisMode("trends")}
            style={{
              padding: "6px 14px",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 12,
              fontWeight: analysisMode === "trends" ? 700 : 400,
              background: analysisMode === "trends" ? "#7c3aed" : "#f9fafb",
              color: analysisMode === "trends" ? "#fff" : "#7c3aed",
            }}
          >
            Competency Trends
          </button>
        </div>

        {analysisMode === "coverage" && analysis && (<>
          <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8 }}>
            {selected
              ? `Дисциплина: ${selected.name} (направление ${selectedDir})`
              : `Направление ${selectedDir} — сводка по всем ${analysis.total_disciplines} дисциплинам, ни одна дисциплина не выбрана`}
          </div>
          {/* When a discipline is selected, show its coverage instead of direction average */}
          {(() => {
            const discData = selected && analysis.disciplines
              ? analysis.disciplines.find((d: any) => d.name === selected.name)
              : null;
            const cov = discData ? discData.coverage_ratio : analysis.average_coverage;
            const gaps = discData ? discData.gaps : analysis.total_gaps_across_all;
            const level = discData ? discData.coverage_level : analysis.coverage_level;
          return (<>
          <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
            <div style={card}>
              <div style={{ fontSize: 11, color: "#6b7280" }}>
                {discData ? `Coverage: ${selected.name}` : "Average Coverage"}
              </div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: covColor(cov) }}>
                    {(cov * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: 11, color: covColor(cov), fontWeight: 600 }}>
                    {(level || "").toUpperCase()}
                  </div>
                </div>
                <div style={card}>
                  <div style={{ fontSize: 11, color: "#6b7280" }}>Gaps</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: "#fca5a5" }}>
                    {gaps}
                  </div>
                </div>
                <div style={card}>
                  <div style={{ fontSize: 11, color: "#6b7280" }}>Disciplines</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: "#93c5fd" }}>
                    {analysis.total_disciplines}
                  </div>
                </div>
              </div>
          </>)})()}

              {/* Общие блоки направления — только пока дисциплина не выбрана.
                  При выбранной дисциплине её детали ниже (AnalysisPanel). */}
              {!selected && (<>
              {/* Direction-level recommendations */}
              {(analysis.recommendations || []).length > 0 && (
                <div style={card}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", marginBottom: 8 }}>
                    Рекомендации
                  </div>
                  {(analysis.recommendations || []).map((r, i) => (
                    <div key={i} style={{ padding: "8px 10px", marginBottom: 6, background: "#f9fafb", borderRadius: 6, borderLeft: `3px solid ${r.priority === "high" ? "#dc2626" : r.priority === "medium" ? "#d97706" : "#2563eb"}`, fontSize: 12 }}>
                      <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 4 }}>
                        <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: 4, fontSize: 11, background: r.priority === "high" ? "#fee2e2" : r.priority === "medium" ? "#fef3c7" : "#dbeafe", color: r.priority === "high" ? "#dc2626" : r.priority === "medium" ? "#92400e" : "#1d4ed8", fontWeight: 600 }}>{r.priority === "high" ? "высокий" : r.priority === "medium" ? "средний" : "низкий"}</span>
                        <span style={{ fontSize: 11, color: "#6b7280" }}>{r.type}</span>
                      </div>
                      <div style={{ color: "#374151", lineHeight: 1.4 }}>{r.message}</div>
                    </div>
                  ))}
                </div>
              )}

                 <div style={card}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", marginBottom: 8 }}>Междисциплинарные разрывы</div>
                {((analysis.top_cross_discipline_gaps || []) as any[]).map((g: any, i: number) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #e5e7eb", fontSize: 12 }}>
                    <span style={{ color: "#b91c1c", fontWeight: 500 }}>{g.skill}</span>
                    <span style={{ color: "#6b7280" }}>{g.disciplines} дисциплин</span>
                  </div>
                ))}
              </div>

              <div style={card}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", marginBottom: 8 }}>Востребованные навыки рынка</div>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                  {((analysis.top_emerging_across_all || []) as any[]).map((s: any, i: number) => (
                    <span key={i} style={{ display: "inline-block", padding: "2px 8px", borderRadius: 4, fontSize: 11, background: "#e0e7ff", color: "#4338ca", margin: 2 }}>
                      {s.skill} <span style={{ opacity: 0.5 }}>×{s.frequency}</span>
                    </span>
                  ))}
                </div>
              </div>

              {analysis.trends && (
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  {analysis.trends.rising?.length > 0 && (
                    <div style={{ ...card, flex: 1, minWidth: 200 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "#059669", marginBottom: 8 }}>Растущие навыки</div>
                      {analysis.trends.rising.map((t, i) => (
                        <div key={i} style={{ fontSize: 11, padding: "2px 0", color: "#4b5563" }}>
                          {t.skill} <span style={{ color: "#059669" }}>+{t.change_pct}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {analysis.trends.declining?.length > 0 && (
                    <div style={{ ...card, flex: 1, minWidth: 200 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "#dc2626", marginBottom: 8 }}>Падающие навыки</div>
                      {analysis.trends.declining.map((t, i) => (
                        <div key={i} style={{ fontSize: 11, padding: "2px 0", color: "#4b5563" }}>
                          {t.skill} <span style={{ color: "#dc2626" }}>{t.change_pct}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div style={card}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#7c3aed", marginBottom: 8 }}>Разбивка по дисциплинам</div>
                {analysis.disciplines.map((d, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid #e5e7eb", fontSize: 12, cursor: "pointer" }}
                    onClick={() => { const found = disciplines.find((dd) => dd.name === d.name); if (found) loadDiscipline(found.name); }}
                  >
                    <span style={{ color: "#374151", maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                    <div style={{ display: "flex", gap: 12 }}>
                      <span style={{ color: covColor(d.coverage_ratio), fontWeight: 600 }}>{(d.coverage_ratio * 100).toFixed(1)}%</span>
                      {d.weighted_coverage != null && (
                        <span style={{ color: covColor(d.weighted_coverage), fontSize: 11 }} title="Quality-weighted coverage">Q:{(d.weighted_coverage * 100).toFixed(0)}%</span>
                      )}
                      <span style={{ color: "#dc2626" }}>{d.gaps}g</span>
                      <span style={{ color: "#2563eb" }}>{d.emerging}e</span>
                    </div>
                  </div>
                ))}
              </div>
              </>)}
            </>)}

          {analysisMode === "trends" && (
            <CompetencyTrendsPanel
              dirCode={selectedDir}
              competencyCodes={selected ? selected.competencies.map((c) => c.code) : undefined}
            />
          )}

        {/* Discipline detail */}
        {selected && (
          <>
            <h2 style={{ fontSize: 20, margin: "0 0 20px", color: "#111827" }}>
              {selected.name}
            </h2>

            {/* Analysis panel for this discipline */}
            <AnalysisPanel disciplineName={selected.name} dirCode={selectedDir} />

            {selected.competencies.map((comp) => {
              const ksa = comp.ksa;
              const hasGroups = !!ksa && (
                (ksa.knowledge?.length || 0) + (ksa.abilities?.length || 0) + (ksa.skills?.length || 0)
              ) > 0;
              const total = hasGroups
                ? (ksa!.knowledge.length + ksa!.abilities.length + ksa!.skills.length)
                : comp.skills.length;
              const groups = hasGroups ? [
                { title: "Знания", items: ksa!.knowledge },
                { title: "Умения", items: ksa!.abilities },
                { title: "Навыки", items: ksa!.skills },
              ] : [];
              return (
              <div
                key={comp.code}
                className="mb-2 border border-gray-200 rounded-lg overflow-hidden"
              >
                <div className="px-4 py-2.5 bg-gray-50 flex items-center gap-2">
                  <span className="font-semibold text-sm text-purple-600">
                    {comp.code}
                  </span>
                  <button
                    onClick={() => { setZunForm({ compId: comp.id, ksaType: "skills", text: "" }); setZunMsg(""); }}
                    className="ml-auto text-xs text-purple-600 hover:text-purple-800 border-0 bg-transparent cursor-pointer"
                    title="Добавить пункт (знание / умение / навык) в эту компетенцию"
                  >
                    + ЗУН
                  </button>
                  <span className="text-xs text-gray-400">
                    {total} {plural(total, "навык", "навыка", "навыков")}
                  </span>
                </div>
                <div className="px-4 py-2">
                {zunForm && zunForm.compId === comp.id && (
                  <div className="mb-2 rounded-lg border border-purple-200 bg-purple-50 p-2">
                    <div className="flex gap-2 mb-2">
                      <select
                        value={zunForm.ksaType}
                        onChange={(e) => setZunForm({ ...zunForm, ksaType: e.target.value })}
                        className="text-xs border border-gray-300 rounded-md px-2 py-1 bg-white"
                      >
                        <option value="knowledge">Знания</option>
                        <option value="abilities">Умения</option>
                        <option value="skills">Навыки</option>
                      </select>
                      <button
                        onClick={() => setZunForm(null)}
                        className="text-xs text-gray-500 hover:text-gray-700 border-0 bg-transparent cursor-pointer"
                      >
                        Отмена
                      </button>
                    </div>
                    <textarea
                      value={zunForm.text}
                      onChange={(e) => setZunForm({ ...zunForm, text: e.target.value })}
                      rows={3}
                      maxLength={2000}
                      placeholder="Текст пункта: знание, умение или навык…"
                      className="w-full text-xs border border-gray-300 rounded-md px-2 py-1 mb-2"
                    />
                    <div className="flex items-center gap-2">
                      <button
                        onClick={addZunEntry}
                        disabled={zunSaving || !zunForm.text.trim()}
                        className="text-xs bg-purple-600 text-white px-3 py-1.5 rounded-lg hover:bg-purple-700 transition-colors cursor-pointer border-0 disabled:opacity-50"
                      >
                        {zunSaving ? "Сохранение…" : "Добавить"}
                      </button>
                      {zunMsg && <span className="text-xs text-gray-600">{zunMsg}</span>}
                    </div>
                    <div className="text-[11px] text-gray-400 mt-1">Пункт попадёт в анализ при следующем пересчёте.</div>
                  </div>
                )}
                  {total === 0 ? (
                    <div className="text-xs text-gray-400">Навыки не извлечены</div>
                  ) : hasGroups ? (
                    groups.map((g) => g.items.length > 0 && (
                      <div key={g.title} className="mb-2 last:mb-0">
                        <div className="text-[11px] font-semibold text-purple-500 uppercase tracking-wide mt-1.5 mb-1">
                          {g.title} ({g.items.length})
                        </div>
                        {g.items.map((sk) => (
                          <div key={sk.id} className="py-0.5 text-xs leading-relaxed border-b border-gray-100 last:border-0">
                            {ksaEditing && ksaEditing.id === sk.id ? (
                              <div className="flex gap-2 items-start">
                                <textarea
                                  value={ksaEditing.text}
                                  onChange={(e) => setKsaEditing({ ...ksaEditing, text: e.target.value })}
                                  rows={2}
                                  maxLength={2000}
                                  className="flex-1 text-xs border border-gray-300 rounded-md px-2 py-1"
                                />
                                <button onClick={saveKsaEdit} disabled={zunSaving} className="text-xs bg-purple-600 text-white px-2 py-1 rounded-md hover:bg-purple-700 cursor-pointer border-0 disabled:opacity-50">OK</button>
                                <button onClick={() => setKsaEditing(null)} className="text-xs text-gray-500 hover:text-gray-700 border-0 bg-transparent cursor-pointer">Отмена</button>
                              </div>
                            ) : (
                              <div className="flex gap-1 items-start group">
                                <span className="flex-1">{sk.text}</span>
                                <button onClick={() => { setKsaEditing({ id: sk.id, text: sk.text }); setZunMsg(""); }} title="Редактировать" className="text-gray-300 hover:text-purple-600 border-0 bg-transparent cursor-pointer text-xs">Изменить</button>
                                <button onClick={() => delKsa(sk.id)} title="Удалить" className="text-gray-300 hover:text-red-600 border-0 bg-transparent cursor-pointer text-xs">Удалить</button>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ))
                  ) : (
                    comp.skills.map((sk, i) => (
                      <div key={i} className="py-0.5 text-xs leading-relaxed border-b border-gray-100 last:border-0">
                        {sk}
                      </div>
                    ))
                  )}
                </div>
              </div>
              );
            })}

            <div className="mt-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="flex items-center justify-center w-8 h-8 bg-purple-600 rounded-lg">
                  <span className="text-white text-sm font-bold">!</span>
                </div>
                <h3 className="text-sm font-semibold text-gray-900">Recommendations</h3>
                <span className="text-xs text-gray-400">({recs.filter((r) => r.discipline_id === selected?.name).length})</span>
                {seedMsg && <span className="text-xs text-gray-500">{seedMsg}</span>}
                <button
                  onClick={() => setShowAddForm(!showAddForm)}
                  className="ml-auto text-xs bg-purple-600 text-white px-3 py-1.5 rounded-lg hover:bg-purple-700 transition-colors cursor-pointer border-0"
                >
                  {showAddForm ? "Cancel" : "Add Recommendation"}
                </button>
                <button
                  onClick={seedAutoRecs}
                  disabled={seedLoading}
                  title="Заполнить панель топ-рекомендациями из автоанализа (ручные сохранятся)"
                  className="text-xs bg-white text-purple-700 border border-purple-300 px-3 py-1.5 rounded-lg hover:bg-purple-50 transition-colors cursor-pointer disabled:opacity-50"
                >
                  {seedLoading ? "Заполнение…" : "Заполнить из анализа"}
                </button>
              </div>
              {showAddForm && (
                <div className="mb-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                  {(recType === "modify" || recType === "remove") && (
                    <select
                      value={selectedCompetency}
                      onChange={(e) => setSelectedCompetency(e.target.value)}
                      className="w-full h-9 px-2 mb-2 text-sm bg-white border border-gray-300 rounded-lg outline-none text-gray-900"
                    >
                      <option value="">-- Select competency --</option>
                      {selected?.competencies.map((c) => (
                        <option key={c.code} value={c.code}>{c.code}</option>
                      ))}
                    </select>
                  )}
                  <textarea
                    placeholder={recType === "add" ? "Your recommendation for this discipline..." : "Your recommendation for this competency..."}
                    value={suggestion}
                    onChange={(e) => setSuggestion(e.target.value)}
                    rows={2}
                    className="w-full p-2 text-sm border border-gray-300 rounded-lg resize-vertical outline-none box-border"
                    style={{ background: "#fff", color: "#1f2937" }}
                  />
                  <div className="flex gap-2 mt-2 items-center">
                    <select
                      value={recType}
                      onChange={(e) => { setRecType(e.target.value); setSelectedCompetency(""); }}
                      className="h-9 px-2 text-sm bg-white border border-gray-300 rounded-lg outline-none text-gray-900"
                    >
                      <option value="modify">Modify</option>
                      <option value="add">Add</option>
                      <option value="remove">Remove</option>
                    </select>
                    <button
                      onClick={() => { addRec(); setShowAddForm(false); setSelectedCompetency(""); }}
                      className="h-9 px-4 text-sm bg-purple-600 text-white border-0 rounded-lg cursor-pointer hover:bg-purple-700 transition-colors"
                    >
                      Send
                    </button>
                  </div>
                </div>
              )}
              {recs.filter((r) => r.discipline_id === selected?.name).length === 0 ? (
                <div className="text-xs text-gray-400 bg-gray-50 rounded-lg p-4 text-center">
                  No recommendations for this discipline yet
                </div>
              ) : (
                recs
                  .filter((r) => r.discipline_id === selected?.name)
                  .map((r, i) => (
                    <div
                      key={i}
                      className="border border-gray-100 rounded-lg p-3 mb-2 text-sm"
                      style={{
                        borderLeft: "3px solid #7c3aed",
                      }}
                    >
                      <div className="text-gray-400 mb-1 text-xs">
                        [{r.suggestion_type}] {r.competency_id}
                      </div>
                      <div className="text-gray-900">{r.suggestion}</div>
                      <button
                        onClick={() => deleteRec(r.id)}
                        className="mt-2 text-xs text-red-500 border border-red-500 rounded px-2 py-0.5 hover:bg-red-50 transition-colors bg-transparent cursor-pointer"
                      >
                        Delete
                      </button>
                    </div>
                  ))
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
