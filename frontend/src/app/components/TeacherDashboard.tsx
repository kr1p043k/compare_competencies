import { useState, useEffect, useRef } from "react";
import { api } from "../api";
import { useTheme } from "../../lib/theme";
import { AnalysisPanel } from "./AnalysisPanel";
import { TaxonomyBrowser } from "./TaxonomyBrowser";
import CompetencyTrendsPanel from "./CompetencyTrendsPanel";
import { reportError } from "./../../lib/errors";
import { ApiError, request } from "../../lib/api";

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
  /** Охват спроса: доля релевантных рынку навыков, реально закрытых. */
  average_market_top_coverage?: number;
  /** False, когда релевантных навыков слишком мало для процента. */
  average_market_top_reliable?: boolean;
  average_quality_coverage?: number;
  coverage_level: string;
  total_gaps_across_all: number;
  top_cross_discipline_gaps: { skill: string; disciplines: number }[];
  top_emerging_across_all: { skill: string; frequency: number }[];
  recommendations: { type: string; priority: string; message: string }[];
  trends: { rising: any[]; declining: any[] };
  disciplines: { name: string; coverage_ratio: number; weighted_coverage?: number; coverage_level: string; gaps: number; emerging: number; market_top_coverage?: number; market_top_total?: number; market_top_reliable?: boolean; market_top_source?: string; years?: number[] }[];
  generated_at: string;
};

type Discipline = {
  id: string;
  name: string;
  in_scope?: boolean;
  scope_source?: string;
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
  const { theme } = useTheme();
  const dk = theme === "dark";
  const [disciplines, setDisciplines] = useState<Discipline[]>([]);
  const [selected, setSelected] = useState<DisciplineDetail | null>(null);
  const [scopeSaving, setScopeSaving] = useState<string | null>(null);
  // Фильтр по году набора: дисциплина может быть только в одном плане.
  const [yearFilter, setYearFilter] = useState<0 | 2024 | 2025>(0);
  const [metaStale, setMetaStale] = useState(false);
  const [scopeMsg, setScopeMsg] = useState("");

  async function saveScopeBulk(included: boolean) {
    setScopeSaving("__all__"); setScopeMsg("");
    try {
      const names = disciplines.map((d) => d.name);
      await api(`/teacher/zun/scope`, {
        method: "PUT",
        body: JSON.stringify({ dir_code: selectedDir, changes: names.map((n) => ({ discipline_name: n, included })) }),
      });
      setDisciplines((prev) => prev.map((d) => ({ ...d, in_scope: included, scope_source: "custom" })));
      setScopeMsg("Scope обновлён для всех. Перезапустите teacher-анализ, чтобы средние пересчитались.");
    } catch (e: any) {
      setScopeMsg("Ошибка scope: " + (e.message || "неизвестная ошибка"));
    } finally {
      setScopeSaving(null);
    }
  }

  async function saveScope(name: string, included: boolean) {
    setScopeSaving(name); setScopeMsg("");
    try {
      await api(`/teacher/zun/scope`, {
        method: "PUT",
        body: JSON.stringify({ dir_code: selectedDir, changes: [{ discipline_name: name, included }] }),
      });
      setDisciplines((prev) => prev.map((d) =>
        d.name === name ? { ...d, in_scope: included, scope_source: "custom" } : d));
      setScopeMsg("Scope обновлён. Перезапустите teacher-анализ, чтобы средние пересчитались.");
    } catch (e: any) {
      let msg = e.message || "неизвестная ошибка";
      try { const j = JSON.parse(msg); msg = j.detail || msg; } catch (err) { reportError("app/components/TeacherDashboard.tsx:145", err); }
      setScopeMsg("Ошибка scope: " + msg);
    } finally {
      setScopeSaving(null);
    }
  }

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
  const [analysisMode, setAnalysisMode] = useState<"coverage" | "trends" | "taxonomy">("coverage");
  const [teacherSection, setTeacherSection] = useState("t-overview");
  const [collapsedComps, setCollapsedComps] = useState<Record<string, boolean>>({});
  const scrollTeacherTo = (id: string) => {
    setTeacherSection(id);
    try {
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) { reportError("app/components/TeacherDashboard.tsx:169", err); }
  };
  // Подсветка активного раздела при скролле (иначе залипает на кликнутом).
  useEffect(() => {
    const root = document.getElementById("teacher-main");
    if (!root) return;
    const ids = ["t-overview", "t-recs", "t-gaps", "t-trends", "t-discs",
      "d-cover", "d-recs", "d-top", "d-gaps", "d-comps"];
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) setTeacherSection(e.target.id);
        }
      },
      { root, rootMargin: "-20% 0px -70% 0px", threshold: 0 }
    );
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, [analysis, analysisMode]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [seedMsg, setSeedMsg] = useState("");
  const [seedLoading, setSeedLoading] = useState(false);
  const [zunForm, setZunForm] = useState<{ compId: string; ksaType: string; text: string } | null>(null);
  const [compForm, setCompForm] = useState(false);
  const [compCode, setCompCode] = useState("");
  const [compName, setCompName] = useState("");
  const [compErr, setCompErr] = useState<{ code?: string; name?: string; msg?: string }>({});
  const [compSaving, setCompSaving] = useState(false);

  const COMP_CODE_RE = /^(УК|ОПК|ПК|ППК|ИП|ВПК)[- ]\d+(\.\d+)*$/i;

  async function saveCompetency() {
    const code = compCode.trim().toUpperCase();
    const errs: { code?: string; name?: string } = {};
    if (!COMP_CODE_RE.test(code)) errs.code = "Формат: УК-1, ОПК-2, ПК-2.1…";
    if (compName.trim().length > 500) errs.name = "Слишком длинное (макс. 500)";
    setCompErr(errs);
    if (errs.code || errs.name || !selected) return;
    setCompSaving(true);
    try {
      await api(`/teacher/zun/disciplines/${selected.id}/competencies`, {
        method: "POST",
        body: JSON.stringify({ code, name: compName.trim() }),
      });
      setCompForm(false); setCompCode(""); setCompName(""); setCompErr({});
      await loadDiscipline(selected.name);
    } catch (e: any) {
      let msg = e.message || "неизвестная ошибка";
      try { const j = JSON.parse(msg); msg = j.detail || msg; } catch (err) { reportError("app/components/TeacherDashboard.tsx:220", err); }
      if (/code|формат/i.test(msg)) setCompErr({ code: msg });
      else setCompErr({ msg });
    } finally {
      setCompSaving(false);
    }
  }
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
      .catch((err) => reportError("app/components/TeacherDashboard.tsx", err));
    api(`/teacher/analysis?dir_code=${selectedDir}`)
      .then(setAnalysis)
      .catch(() => setAnalysis(null));
    api(`/teacher/analysis/meta?dir_code=${selectedDir}`)
      .then((m: any) => setMetaStale(!!m?.stale))
      .catch(() => setMetaStale(false));
  }, [selectedDir]);

  useEffect(() => {
    if (collectCooldown <= 0) return;
    const t = setTimeout(() => setCollectCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => clearTimeout(t);
  }, [collectCooldown]);

  useEffect(() => {
    api("/teacher/rpd/sources")
      .then(setRpdSources)
      .catch((err) => reportError("app/components/TeacherDashboard.tsx", err));
  }, []);

  // Возобновление поллинга после переключения вкладок (табы размонтируют компонент).
  useEffect(() => {
    let stored: string | null = null;
    try { stored = sessionStorage.getItem("rpdRunId"); } catch (err) { reportError("app/components/TeacherDashboard.tsx:302", err); }
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
            try { sessionStorage.removeItem("rpdRunId"); } catch (err) { reportError("app/components/TeacherDashboard.tsx:323", err); }
            setRpdMsg(s?.status === "completed" ? "Готово" : s?.status === "cancelled" ? "Отменено пользователем" : `Ошибка: ${s?.error || "unknown"}`);
            if (s?.status === "completed") {
              api(`/teacher/analysis?dir_code=${selectedDir}`)
                .then(setAnalysis)
                .catch((err) => reportError("app/components/TeacherDashboard.tsx", err));
            }
          } else {
            setTimeout(poll, 3000);
          }
        })
        .catch(() => {
          if (cancelled) return;
          // Разовый сбой сети/рестарт сервера - ждём, а не висим молча.
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
      const data = await request<{ run_id?: string }>("/api/teacher/rpd/upload", {
        method: "POST",
        body: fd,
      });
      if (!data.run_id) throw new Error("Сервер не вернул ID задачи");
      try { sessionStorage.setItem("rpdRunId", data.run_id); } catch (err) { reportError("app/components/TeacherDashboard.tsx:369", err); }
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
        await request<void>(`/api/teacher/rpd/cancel/${rpdRun.run_id}`, { method: "POST" });
      } catch {
        setRpdCollecting(false);
        setRpdMsg("");
      }
      return;
    }
    setRpdCollecting(true); setRpdMsg("Сбор аннотаций с Yandex Disk...");
    try {
      const body = new URLSearchParams({ dir_code: selectedDir });
      if (rpdYandexUrl.trim()) body.set("public_url", rpdYandexUrl.trim());
      let data: { run_id?: string };
      try {
        data = await request<{ run_id?: string }>("/api/teacher/rpd/collect", {
          method: "POST",
          body,
        });
      } catch (err) {
        if (err instanceof ApiError && err.isRateLimited) {
          setCollectCooldown(30);
          throw new Error("Слишком частые запросы - кнопка заблокирована на 30 секунд");
        }
        throw err;
      }
      if (!data.run_id) throw new Error("Сервер не вернул ID задачи");
      try { sessionStorage.setItem("rpdRunId", data.run_id); } catch (err) { reportError("app/components/TeacherDashboard.tsx:417", err); }
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
    } catch (err) { reportError("app/components/TeacherDashboard.tsx:500", err); }
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
    } catch (err) { reportError("app/components/TeacherDashboard.tsx:527", err); }
  }

  async function deleteRec(id: number) {
    try {
      await api(`/teacher/krm/recommendations/${id}`, { method: "DELETE" });
      setRecs((prev) => prev.filter((r) => r.id !== id));
    } catch (err) { reportError("app/components/TeacherDashboard.tsx:534", err); }
  }

  if (loading) {
    return (
      <div className="p-10 text-gray-500">
        Загрузка...
      </div>
    );
  }

  const analysisByName = new Map((analysis?.disciplines ?? []).map((a) => [a.name, a]));
  const filtered = disciplines.filter((d) => {
    if (!d.name.toLowerCase().includes(search.toLowerCase())) return false;
    if (yearFilter === 0) return true;
    const years = analysisByName.get(d.name)?.years;
    return Array.isArray(years) && years.includes(yearFilter);
  });

  const containerStyle: React.CSSProperties = {
    display: "flex",
    height: "calc(100vh - 100px)",
    fontFamily: "system-ui, -apple-system, sans-serif",
    color: dk ? "#f1f5f9" : "#111827",
    background: dk ? "#0f172a" : "#fff",
    borderRadius: 16,
    border: "1px solid " + (dk ? "#334155" : "#e5e7eb"),
    overflow: "hidden",
  };

  const sidebarStyle: React.CSSProperties = {
    width: 380,
    minWidth: 380,
    borderRight: "1px solid " + (dk ? "#334155" : "#e5e7eb"),
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
    background: dk ? "#0f172a" : "#fff",
    borderRadius: 8,
    border: "1px solid " + (dk ? "#334155" : "#e5e7eb"),
    padding: 14,
    marginBottom: 10,
    boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
  };

  function covColor(cov: number) {
    return cov > 0.5 ? "#059669" : cov > 0.2 ? "#d97706" : "#dc2626";
  }

  function covTextClass(cov: number) {
    return cov > 0.5 ? "text-emerald-600" : cov > 0.2 ? "text-amber-600" : "text-red-600";
  }

  return (
    <div style={containerStyle}>
      <div style={sidebarStyle}>
          <div className="p-4 border-b border-gray-200 dark:border-slate-700">
            <div className="flex justify-between items-center">
              <h1 className="text-lg m-0 text-gray-900 dark:text-slate-100">
                KRM Teacher
              </h1>
              {stats && (
                  <div className="text-xs text-gray-500 dark:text-slate-400 text-right">
                    <div>{(() => { const n = (stats as any).total_disciplines ?? (stats as any).total_reports ?? 0; return `${n} ${plural(n, "дисциплина", "дисциплины", "дисциплин")}`; })()}</div>
                    <div>{(() => { const n = (stats as any).total_competencies ?? Object.keys((stats as any).by_profession || {}).length; return `${n} ${plural(n, "компетенция", "компетенции", "компетенций")}`; })()}</div>
                    <div>{(() => { const n = (stats as any).total_skills ?? 0; return `${n.toLocaleString()} ${plural(n, "навык", "навыка", "навыков")}`; })()}</div>
                  </div>
                )}
          </div>

          {/* Direction selector */}
          <select
            value={selectedDir}
            onChange={(e) => setSelectedDir(e.target.value)}
            className="w-full mt-2.5 px-2 py-1.5 rounded-md text-xs" style={{
    background: dk ? "#0f172a" : "#fff",
    border: "1px solid " + (dk ? "#334155" : "#e5e7eb"),
    color: dk ? "#f1f5f9" : "#111827",
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
                  if (i === 359) setRunMsg("Превышено ожидание - проверьте результат позже");
                }
              } catch (e: any) {
                setRunMsg(`Ошибка запуска: ${e?.message || e}`);
              } finally {
                setRunLoading(false);
              }
            }}
            disabled={runLoading}
            className="w-full mt-2.5 px-3 py-2 border-0 rounded-md text-xs font-semibold" style={{
    background: runLoading ? "#9ca3af" : "#7c3aed",
    color: "#fff",
    cursor: runLoading ? "default" : "pointer",
  }}
          >
            {runLoading ? "Анализ запущен..." : "Запустить анализ"}
          </button>
            {runMsg && (
              <div className="text-xs text-amber-700 mt-1.5">{runMsg}</div>
            )}

          {/* RPD upload block */}
          <div
            className="mt-3 px-3 py-2.5 rounded-md" style={{
    background: dk ? "#2a230f" : "#fffbeb",
    border: "1px solid " + (dk ? "#57481c" : "#fde68a"),
  }}
          >
            <div className="text-xs font-bold text-amber-800 mb-2">
              Подгрузка РПД
            </div>
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
            <button
              onClick={() => rpdInputRef.current?.click()}
              disabled={rpdUploading}
              className="w-full px-3 py-2 border-0 rounded-md text-xs font-semibold" style={{
    background: rpdUploading ? "#9ca3af" : "#7c3aed",
    color: "#fff",
    cursor: rpdUploading ? "default" : "pointer",
  }}
            >
              {rpdUploading ? "Обработка..." : "Загрузить PDF и обработать"}
            </button>
            {rpdSources.yandex_covered?.includes(selectedDir) && (
              <button
                onClick={collectRpd}
                disabled={collectCooldown > 0}
                className="w-full mt-2 px-3 py-2 border-0 rounded-md text-xs font-semibold" style={{
    background: rpdCollecting ? "#dc2626" : collectCooldown > 0 ? "#9ca3af" : "#7c3aed",
    color: "#fff",
    cursor: collectCooldown > 0 ? "default" : "pointer",
  }}
              >
                {rpdCollecting ? "Отмена" : collectCooldown > 0 ? `Подождите ${collectCooldown} сек` : "Собрать с Yandex Disk"}
              </button>
            )}
            <div className="flex flex-col gap-1.5 border-t border-dashed border-amber-200 dark:border-amber-900 pt-2 mt-2">
              <input
                type="text"
                value={rpdYandexUrl}
                onChange={(e) => setRpdYandexUrl(e.target.value)}
                placeholder="Ссылка на папку Yandex Disk (https://disk.360.yandex.ru/d/...)"
                className="w-full px-2.5 py-2 rounded-md text-xs box-border" style={{
    border: "1px solid " + (dk ? "#57481c" : "#fde68a"),
  }}
              />
              <button
                onClick={collectRpd}
                disabled={rpdCollecting || collectCooldown > 0 || !rpdYandexUrl.trim()}
                className="w-full px-3 py-2 border-0 rounded-md text-xs font-semibold" style={{
    background: rpdCollecting || collectCooldown > 0 || !rpdYandexUrl.trim() ? "#9ca3af" : "#0d9488",
    color: "#fff",
    cursor: rpdCollecting || collectCooldown > 0 || !rpdYandexUrl.trim() ? "default" : "pointer",
  }}
              >
                {rpdCollecting ? "Сбор..." : "Загрузить по ссылке"}
              </button>
            </div>
            {rpdStatus && (rpdStatus.status === "completed" || rpdStatus.status === "failed") && (
              <div
                className="mt-2 text-xs" style={{
    color: rpdStatus.status === "completed" ? "#15803d" : "#b91c1c",
  }}
              >
                {rpdMsg}
                {rpdStatus.stats?.disciplines != null && rpdStatus.status === "completed" && (
                  <span> - {rpdStatus.stats.disciplines} дисциплин</span>
                )}
              </div>
            )}
            {(rpdUploading || rpdCollecting) && rpdStatus && (rpdStatus.status === "running" || rpdStatus.status === "started") && (
              <div className="mt-2.5">
                <div
                  className="h-2 rounded overflow-hidden" style={{
    background: dk ? "#57481c" : "#fde68a",
  }}
                >
                  <div
                    className="h-full bg-teal-600 transition-[width] duration-500" style={{
    width: `${rpdStatus.stats?.progress ?? 10}%`,
  }}
                  />
                </div>
                <div className="mt-1.5 text-xs text-amber-800">
                  Этап: {rpdStageLabel(rpdStatus.stats?.stage)} · {rpdStatus.stats?.progress ?? 10}%
                </div>
              </div>
            )}
          </div>

          {/* Analysis summary button */}
          {analysis && (
            <div
              onClick={() => setShowAnalysis(!showAnalysis)}
              className="mt-2.5 px-3 py-2 rounded-md cursor-pointer text-xs" style={{
    background: dk ? "#1e293b" : "#f9fafb",
    border: `1px solid ${covColor(analysis.average_coverage)}`,
  }}
            >
              <div className="flex justify-between">
                {/* Показываем охват спроса, а не coverage_ratio: второй почти
                    всегда ~100%, потому что навыки РПД уже нормализованы к
                    словарю рынка, и читается как «всё покрыто». */}
                <span className="text-violet-600 font-semibold">Охват рынка</span>
                <span
                  className={`font-bold ${covTextClass(
                    analysis.average_market_top_coverage ?? analysis.average_coverage,
                  )}`}
                >
                  {((analysis.average_market_top_coverage ?? analysis.average_coverage) * 100).toFixed(1)}%
                  <span className="block text-xs mt-0.5 font-normal text-gray-500 dark:text-slate-400">
                    релевантных навыков рынка
                  </span>
                </span>
              </div>
              <div className="text-gray-500 dark:text-slate-400 mt-0.5">
                {analysis.total_disciplines} {plural(analysis.total_disciplines, "дисциплина", "дисциплины", "дисциплин")}, {analysis.total_gaps_across_all} {plural(analysis.total_gaps_across_all, "пробел", "пробела", "пробелов")}
              </div>
            </div>
          )}

          <input
            placeholder="Поиск дисциплин..."
            aria-label="Поиск дисциплины"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full mt-2.5 px-3 py-2 rounded-md text-[13px] box-border" style={{
    background: dk ? "#0f172a" : "#fff",
    border: "1px solid " + (dk ? "#334155" : "#e5e7eb"),
    color: dk ? "#f1f5f9" : "#111827",
    outline: "none",
  }}
          />
        </div>
        <div className="flex-1 overflow-auto">
          <div className="px-4 py-1.5 text-xs text-gray-500 dark:text-slate-400 border-b border-gray-200 dark:border-slate-700">
            В учёте: {disciplines.filter((x) => x.in_scope ?? true).length} из {disciplines.length}
            {scopeMsg && (<div className="text-amber-800 mt-0.5">{scopeMsg}</div>)}
            <div className="flex gap-1.5 mt-1 items-center flex-wrap">
              <button onClick={() => saveScopeBulk(true)} disabled={scopeSaving !== null}
                className="text-xs px-2.5 py-0.5 rounded border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-900 cursor-pointer">
                Выбрать все
              </button>
              <button onClick={() => saveScopeBulk(false)} disabled={scopeSaving !== null}
                className="text-xs px-2.5 py-0.5 rounded border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-900 cursor-pointer">
                Снять все
              </button>
              {metaStale && (<span className="text-xs text-amber-800 dark:text-amber-200 bg-amber-100 dark:bg-amber-950 rounded px-1.5 py-0.5">данные устарели - обновите анализ</span>)}
            </div>
          </div>
          <div className="flex gap-1.5 px-2 pb-2">
            {([0, 2024, 2025] as const).map((y) => (
              <button
                key={y}
                onClick={() => setYearFilter(y)}
                className={
                  "text-xs px-2 py-0.5 rounded border cursor-pointer " +
                  (yearFilter === y
                    ? "border-violet-400 bg-violet-50 text-violet-700 dark:bg-violet-950 dark:text-violet-200"
                    : "border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-gray-600 dark:text-slate-300")
                }
              >
                {y === 0 ? "Все годы" : y}
              </button>
            ))}
          </div>
          {filtered.map((d) => {
            const discAnalysis = analysis?.disciplines.find((a) => a.name === d.name);
            return (
              <div
                key={d.name}
                onClick={() => loadDiscipline(d.name)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === "Enter") loadDiscipline(d.name); }}
                className="flex gap-2 items-start py-2.5 pl-2 pr-3 cursor-pointer" style={{
    borderBottom: "1px solid " + (dk ? "#334155" : "#e5e7eb"),
    background: selected?.name === d.name ? dk ? "#1d2440" : "#eef2ff" : "transparent",
    opacity: (d.in_scope ?? true) ? 1 : 0.55,
  }}
              >
                <input
                  type="checkbox"
                  checked={d.in_scope ?? true}
                  aria-label="Учитывать в анализе"
                  disabled={scopeSaving === d.name}
                  title={(d.scope_source === "methodology" ? "Исключена методологией (можно вернуть). " : "") + "Учитывать в анализе"}
                  onChange={(e) => saveScope(d.name, e.target.checked)}
                  onClick={(e) => e.stopPropagation()}
                  className="size-[15px] accent-violet-600 cursor-pointer shrink-0 mt-0.5"
                />
                <div className="flex-1 min-w-0">
                <div className="flex justify-between flex-1 min-w-0">
                  <div className="text-[13px] font-semibold text-violet-600 truncate flex-1 min-w-0">
                    {d.name}
                    {(() => {
                      const years = analysisByName.get(d.name)?.years;
                      if (!years || years.length === 0) return null;
                      const only = years.length === 1 ? years[0] : null;
                      return (
                        <span
                          title={only ? `Только в наборе ${only}` : `В наборах ${years.join(" и ")}`}
                          className={
                            "ml-1.5 align-middle text-xs font-medium rounded px-1.5 py-0.5 whitespace-nowrap " +
                            (only === 2024
                              ? "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                              : only === 2025
                                ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                                : "bg-gray-100 text-gray-500 dark:bg-slate-800 dark:text-slate-400")
                          }
                        >
                          {only ?? years.join("/")}
                        </span>
                      );
                    })()}
                  </div>
                  {discAnalysis && (
                    <span className={`text-xs font-bold ${covTextClass(discAnalysis.coverage_ratio)}`}>
                      {(discAnalysis.coverage_ratio * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-500 dark:text-slate-400 mt-0.5">
                    {d.scope_source === "methodology" && (
                      <span className="mr-1.5 text-xs text-purple-700 dark:text-purple-300 bg-purple-100 dark:bg-purple-950 rounded px-1 py-0.5">методология</span>
                    )}
                    {d.scope_source === "custom" && (
                      <span className="mr-1.5 text-xs text-amber-800 dark:text-amber-200 bg-amber-100 dark:bg-amber-950 rounded px-1 py-0.5">вручную</span>
                    )}
                  {d.course != null && `${d.course} курс · `}
                  {d.competencies_count} {plural(d.competencies_count, "комп.", "комп.", "комп.")} · {d.skills_count} {plural(d.skills_count, "навык", "навыка", "навыков")}
                  {d.abilities_count != null && ` · ${d.abilities_count} ${plural(d.abilities_count, "умение", "умения", "умений")}`}
                  {d.knowledge_count != null && ` · ${d.knowledge_count} ${plural(d.knowledge_count, "знание", "знания", "знаний")}`}
                  {discAnalysis && ` / ${discAnalysis.gaps} ${plural(discAnalysis.gaps, "пробел", "пробела", "пробелов")}`}
                </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div style={mainStyle} id="teacher-main">
        {!selected && !showAnalysis && (
          <div className="text-center mt-20 text-gray-400 text-sm">
            Выберите дисциплину или откройте анализ
          </div>
        )}

        {/* Always visible mode toggle */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setAnalysisMode("coverage")}
            className="px-3.5 py-1.5 border-0 rounded-md cursor-pointer text-xs" style={{
    fontWeight: analysisMode === "coverage" ? 700 : 400,
    background: analysisMode === "coverage" ? "#7c3aed" : dk ? "#1e293b" : "#f9fafb",
    color: analysisMode === "coverage" ? "#fff" : "#7c3aed",
  }}
          >
            Покрытие
          </button>
            <button
              onClick={() => setAnalysisMode("trends")}
              className="px-3.5 py-1.5 border-0 rounded-md cursor-pointer text-xs" style={{
    fontWeight: analysisMode === "trends" ? 700 : 400,
    background: analysisMode === "trends" ? "#7c3aed" : dk ? "#1e293b" : "#f9fafb",
    color: analysisMode === "trends" ? "#fff" : "#7c3aed",
  }}
            >
              Тренды компетенций
            </button>
            <button
              onClick={() => setAnalysisMode("taxonomy")}
              className="px-3.5 py-1.5 border-0 rounded-md cursor-pointer text-xs" style={{
    fontWeight: analysisMode === "taxonomy" ? 700 : 400,
    background: analysisMode === "taxonomy" ? "#7c3aed" : dk ? "#1e293b" : "#f9fafb",
    color: analysisMode === "taxonomy" ? "#fff" : "#7c3aed",
  }}
            >
              Таксономия
            </button>
        </div>

        {analysisMode === "coverage" && analysis && (<>
          {/* Суб-навигация: при выбранной дисциплине - по её разделам, иначе по сводке */}
          <div className="inline-flex items-center gap-1 rounded-lg bg-gray-100 dark:bg-slate-900 p-1 mb-4 sticky top-0 z-10 flex-wrap">
            {(selected
              ? [
                  ["d-cover", "Покрытие"],
                  ["d-recs", "Рекомендации"],
                  ["d-top", "Совпадения"],
                  ["d-gaps", "Пробелы"],
                  ["d-comps", "Компетенции"],
                ]
              : [
                  ["t-overview", "Обзор"],
                  ["t-recs", "Рекомендации"],
                  ["t-gaps", "Разрывы"],
                  ["t-trends", "Тренды"],
                  ["t-discs", "Дисциплины"],
                ]
            ).map(([id, label]) => (
              <button
                key={id}
                onClick={() => scrollTeacherTo(id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all cursor-pointer ${
                  teacherSection === id
                    ? "bg-white text-gray-900 shadow-sm dark:bg-slate-800 dark:text-slate-100"
                    : "text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-200"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <div id="t-overview" className="text-xs text-gray-500 dark:text-slate-400 mb-2 scroll-mt-2">
            {selected
              ? `Дисциплина: ${selected.name} (направление ${selectedDir})`
              : `Направление ${selectedDir} - сводка по всем ${analysis.total_disciplines} дисциплинам, ни одна дисциплина не выбрана`}
          </div>
          {/* When a discipline is selected, show its coverage instead of direction average */}
          {(() => {
            const discData = selected && analysis.disciplines
              ? analysis.disciplines.find((d: any) => d.name === selected.name)
              : null;
            // Охват рынка вместо coverage_ratio (см. комментарий в KPI выше).
            const cov = discData
              ? discData.market_top_coverage ?? discData.coverage_ratio
              : analysis.average_market_top_coverage ?? analysis.average_coverage;
            const covTotal = discData?.market_top_total;
            const covReliable = discData
              ? discData.market_top_reliable !== false
              : analysis.average_market_top_reliable !== false;
            const gaps = discData ? discData.gaps : analysis.total_gaps_across_all;
            const level = discData ? discData.coverage_level : analysis.coverage_level;
          return (<>
          <div className="flex gap-3 mb-4 flex-wrap">
            <div style={card}>
              <div className="text-xs text-gray-500 dark:text-slate-400">
                {discData ? `Охват рынка: ${selected?.name ?? ""}` : "Средний охват рынка"}
                {covTotal ? ` (из ${covTotal})` : ""}
              </div>
                  <div className={`text-2xl font-bold ${covReliable ? covTextClass(cov) : "text-gray-400"}`}>
                    {covReliable ? `${(cov * 100).toFixed(1)}%` : "н/д"}
                  </div>
                  <div className="text-xs font-semibold text-gray-500 dark:text-slate-400">
                    {covReliable ? (level || "").toUpperCase() : "мало релевантных навыков"}
                  </div>
                  {/* Откуда знаменатель: ручная привязка, автоподбор по навыкам
                      дисциплины или слабая оценка по названию. */}
                  {discData?.market_top_source && (
                    <div className="text-xs text-gray-400 dark:text-slate-500 mt-0.5">
                      {discData.market_top_source === "explicit"
                        ? "знаменатель: ручная привязка"
                        : discData.market_top_source === "derived"
                          ? "знаменатель: авто по навыкам"
                          : "знаменатель: по профилю дисциплины"}
                    </div>
                  )}
                </div>
                <div style={card}>
                  <div className="text-xs text-gray-500 dark:text-slate-400">Gaps</div>
                  <div className="text-2xl font-bold text-red-300">
                    {gaps}
                  </div>
                </div>
                <div style={card}>
                  <div className="text-xs text-gray-500 dark:text-slate-400">Disciplines</div>
                  <div className="text-2xl font-bold text-blue-300">
                    {analysis.total_disciplines}
                  </div>
                </div>
              </div>
          </>)})()}

              {/* Общие блоки направления - только пока дисциплина не выбрана.
                  При выбранной дисциплине её детали ниже (AnalysisPanel). */}
              {!selected && (<>
              {/* Direction-level recommendations */}
              {(analysis.recommendations || []).length > 0 && (
                <div id="t-recs" className={`${card} scroll-mt-2`}>
                  <div className="text-xs font-semibold text-violet-600 mb-2">
                    Рекомендации
                  </div>
                  {(analysis.recommendations || []).map((r, i) => (
                    <div
                      key={i}
                      className={`mb-1.5 rounded-md border px-2.5 py-2 text-xs ${
                        r.priority === "high"
                          ? "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950/40"
                          : r.priority === "medium"
                            ? "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-950/40"
                            : "border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950/40"
                      }`}
                    >
                      <div className="flex gap-1.5 items-center mb-1">
                        <span
                          className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${
                            r.priority === "high"
                              ? "bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-300"
                              : r.priority === "medium"
                                ? "bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-200"
                                : "bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300"
                          }`}
                        >
                          {r.priority}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-slate-400">{r.type}</span>
                      </div>
                      <div className="text-gray-700 dark:text-slate-300 leading-snug">{r.message}</div>
                    </div>
                  ))}
                </div>
              )}

                 <div id="t-gaps" className={`${card} scroll-mt-2`}>
                <div className="text-xs font-semibold text-violet-600 mb-2">Междисциплинарные разрывы</div>
                {((analysis.top_cross_discipline_gaps || []) as any[]).map((g: any, i: number) => (
                  <div key={i} className="flex justify-between py-1 border-b border-gray-200 dark:border-slate-700 text-xs">
                    <span className="text-red-700 font-medium">{g.skill}</span>
                    <span className="text-gray-500 dark:text-slate-400">{g.disciplines} дисциплин</span>
                  </div>
                ))}
              </div>

              <div style={card}>
                <div className="text-xs font-semibold text-violet-600 mb-2">Востребованные навыки рынка</div>
                <div className="flex gap-1 flex-wrap">
                  {((analysis.top_emerging_across_all || []) as any[]).map((s: any, i: number) => (
                    <span key={i} className="inline-block px-2 py-0.5 rounded text-xs bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 m-0.5">
                      {s.skill}
                    </span>
                  ))}
                </div>
              </div>

              {analysis.trends && (
                <div id="t-trends" className="flex gap-3 flex-wrap scroll-mt-2">
                  {analysis.trends.rising?.length > 0 && (
                    <div className={`${card} flex-1 min-w-[200px]`}>
                      <div className="text-xs font-semibold text-emerald-600 mb-2">Растущие навыки</div>
                      {analysis.trends.rising.map((t, i) => (
                        <div key={i} className="text-xs py-0.5 text-gray-600 dark:text-slate-400">
                          {t.skill} <span className="text-emerald-600">{t.change_pct > 0 ? "+" : ""}{t.change_pct}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {analysis.trends.declining?.length > 0 && (
                    <div className={`${card} flex-1 min-w-[200px]`}>
                      <div className="text-xs font-semibold text-red-600 mb-2">Падающие навыки</div>
                      {analysis.trends.declining.map((t, i) => (
                        <div key={i} className="text-xs py-0.5 text-gray-600 dark:text-slate-400">
                          {t.skill} <span className="text-red-600">{t.change_pct}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div id="t-discs" className={`${card} scroll-mt-2`}>
                <div className="text-xs font-semibold text-violet-600 mb-2">Разбивка по дисциплинам</div>
                {(() => {
                  const aMap = new Map(analysis.disciplines.map((x) => [x.name, x]));
                  const rows = disciplines.map((k) => ({
                    name: k.name,
                    a: aMap.get(k.name),
                    inScope: k.in_scope ?? true,
                    src: k.scope_source ?? "default",
                  }));
                  const inN = rows.filter((r) => r.inScope).length;
                  return (<>
                    <div className="text-xs text-gray-500 dark:text-slate-400 mb-1.5">
                      В учёте: {inN} из {rows.length}
                      {scopeMsg && (<span className="ml-2 text-amber-800">{scopeMsg}</span>)}
                    </div>
                    {rows.map((r, i) => (
                    <div key={i} className="flex items-center gap-2 py-1.5 border-b border-gray-200 dark:border-slate-700 text-xs">
                      <input
                        type="checkbox"
                        checked={r.inScope}
                          aria-label="Учитывать в анализе"
                        disabled={scopeSaving === r.name}
                        title={r.src === "methodology" ? "Исключена методологией (можно вернуть)" : "Учитывать в анализе"}
                        onChange={(e) => saveScope(r.name, e.target.checked)}
                        onClick={(e) => e.stopPropagation()}
                        className="size-[15px] accent-violet-600 cursor-pointer shrink-0"
                      />
                      <span
                        className={"text-gray-700 dark:text-slate-300 max-w-[240px] truncate flex-1"}
                        onClick={() => { if (r.a) { const found = disciplines.find((dd) => dd.name === r.name); if (found) loadDiscipline(found.name); } }}
                      >
                        {r.name}
                        {r.src === "methodology" && (
                          <span className="ml-1.5 text-xs text-purple-700 dark:text-purple-300 bg-purple-100 dark:bg-purple-950 rounded px-1 py-0.5">методология</span>
                        )}
                        {r.src === "custom" && (
                          <span className="ml-1.5 text-xs text-amber-800 dark:text-amber-200 bg-amber-100 dark:bg-amber-950 rounded px-1 py-0.5">вручную</span>
                        )}
                      </span>
                      {r.a ? (
                        <div className="flex gap-3">
                          <span className={`font-semibold ${covTextClass(r.a.coverage_ratio)}`}>{(r.a.coverage_ratio * 100).toFixed(1)}%</span>
                          {r.a.weighted_coverage != null && (
                            <span className={`text-xs ${covTextClass(r.a.weighted_coverage)}`} title="Взвешенное покрытие (качество)">Кач.:{(r.a.weighted_coverage * 100).toFixed(0)}%</span>
                          )}
                          <span className="text-red-600" title="Пробелов">{r.a.gaps} пр.</span>
                          <span className="text-blue-600" title="Новые навыки рынка">{r.a.emerging} нов.</span>
                        </div>
                      ) : (
                        <span className="text-xs text-gray-400">вне учёта</span>
                      )}
                    </div>
                    ))}
                  </>);
                })()}
              </div>
              </>)}
            </>)}

          {analysisMode === "trends" && (
            <CompetencyTrendsPanel
              dirCode={selectedDir}
              competencyCodes={selected ? selected.competencies.map((c) => c.code) : undefined}
            />
          )}

          {analysisMode === "taxonomy" && (
            <div className="mt-4">
              <TaxonomyBrowser showSuggest />
            </div>
          )}

        {/* Discipline detail */}
        {selected && (
          <>
            <h2 className="text-xl mb-5 text-gray-900 dark:text-slate-100">
              {selected.name}
            </h2>

            {/* Analysis panel for this discipline */}
            <AnalysisPanel disciplineName={selected.name} dirCode={selectedDir} />

            <div className="mb-3">
              {!compForm ? (
                <button onClick={() => { setCompForm(true); setCompErr({}); }}
                  className="text-xs text-white bg-emerald-600 hover:bg-emerald-700 transition-colors px-3 py-1.5 rounded-lg cursor-pointer border-0">
                  + Компетенция
                </button>
              ) : (
                <div className="rounded-lg border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/30 p-3 space-y-2">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    <div>
                      <input value={compCode} onChange={(e) => setCompCode(e.target.value)}
                        placeholder="Код: ПК-2.1"
                        aria-label="Код компетенции"
                        className={`h-9 w-full px-2 rounded border bg-white dark:bg-slate-950 text-xs ${compErr.code ? "border-red-400" : "border-gray-300 dark:border-slate-600"}`} />
                      {compErr.code && <div className="text-xs text-red-600 mt-1">{compErr.code}</div>}
                    </div>
                    <div>
                      <input value={compName} onChange={(e) => setCompName(e.target.value)}
                        placeholder="Название (необязательно)"
                        aria-label="Название компетенции"
                        className={`h-9 w-full px-2 rounded border bg-white dark:bg-slate-950 text-xs ${compErr.name ? "border-red-400" : "border-gray-300 dark:border-slate-600"}`} />
                      {compErr.name && <div className="text-xs text-red-600 mt-1">{compErr.name}</div>}
                    </div>
                  </div>
                  {compErr.msg && <div className="text-xs text-red-600">{compErr.msg}</div>}
                  <div className="flex items-center gap-2">
                    <button onClick={saveCompetency} disabled={compSaving}
                      className="text-xs bg-emerald-600 text-white px-3 py-1.5 rounded-lg hover:bg-emerald-700 transition-colors cursor-pointer border-0 disabled:opacity-50">
                      {compSaving ? "..." : "Создать"}
                    </button>
                    <button onClick={() => setCompForm(false)}
                      className="text-xs text-gray-500 dark:text-slate-400 hover:text-gray-700 border-0 bg-transparent cursor-pointer">Отмена</button>
                  </div>
                  <div className="text-xs text-gray-400 dark:text-slate-500">ЗУН и инструменты дописываются после создания: +ЗУН у компетенции, инструменты - через Администрирование → Навыки.</div>
                </div>
              )}
            </div>

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
                className="mb-2 border border-gray-200 dark:border-slate-700 rounded-lg overflow-hidden"
              >
                <div className="px-4 py-2.5 bg-gray-50 dark:bg-slate-900 flex items-center gap-2">
                  <span className="font-semibold text-sm text-violet-600">
                    {comp.code}
                  </span>
                  <button
                    onClick={() => { setZunForm({ compId: comp.id, ksaType: "skills", text: "" }); setZunMsg(""); }}
                    className="ml-auto text-xs text-violet-600 hover:text-violet-800 border-0 bg-transparent cursor-pointer"
                    title="Добавить пункт (знание / умение / навык) в эту компетенцию"
                  >
                    + ЗУН
                  </button>
                  <span className="text-xs text-gray-400 dark:text-slate-500">
                    {total} {plural(total, "навык", "навыка", "навыков")}
                  </span>
                  <button
                    onClick={() => setCollapsedComps((p) => ({ ...p, [comp.id]: !(p[comp.id] ?? total === 0) }))}
                    className="text-gray-400 hover:text-gray-700 dark:hover:text-slate-200 border-0 bg-transparent cursor-pointer text-xs px-1"
                    title={(collapsedComps[comp.id] ?? total === 0) ? "Развернуть" : "Свернуть"}
                  >
                    {(collapsedComps[comp.id] ?? total === 0) ? "▸" : "▾"}
                  </button>
                </div>
                {!(collapsedComps[comp.id] ?? total === 0) && (
                <div className="px-4 py-2">
                {zunForm && zunForm.compId === comp.id && (
                  <div className="mb-2 rounded-lg border border-violet-200 dark:border-violet-800 bg-violet-50 dark:bg-violet-950/30 p-2">
                    <div className="flex gap-2 mb-2">
                      <select
                        value={zunForm.ksaType}
                        onChange={(e) => setZunForm({ ...zunForm, ksaType: e.target.value })}
                        className="text-xs border border-gray-300 dark:border-slate-600 rounded-md px-2 py-1 bg-white dark:bg-slate-950"
                      >
                        <option value="knowledge">Знания</option>
                        <option value="abilities">Умения</option>
                        <option value="skills">Навыки</option>
                      </select>
                      <button
                        onClick={() => setZunForm(null)}
                        className="text-xs text-gray-500 dark:text-slate-400 hover:text-gray-700 border-0 bg-transparent cursor-pointer"
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
                      className="w-full text-xs border border-gray-300 dark:border-slate-600 rounded-md px-2 py-1 mb-2"
                    />
                    <div className="flex items-center gap-2">
                      <button
                        onClick={addZunEntry}
                        disabled={zunSaving || !zunForm.text.trim()}
                        className="text-xs bg-violet-600 text-white px-3 py-1.5 rounded-lg hover:bg-violet-700 transition-colors cursor-pointer border-0 disabled:opacity-50"
                      >
                        {zunSaving ? "Сохранение…" : "Добавить"}
                      </button>
                      {zunMsg && <span className="text-xs text-gray-600 dark:text-slate-400">{zunMsg}</span>}
                    </div>
                    <div className="text-xs text-gray-400 dark:text-slate-500 mt-1">Пункт попадёт в анализ при следующем пересчёте.</div>
                  </div>
                )}
                  {total === 0 ? (
                    <div className="text-xs text-gray-400 dark:text-slate-500">Навыки не извлечены</div>
                  ) : hasGroups ? (
                    groups.map((g) => g.items.length > 0 && (
                      <div key={g.title} className="mb-2 last:mb-0">
                        <div className="text-xs font-semibold text-violet-500 uppercase tracking-wide mt-1.5 mb-1">
                          {g.title} ({g.items.length})
                        </div>
                        {g.items.map((sk) => (
                          <div key={sk.id} className="py-0.5 text-xs leading-relaxed border-b border-gray-100 dark:border-slate-800 last:border-0">
                            {ksaEditing && ksaEditing.id === sk.id ? (
                              <div className="flex gap-2 items-start">
                                <textarea
                                  value={ksaEditing.text}
                                  onChange={(e) => setKsaEditing({ ...ksaEditing, text: e.target.value })}
                                  rows={2}
                                  maxLength={2000}
                                  className="flex-1 text-xs border border-gray-300 dark:border-slate-600 rounded-md px-2 py-1"
                                />
                                <button onClick={saveKsaEdit} disabled={zunSaving} className="text-xs bg-violet-600 text-white px-2 py-1 rounded-md hover:bg-violet-700 cursor-pointer border-0 disabled:opacity-50">OK</button>
                                <button onClick={() => setKsaEditing(null)} className="text-xs text-gray-500 dark:text-slate-400 hover:text-gray-700 border-0 bg-transparent cursor-pointer">Отмена</button>
                              </div>
                            ) : (
                              <div className="flex gap-1 items-start group">
                                <span className="flex-1">{sk.text}</span>
                                <button onClick={() => { setKsaEditing({ id: sk.id, text: sk.text }); setZunMsg(""); }} title="Редактировать" className="text-gray-300 hover:text-violet-600 border-0 bg-transparent cursor-pointer text-xs">Изменить</button>
                                <button onClick={() => delKsa(sk.id)} title="Удалить" className="text-gray-300 hover:text-red-600 border-0 bg-transparent cursor-pointer text-xs">Удалить</button>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ))
                  ) : (
                    comp.skills.map((sk, i) => (
                      <div key={i} className="py-0.5 text-xs leading-relaxed border-b border-gray-100 dark:border-slate-800 last:border-0">
                        {sk}
                      </div>
                    ))
                  )}
                </div>
                )}
              </div>
              );
            })}

            <div className="mt-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="flex items-center justify-center w-8 h-8 bg-violet-600 rounded-lg">
                  <span className="text-white text-sm font-bold">!</span>
                </div>
                <h3 className="text-sm font-semibold text-gray-900 dark:text-slate-100">Рекомендации</h3>
                <span className="text-xs text-gray-400 dark:text-slate-500">({recs.filter((r) => r.discipline_id === selected?.name).length})</span>
                {seedMsg && <span className="text-xs text-gray-500 dark:text-slate-400">{seedMsg}</span>}
                <button
                  onClick={() => setShowAddForm(!showAddForm)}
                  className="ml-auto text-xs bg-violet-600 text-white px-3 py-1.5 rounded-lg hover:bg-violet-700 transition-colors cursor-pointer border-0"
                >
                  {showAddForm ? "Отмена" : "Добавить рекомендацию"}
                </button>
                <button
                  onClick={seedAutoRecs}
                  disabled={seedLoading}
                  title="Заполнить панель топ-рекомендациями из автоанализа (ручные сохранятся)"
                  className="text-xs bg-white dark:bg-slate-950 text-violet-700 dark:text-violet-300 border border-purple-300 dark:border-purple-700 px-3 py-1.5 rounded-lg hover:bg-violet-50 transition-colors cursor-pointer disabled:opacity-50"
                >
                  {seedLoading ? "Заполнение…" : "Заполнить из анализа"}
                </button>
              </div>
              {showAddForm && (
                <div className="mb-4 p-4 bg-gray-50 dark:bg-slate-900 rounded-lg border border-gray-200 dark:border-slate-700">
                  {(recType === "modify" || recType === "remove") && (
                    <select
                      value={selectedCompetency}
                      onChange={(e) => setSelectedCompetency(e.target.value)}
                      className="w-full h-9 px-2 mb-2 text-sm bg-white dark:bg-slate-950 border border-gray-300 dark:border-slate-600 rounded-lg outline-none text-gray-900 dark:text-slate-100"
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
                    className="w-full p-2 text-sm border border-gray-300 dark:border-slate-600 rounded-lg resize-vertical outline-none box-border bg-white dark:bg-slate-900 text-gray-800 dark:text-slate-200"
                  />
                  <div className="flex gap-2 mt-2 items-center">
                    <select
                      value={recType}
                      onChange={(e) => { setRecType(e.target.value); setSelectedCompetency(""); }}
                      className="h-9 px-2 text-sm bg-white dark:bg-slate-950 border border-gray-300 dark:border-slate-600 rounded-lg outline-none text-gray-900 dark:text-slate-100"
                    >
                      <option value="modify">Изменить</option>
                      <option value="add">Добавить</option>
                      <option value="remove">Удалить</option>
                    </select>
                    <button
                      onClick={() => { addRec(); setShowAddForm(false); setSelectedCompetency(""); }}
                      className="h-9 px-4 text-sm bg-violet-600 text-white border-0 rounded-lg cursor-pointer hover:bg-violet-700 transition-colors"
                    >
                      Отправить
                    </button>
                  </div>
                </div>
              )}
              {recs.filter((r) => r.discipline_id === selected?.name).length === 0 ? (
                <div className="text-xs text-gray-400 dark:text-slate-500 bg-gray-50 dark:bg-slate-900 rounded-lg p-4 text-center">
                  No recommendations for this discipline yet
                </div>
              ) : (
                recs
                  .filter((r) => r.discipline_id === selected?.name)
                  .map((r, i) => (
                    <div
                      key={i}
                      className="mb-2 rounded-lg border border-violet-200 bg-violet-50/40 p-3 text-sm dark:border-violet-900 dark:bg-violet-950/20"
                    >
                      <div className="text-gray-400 dark:text-slate-500 mb-1 text-xs">
                        [{r.suggestion_type}] {r.competency_id}
                      </div>
                      <div className="text-gray-900 dark:text-slate-100">{r.suggestion}</div>
                      <button
                        onClick={() => { if (window.confirm("Удалить рекомендацию безвозвратно?")) deleteRec(r.id); }}
                        className="mt-2 text-xs text-red-500 border border-red-500 rounded px-2 py-0.5 hover:bg-red-50 transition-colors bg-transparent cursor-pointer"
                      >
                        Удалить
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
