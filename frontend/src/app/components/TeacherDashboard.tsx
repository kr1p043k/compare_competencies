import { useState, useEffect, useRef, useMemo } from "react";
import { api } from "../api";
import { Badge, Meter, Skeleton, StatCard } from "../../design";
import { AnalysisPanel } from "./AnalysisPanel";
import { MetricsGlossary } from "./MetricsGlossary";
import { TaxonomyBrowser } from "./TaxonomyBrowser";
import CompetencyTrendsPanel from "./CompetencyTrendsPanel";
import { LlmChatPanel } from "./LlmChatPanel";
import {
  competencyFilterCounts,
  disciplineFilterCounts,
  filterCompetencies,
  filterDisciplines,
  filterRecommendations,
  sortDisciplines,
  type CompFilter,
  type DiscFilter,
  type DiscSort,
} from "./teacherFilters";
import { errMessage, reportError } from "./../../lib/errors";
import { ApiError, request } from "../../lib/api";
import { covBorderClass, covTextClass, plural, rpdStageLabel } from "./teacher-dashboard/helpers";
import type {
  Direction,
  DirectionAnalysis,
  Discipline,
  DisciplineDetail,
  Recommendation,
  RpdStatus,
  ScopeChain,
  Stats,
} from "./teacher-dashboard/types";
import { DirectionInsights } from "./teacher-dashboard/DirectionInsights";
import { DisciplineList } from "./teacher-dashboard/DisciplineList";
import { DisciplineRecsSection } from "./teacher-dashboard/DisciplineRecsSection";
import { IntakeComparison } from "./teacher-dashboard/IntakeComparison";
import { ZunSection } from "./teacher-dashboard/ZunSection";

export function TeacherDashboard() {
  const [disciplines, setDisciplines] = useState<Discipline[]>([]);
  const [selected, setSelected] = useState<DisciplineDetail | null>(null);
  const [scopeSaving, setScopeSaving] = useState<string | null>(null);
  // Фильтр видимости («Показать в списке»): скрывает строки уже
  // загруженного списка по году набора, данных не перезагружает.
  const [yearFilter, setYearFilter] = useState<0 | 2024 | 2025>(0);
  // Набор («Набор» в сайдбаре): по его плану ЗАГРУЖАЮТСЯ данные. Это НЕ
  // фильтр видимости: у наборов разные планы и разные KRM, поэтому
  // дисциплины, охваты и анализ запрашиваются заново.
  const [intakeYear, setIntakeYear] = useState<2024 | 2025>(2025);
  // Арифметика «в наборе -> в учёте -> в анализе»: без неё три разных числа
  // дисциплин выглядят как ошибка, а не как разные выборки.
  const [scopeChain, setScopeChain] = useState<ScopeChain | null>(null);
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
    } catch (e) {
      setScopeMsg("Ошибка scope: " + (errMessage(e) || "неизвестная ошибка"));
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
    } catch (e) {
      let msg = errMessage(e) || "неизвестная ошибка";
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
  // Фильтры и сортировка списка дисциплин (поиск — выше).
  const [discFilter, setDiscFilter] = useState<DiscFilter>("all");
  const [discSort, setDiscSort] = useState<DiscSort>("plan");
  // Фильтры шага «Компетенции и ЗУН».
  const [compSearch, setCompSearch] = useState("");
  const [compFilter, setCompFilter] = useState<CompFilter>("all");
  // Фильтры шага «Рекомендации».
  const [recComp, setRecComp] = useState("");
  const [recTypeF, setRecTypeF] = useState("");
  const [loading, setLoading] = useState(true);
  const [directions, setDirections] = useState<Direction[]>([]);
  const [selectedDir, setSelectedDir] = useState("09.03.02");
  const [analysis, setAnalysis] = useState<DirectionAnalysis | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<"coverage" | "trends" | "taxonomy">("coverage");
  const [teacherSection, setTeacherSection] = useState("t-overview");
  // Пошаговый вид дисциплины: покрытие → компетенции и ЗУН → рекомендации.
  const [disciplineView, setDisciplineView] = useState<"coverage" | "zun" | "recs">("coverage");
  // Аккордеон компетенций: развёрнута максимум одна, по умолчанию все свёрнуты.
  const [openComp, setOpenComp] = useState<string | null>(null);
  // Сворачиваемая группа «Данные и РПД» в сайдбаре (раскрывается сама, пока что-то обрабатывается).
  const [dataOpsOpen, setDataOpsOpen] = useState(false);
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
    const ids = ["t-overview", "t-recs", "t-gaps", "t-trends",
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
  const [recError, setRecError] = useState("");
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
    } catch (e) {
      let msg = errMessage(e) || "неизвестная ошибка";
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
  const [rpdStatus, setRpdStatus] = useState<RpdStatus | null>(null);
  const [rpdMsg, setRpdMsg] = useState("");
  const [collectCooldown, setCollectCooldown] = useState(0);
  const runAnalysisBtnRef = useRef<HTMLButtonElement | null>(null);
  // Для поллинга run-analysis: не даём долгому циклу писать в размонтированный
  // компонент или в чужое направление после переключения.
  const mountedRef = useRef(true);
  const selectedDirRef = useRef(selectedDir);
  useEffect(() => () => { mountedRef.current = false; }, []);
  useEffect(() => { selectedDirRef.current = selectedDir; }, [selectedDir]);

  // Кнопка "Запустить анализ" из пустого состояния деталки дисциплины.
  useEffect(() => {
    const handler = () => { runAnalysisBtnRef.current?.click(); };
    window.addEventListener("run-direction-analysis", handler);
    return () => window.removeEventListener("run-direction-analysis", handler);
  }, []);

  useEffect(() => {
    Promise.all([
      api<Recommendation[]>("/teacher/krm/recommendations"),
      api<Direction[]>("/teacher/krm/directions"),
    ]).then(([r, dirs]) => {
      setRecs(r);
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
    // При быстром переключении направлений медленный ответ старого запроса
    // затирал данные нового. Флаг отмены отбрасывает устаревшие ответы.
    let cancelled = false;
    api<Discipline[]>(`/teacher/krm/disciplines?dir_code=${selectedDir}&year=${intakeYear}`)
      .then((d) => { if (!cancelled) setDisciplines(d); })
      .catch(() => { if (!cancelled) setDisciplines([]); });
    api<ScopeChain>(`/teacher/krm/scope-summary?dir_code=${selectedDir}&year=${intakeYear}`)
      .then((s) => { if (!cancelled) setScopeChain(s); })
      .catch(() => { if (!cancelled) setScopeChain(null); });
    api<Stats>(`/teacher/krm/stats?dir_code=${selectedDir}`)
      .then((s) => { if (!cancelled) setStats(s); })
      .catch((err) => { if (!cancelled) reportError("app/components/TeacherDashboard.tsx", err); });
    api<DirectionAnalysis>(`/teacher/analysis?dir_code=${selectedDir}&year=${intakeYear}`)
      .then((a) => { if (!cancelled) setAnalysis(a); })
      .catch(() => { if (!cancelled) setAnalysis(null); });
    api<{ stale?: boolean }>(`/teacher/analysis/meta?dir_code=${selectedDir}`)
      .then((m) => { if (!cancelled) setMetaStale(!!m?.stale); })
      .catch(() => { if (!cancelled) setMetaStale(false); });
    return () => { cancelled = true; };
  }, [selectedDir, intakeYear]);

  useEffect(() => {
    if (collectCooldown <= 0) return;
    const t = setTimeout(() => setCollectCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => clearTimeout(t);
  }, [collectCooldown]);

  useEffect(() => {
    api<{ yandex_covered: string[] }>("/teacher/rpd/sources")
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
    // восстановление после перезагрузки страницы: выполняем один раз.
  }, []);

  useEffect(() => {
    if (!rpdRun?.run_id) return;
    let cancelled = false;
    let misses = 0;
    const poll = () => {
      api<RpdStatus>(`/teacher/rpd/status/${rpdRun.run_id}`)
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
              api<DirectionAnalysis>(`/teacher/analysis?dir_code=${selectedDir}`)
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
    } catch (e) {
      setRpdMsg("Ошибка: " + errMessage(e));
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
          throw new Error("Слишком частые запросы - кнопка заблокирована на 30 секунд", { cause: err });
        }
        throw err;
      }
      if (!data.run_id) throw new Error("Сервер не вернул ID задачи");
      try { sessionStorage.setItem("rpdRunId", data.run_id); } catch (err) { reportError("app/components/TeacherDashboard.tsx:417", err); }
      setRpdRun({ run_id: data.run_id });
      setRpdMsg("Сбор и обработка запущены...");
    } catch (e) {
      setRpdMsg("Ошибка: " + errMessage(e));
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
    } catch (e) {
      setZunMsg("Ошибка: " + (errMessage(e) || "не удалось сохранить"));
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
    } catch (e) {
      setZunMsg("Ошибка: " + (errMessage(e) || "не удалось сохранить"));
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
    } catch (e) {
      setZunMsg("Ошибка: " + (errMessage(e) || "не удалось удалить"));
    } finally {
      setZunSaving(false);
    }
  }

  async function seedAutoRecs() {
    setSeedLoading(true);
    setSeedMsg("");
    try {
      const r = await api<{ seeded: number }>(`/teacher/krm/recommendations/seed/auto?dir_code=${selectedDir}`, { method: "POST" });
      const data = await api<Recommendation[]>("/teacher/krm/recommendations");
      setRecs(data);
      setSeedMsg(`Добавлено ${r.seeded} авторекомендаций (ручные не тронуты).`);
    } catch (e) {
      setSeedMsg("Ошибка: " + (errMessage(e) || "не удалось заполнить"));
    } finally {
      setSeedLoading(false);
    }
  }

  async function loadDiscipline(name: string) {
    try {
      const data = await api<DisciplineDetail>(`/teacher/krm/disciplines/${encodeURIComponent(name)}?dir_code=${selectedDir}&year=${intakeYear}`);
      setSelected(data);
      setShowAnalysis(false);
      // Каждая дисциплина начинается с первого шага, свёрнутого аккордеона
      // и чистых фильтров.
      setDisciplineView("coverage");
      setOpenComp(null);
      setCompSearch("");
      setCompFilter("all");
      setRecComp("");
      setRecTypeF("");
    } catch (err) { reportError("app/components/TeacherDashboard.tsx:500", err); }
  }

  async function addRec(): Promise<boolean> {
    if (!selected || !suggestion.trim()) return false;
    try {
      const compId = recType === "add" ? "" : selectedCompetency;
      const resp = await api<{ id: number }>("/teacher/krm/recommendations", {
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
      return true;
    } catch (err) {
      // Не молчим: форма закрывалась сразу после клика, и сбой сохранения
      // был невидим пользователю.
      reportError("app/components/TeacherDashboard.tsx:527", err);
      setRecError(err instanceof Error ? err.message : String(err));
      return false;
    }
  }

  async function deleteRec(id: number) {
    try {
      await api(`/teacher/krm/recommendations/${id}`, { method: "DELETE" });
      setRecs((prev) => prev.filter((r) => r.id !== id));
    } catch (err) { reportError("app/components/TeacherDashboard.tsx:534", err); }
  }

  // Фильтрация/сортировка списков: useMemo обязательно ДО раннего return
  // скелетона ниже (правила хуков) и ради производительности — раньше каждый
  // чекбокс/символ поиска пересчитывал все списки заново на каждом рендере.
  const analysisByName = useMemo(
    () => new Map((analysis?.disciplines ?? []).map((a) => [a.name, a])),
    [analysis],
  );
  // Поиск + чипы-фильтры задают состав списка, сортировка — только порядок.
  const filtered = useMemo(
    () =>
      sortDisciplines(
        filterDisciplines(disciplines, analysisByName, { search, year: yearFilter, filter: discFilter }),
        analysisByName,
        discSort,
      ),
    [disciplines, analysisByName, search, yearFilter, discFilter, discSort],
  );
  const discCounts = useMemo(
    () => disciplineFilterCounts(disciplines, analysisByName, { search, year: yearFilter }),
    [disciplines, analysisByName, search, yearFilter],
  );
  const discFiltersActive =
    !!search.trim() || yearFilter !== 0 || discFilter !== "all" || discSort !== "plan";
  // Шаг «Компетенции и ЗУН»: поиск + наличие ЗУН.
  const compsVisible = useMemo(
    () => (selected ? filterCompetencies(selected.competencies, { search: compSearch, filter: compFilter }) : []),
    [selected, compSearch, compFilter],
  );
  const compCounts = useMemo(
    () =>
      selected
        ? competencyFilterCounts(filterCompetencies(selected.competencies, { search: compSearch }))
        : null,
    [selected, compSearch],
  );
  const compFiltersActive = !!compSearch.trim() || compFilter !== "all";
  // Шаг «Рекомендации»: по компетенции и по типу.
  const discRecCount = useMemo(
    () => (selected ? recs.filter((r) => r.discipline_id === selected.name).length : 0),
    [recs, selected],
  );
  const recsVisible = useMemo(
    () => (selected ? filterRecommendations(recs, selected.name, { comp: recComp, type: recTypeF }) : []),
    [recs, selected, recComp, recTypeF],
  );
  const recFiltersActive = !!recComp || !!recTypeF;

  if (loading) {
    return (
      <div
        role="status"
        aria-live="polite"
        className="flex h-[calc(100vh-100px)] flex-col overflow-hidden rounded-2xl border border-border bg-card font-sans md:flex-row"
      >
        <span className="sr-only">Загрузка...</span>
        <div className="w-full shrink-0 space-y-2.5 overflow-hidden border-b border-border p-4 md:w-[380px] md:min-w-[380px] md:border-b-0 md:border-r">
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-28" />
            <Skeleton className="h-8 w-24" />
          </div>
          <Skeleton className="h-7 w-full" />
          <Skeleton className="h-8 w-full" />
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={`disc-skeleton-${i}`} className="h-10 w-full" />
          ))}
        </div>
        <div className="flex-1 space-y-3 overflow-hidden p-6">
          <Skeleton className="h-6 w-64" />
          {Array.from({ length: 5 }, (_, i) => (
            <Skeleton key={`card-skeleton-${i}`} className="h-16 w-full" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-100px)] font-sans text-foreground bg-card rounded-2xl border border-border overflow-hidden">
      <div className="w-[380px] min-w-[380px] border-r border-border flex flex-col overflow-hidden">
          <div className="p-4 border-b border-border">
            <div className="flex justify-between items-center">
              <h1 className="text-lg m-0 text-foreground">
                KRM Teacher
              </h1>
              {stats && (
                  <div className="text-xs text-muted-foreground text-right">
                    <div>{(() => { const n = stats.total_disciplines ?? stats.total_reports ?? 0; return `${n} ${plural(n, "дисциплина", "дисциплины", "дисциплин")}`; })()}</div>
                    <div>{(() => { const n = stats.total_competencies ?? Object.keys(stats.by_profession ?? {}).length; return `${n} ${plural(n, "компетенция", "компетенции", "компетенций")}`; })()}</div>
                    <div>{(() => { const n = stats.total_skills ?? 0; return `${n.toLocaleString()} ${plural(n, "навык", "навыка", "навыков")}`; })()}</div>
                  </div>
                )}
          </div>

          {/* Direction selector */}
          <select
            value={selectedDir}
            onChange={(e) => setSelectedDir(e.target.value)}
            aria-label="Направление подготовки"
        className="w-full mt-2.5 px-2 py-1.5 rounded-md text-xs bg-card border border-border text-foreground outline-none"
          >
            {directions.map((d, idx) => (
              <option key={`${d.dir_code}-${idx}`} value={d.dir_code}>{d.dir_code} - {d.name}</option>
            ))}
          </select>

          {/* Группа «Данные и РПД»: свёрнута по умолчанию, раскрывается сама,
              пока идёт анализ или загрузка РПД (чтобы прогресс был виден). */}
          {(() => {
            const processing =
              runLoading ||
              rpdUploading ||
              rpdCollecting ||
              rpdStatus?.status === "running" ||
              rpdStatus?.status === "started";
            const opsOpen = dataOpsOpen || processing;
            return (
              <div className="mt-2.5 rounded-md border border-border">
                <button
                  onClick={() => setDataOpsOpen(!dataOpsOpen)}
                  aria-expanded={opsOpen}
                  className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-foreground bg-muted border-0 rounded-md cursor-pointer"
                >
                  <span>Данные и РПД</span>
                  <span aria-hidden="true">{opsOpen ? "▾" : "▸"}</span>
                </button>
                {opsOpen && (
                  <div className="px-3 pb-3 pt-1">
          {/* Run analysis button */}
          <button
            ref={runAnalysisBtnRef}
            onClick={async () => {
              if (runLoading) return;
              setRunLoading(true);
              setRunMsg("Запуск анализа...");
              const dirAtStart = selectedDir;
              try {
                const started = await api<{ run_id?: string }>(`/teacher/krm/run-analysis?dir_code=${selectedDir}`, { method: "POST" });
                const runId = started?.run_id;
                const t0 = Date.now();
                if (!runId) throw new Error("Сервер не вернул ID задачи");
                for (let i = 0; i < 360; i++) {
                  await new Promise((res) => setTimeout(res, 5000));
                  // Компонент размонтирован или направление сменили - выходим,
                  // не затирая чужое состояние.
                  if (!mountedRef.current || selectedDirRef.current !== dirAtStart) return;
                  const spent = Math.round((Date.now() - t0) / 1000);
                  let s: RpdStatus | null = null;
                  try {
                    s = await api<RpdStatus>(`/teacher/rpd/status/${runId}`);
                  } catch {
                    setRunMsg(`Этап: пересчёт анализа… нет связи (${spent} c)`);
                    continue;
                  }
                  if (!s) continue;
                  if (s.status === "completed") {
                    const a = await api<DirectionAnalysis>(`/teacher/analysis?dir_code=${selectedDir}`).catch(() => null);
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
              } catch (e) {
                setRunMsg(`Ошибка запуска: ${errMessage(e) || String(e)}`);
              } finally {
                setRunLoading(false);
              }
            }}
            disabled={runLoading}
        className="w-full mt-2.5 px-3 py-2 border-0 rounded-md text-xs font-semibold text-primary-foreground bg-primary hover:brightness-110 cursor-pointer disabled:opacity-50 disabled:pointer-events-none"
          >
            {runLoading ? "Анализ запущен..." : "Запустить анализ"}
          </button>
            {runMsg && (
              <div className="text-xs text-warning mt-1.5">{runMsg}</div>
            )}

          {/* RPD upload block */}
          <div
        className="mt-3 px-3 py-2.5 rounded-md bg-warning-soft border border-warning/30"
          >
            <div className="text-xs font-bold text-warning mb-2">
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
          className="w-full px-3 py-2 border-0 rounded-md text-xs font-semibold text-primary-foreground bg-primary hover:brightness-110 cursor-pointer disabled:opacity-50 disabled:pointer-events-none"
            >
              {rpdUploading ? "Обработка..." : "Загрузить PDF и обработать"}
            </button>
            {rpdSources.yandex_covered?.includes(selectedDir) && (
              <button
                onClick={collectRpd}
                disabled={collectCooldown > 0}
            className={`w-full mt-2 px-3 py-2 border-0 rounded-md text-xs font-semibold cursor-pointer ${
              rpdCollecting
                ? "bg-destructive text-destructive-foreground"
                : "bg-primary text-primary-foreground disabled:opacity-50 disabled:pointer-events-none"
            }`}
              >
                {rpdCollecting ? "Отмена" : collectCooldown > 0 ? `Подождите ${collectCooldown} сек` : "Собрать с Yandex Disk"}
              </button>
            )}
            <div className="flex flex-col gap-1.5 border-t border-dashed border-warning/30 pt-2 mt-2">
              <input
                type="text"
                value={rpdYandexUrl}
                onChange={(e) => setRpdYandexUrl(e.target.value)}
                placeholder="Ссылка на папку Yandex Disk (https://disk.360.yandex.ru/d/...)"
            className="w-full px-2.5 py-2 rounded-md text-xs box-border border border-warning/40 bg-card text-foreground"
              />
              <button
                onClick={collectRpd}
                disabled={rpdCollecting || collectCooldown > 0 || !rpdYandexUrl.trim()}
            className="w-full px-3 py-2 border-0 rounded-md text-xs font-semibold text-success-foreground bg-success hover:brightness-110 cursor-pointer disabled:opacity-50 disabled:pointer-events-none"
              >
                {rpdCollecting ? "Сбор..." : "Загрузить по ссылке"}
              </button>
            </div>
            {rpdStatus && (rpdStatus.status === "completed" || rpdStatus.status === "failed") && (
              <div
            className={`mt-2 text-xs ${rpdStatus.status === "completed" ? "text-success" : "text-destructive"}`}
              >
                {rpdMsg}
                {rpdStatus.stats?.disciplines != null && rpdStatus.status === "completed" && (
                  <span> - {rpdStatus.stats.disciplines} дисциплин</span>
                )}
              </div>
            )}
            {(rpdUploading || rpdCollecting) && rpdStatus && (rpdStatus.status === "running" || rpdStatus.status === "started") && (
              <div className="mt-2.5">
                <Meter
                  value={rpdStatus.stats?.progress ?? 10}
                  trackClassName="bg-warning/30"
                  fillClassName="bg-info"
                />
                <div className="mt-1.5 text-xs text-warning">
                  Этап: {rpdStageLabel(rpdStatus.stats?.stage)} · {rpdStatus.stats?.progress ?? 10}%
                </div>
              </div>
            )}
          </div>
                  </div>
                )}
              </div>
            );
          })()}

          {/* Analysis summary button */}
          {analysis && (() => {
            const cov = analysis.average_market_top_coverage ?? analysis.average_coverage;
            // Источник рынка и достоверность: процент без них читается как
            // точный, хотя derived/relevance - это вычисленный знаменатель.
            const src: Record<string, number> = {};
            let unreliable = 0;
            for (const d of analysis.disciplines ?? []) {
              const s = d.market_top_source || "n/a";
              src[s] = (src[s] || 0) + 1;
              if (d.market_top_reliable === false) unreliable += 1;
            }
            const srcTitle =
              "источник рынка - " +
              Object.entries(src)
                .sort((a, b) => b[1] - a[1])
                .map(([k, n]) => `${k}: ${n}`)
                .join(", ") +
              (unreliable > 0 ? ` · недостоверно (мало навыков): ${unreliable}` : "");
            return (
            <div
              onClick={() => setShowAnalysis(!showAnalysis)}
              title={srcTitle}
          className={`mt-2.5 px-3 py-2 rounded-md cursor-pointer text-xs bg-muted border ${covBorderClass(analysis.average_coverage)}`}
            >
              <div className="flex justify-between items-center gap-2">
                {/* Показываем охват спроса, а не coverage_ratio: второй почти
                    всегда ~100%, потому что навыки РПД уже нормализованы к
                    словарю рынка, и читается как «всё покрыто». */}
                <span className="text-primary font-semibold">Охват рынка</span>
                <span className={`font-bold ${covTextClass(cov)}`}>
                  {(cov * 100).toFixed(1)}%
                </span>
              </div>
              {/* Качественная пара к охвату: без него один процент выглядит
                  итоговой оценкой, хотя это лишь строгая доля. */}
              {analysis.average_market_weight_coverage != null && (
                <div
                  className="text-[11px] text-subtle-foreground mt-1"
                  title="Взвешенный охват: те же топ-навыки рынка, но с весом спроса по каждому навыку (см. глоссарий)"
                >
                  по спросу: {(analysis.average_market_weight_coverage * 100).toFixed(1)}%
                </div>
              )}
              {/* Качество рядом с охватом (P1-5): охват отвечает «сколько навыков
                  рынка закрыто», качество - «насколько уверенно они сопоставлены».
                  Пайплайн считает его как average_quality_coverage; без вывода в UI
                  цифра жила только в summary. */}
              {analysis.average_quality_coverage != null && (
                <div
                  className="text-[11px] text-subtle-foreground mt-1"
                  title="Качество: средняя уверенность сопоставления навыков РПД с рынком. Навык может совпасть неточно, поэтому качество обычно ниже строгого охвата (см. глоссарий)"
                >
                  качество: {(analysis.average_quality_coverage * 100).toFixed(1)}%
                </div>
              )}
              <div className="text-subtle-foreground mt-0.5">
                {analysis.total_disciplines} {plural(analysis.total_disciplines, "дисциплина", "дисциплины", "дисциплин")} · {analysis.total_gaps_across_all} {plural(analysis.total_gaps_across_all, "пробел", "пробела", "пробелов")}
              </div>
              <div className="mt-1.5">
                <MetricsGlossary compact />
              </div>
            </div>
            );
          })()}

          <input
            placeholder="Поиск дисциплин..."
            aria-label="Поиск дисциплины"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
        className="w-full mt-2.5 px-3 py-2 rounded-md text-[13px] box-border bg-card border border-border text-foreground outline-none"
          />
          {/* Набор: переключает набор, по которому ЗАГРУЖАЮТСЯ данные
              (см. useEffect на [selectedDir, intakeYear]). Это НЕ фильтр
              видимости: строки ниже фильтрует «Показать в списке». */}
          <div className="flex gap-1.5 mt-2 items-center flex-wrap" role="group" aria-label="Набор: загрузка данных">
            <span
              className="text-xs text-muted-foreground"
              title="Учебный план какого набора грузить: от него зависят список дисциплин, охваты и весь анализ. Переключение заново запрашивает данные с сервера."
            >
              Набор:
            </span>
            {([2025, 2024] as const).map((y) => (
              <button
                key={y}
                onClick={() => setIntakeYear(y)}
                aria-pressed={intakeYear === y}
                title={`Загрузить список дисциплин и анализ по учебному плану набора ${y}`}
                className={
                  "text-xs px-2 py-0.5 rounded border cursor-pointer " +
                  (intakeYear === y
                    ? "border-primary/60 bg-primary-soft text-primary-soft-foreground"
                    : "border-border-strong bg-card text-muted-foreground")
                }
              >
                {y}
              </button>
            ))}
          </div>
          {/* Видимая подсказка: то, что раньше было только в тултипах,
              пользователь не читал и путал набор с фильтром. */}
          <div className="text-[11px] text-subtle-foreground mt-1">
            по плану выбранного набора грузятся список, охваты и анализ
          </div>
          {/* Что изменилось при переходе 2024 -> 2025: план и среднее покрытие.
              Карточка живёт рядом с переключателем набора, потому что отвечает
              на вопрос «стоит ли переключаться» без перезагрузки списка. */}
          <div className="mt-2.5">
            <IntakeComparison dirCode={selectedDir} />
          </div>
        </div>
        <div className="flex-1 overflow-auto">
          {/* Чипы вместо текстовой цепочки с точками: каждый шаг
              снабжён тултипом, откуда взялось число. */}
          <div className="px-4 py-1.5 border-b border-border">
            {scopeChain ? (
              <div className="flex gap-1.5 flex-wrap items-center">
                <Badge tone="neutral" title="Все дисциплины учебного плана выбранного набора">
                  в наборе {scopeChain.total}
                </Badge>
                <Badge tone="primary" title="Отмечены галочками «Учитывать в анализе»">
                  в учёте {scopeChain.in_scope}
                </Badge>
                <Badge tone="success" title="Реально участвуют в анализе: в учёте и хватает данных">
                  в анализе {scopeChain.analyzed ?? "-"}
                </Badge>
                <Badge tone="warning" title="Исключены скриптом методики; галочку можно вернуть">
                  исключено методикой {scopeChain.excluded_manual}
                </Badge>
                <Badge tone="neutral" title="Не участвуют в анализе: в плане нет компетенций, навыков или их связей">
                  без данных {scopeChain.excluded_no_data}
                </Badge>
              </div>
            ) : (
              <div className="text-xs text-muted-foreground">
                В учёте {disciplines.filter((x) => x.in_scope ?? true).length} из {disciplines.length}
              </div>
            )}
            {scopeMsg && (<div className="text-warning mt-0.5">{scopeMsg}</div>)}
            <div className="flex gap-1.5 mt-1 items-center flex-wrap">
              <button onClick={() => saveScopeBulk(true)} disabled={scopeSaving !== null}
                className="text-xs px-2.5 py-0.5 rounded border border-border-strong bg-card cursor-pointer">
                Выбрать все
              </button>
              <button onClick={() => saveScopeBulk(false)} disabled={scopeSaving !== null}
                className="text-xs px-2.5 py-0.5 rounded border border-border-strong bg-card cursor-pointer">
                Снять все
              </button>
              {metaStale && (<span className="text-xs text-warning bg-warning-soft rounded px-1.5 py-0.5">данные устарели - обновите анализ</span>)}
            </div>
          </div>
          {/* «Показать в списке» - фильтр видимости: только скрывает строки
              уже загруженного списка, данных не перезагружает (в отличие от
              выбора набора выше). */}
          <DisciplineList
            filtered={filtered}
            disciplines={disciplines}
            analysisByName={analysisByName}
            selected={selected}
            setSearch={setSearch}
            yearFilter={yearFilter}
            setYearFilter={setYearFilter}
            discFilter={discFilter}
            setDiscFilter={setDiscFilter}
            discSort={discSort}
            setDiscSort={setDiscSort}
            discCounts={discCounts}
            discFiltersActive={discFiltersActive}
            scopeSaving={scopeSaving}
            loadDiscipline={loadDiscipline}
            saveScope={saveScope}
          />
        </div>
      </div>

      <div className="flex-1 overflow-auto p-6" id="teacher-main">
        {!selected && !showAnalysis && (
          <div className="text-center mt-20 text-subtle-foreground text-sm">
            Выберите дисциплину или откройте анализ
          </div>
        )}

        {/* Переключатель вида. Для выбранной дисциплины — пошаговые разделы
            (покрытие → компетенции и ЗУН → рекомендации), для направления —
            режимы анализа. Пока открыты тренды/таксономия, показываем режимы:
            иначе нельзя было бы вернуться к покрытию дисциплины. */}
        {selected && analysisMode === "coverage" ? (
          <div className="flex gap-1.5 mb-4 flex-wrap" role="tablist" aria-label="Разделы дисциплины">
            {([
              ["coverage", "Покрытие"],
              ["zun", `Компетенции и ЗУН (${selected.competencies.length})`],
              ["recs", `Рекомендации (${recs.filter((r) => r.discipline_id === selected.name).length})`],
            ] as const).map(([id, label]) => (
              <button
                key={id}
                role="tab"
                aria-selected={disciplineView === id}
                onClick={() => setDisciplineView(id)}
        className={`px-3.5 py-1.5 border-0 rounded-md cursor-pointer text-xs ${disciplineView === id ? "bg-primary text-primary-foreground font-bold" : "bg-muted text-primary font-normal"}`}
              >
                {label}
              </button>
            ))}
          </div>
        ) : (
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setAnalysisMode("coverage")}
            disabled={!analysis}
            title={analysis ? undefined : "Появится после запуска анализа направления"}
        className={`px-3.5 py-1.5 border-0 rounded-md cursor-pointer text-xs disabled:cursor-not-allowed disabled:opacity-50 ${analysisMode === "coverage" ? "bg-primary text-primary-foreground font-bold" : "bg-muted text-primary font-normal"}`}
          >
            Покрытие
          </button>
            <button
              onClick={() => setAnalysisMode("trends")}
          className={`px-3.5 py-1.5 border-0 rounded-md cursor-pointer text-xs ${analysisMode === "trends" ? "bg-primary text-primary-foreground font-bold" : "bg-muted text-primary font-normal"}`}
            >
              Тренды компетенций
            </button>
            <button
              onClick={() => setAnalysisMode("taxonomy")}
          className={`px-3.5 py-1.5 border-0 rounded-md cursor-pointer text-xs ${analysisMode === "taxonomy" ? "bg-primary text-primary-foreground font-bold" : "bg-muted text-primary font-normal"}`}
            >
              Таксономия
            </button>
        </div>
        )}

        {analysisMode === "coverage" && analysis && (!selected || disciplineView === "coverage") && (<>
          {/* Суб-навигация: при выбранной дисциплине - по её разделам, иначе по сводке */}
          <div className="inline-flex items-center gap-1 rounded-lg bg-muted dark:bg-card p-1 mb-4 sticky top-0 z-10 flex-wrap">
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
                  // якорь t-gaps вернулся вместе с разделом «Разрывы между
                  // планом и рынком», так что причина удаления чипа
                  // (f02495ac) больше не актуальна
                  ["t-gaps", "Разрывы"],
                  ["t-trends", "Тренды"],
                ]
            ).map(([id, label]) => (
              <button
                key={id}
                onClick={() => scrollTeacherTo(id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all cursor-pointer ${
                  teacherSection === id
                      ? "bg-card text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <div id="t-overview" className="text-xs text-muted-foreground mb-2 scroll-mt-2">
            {selected
              ? `Дисциплина: ${selected.name} (направление ${selectedDir})`
              : `Направление ${selectedDir} - сводка по всем ${analysis.total_disciplines} дисциплинам, ни одна дисциплина не выбрана`}
          </div>
          {/* When a discipline is selected, show its coverage instead of direction average */}
          {(() => {
            const discData = selected && analysis.disciplines
              ? analysis.disciplines.find((d) => d.name === selected.name)
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
            // Парная метрика качества: взвешенный по спросу охват. Берём
            // только со своего уровня (дисциплина/направление), не смешиваем.
            const wCov = discData ? discData.market_weight_coverage : analysis.average_market_weight_coverage;
          return (<>
          <div className="flex gap-3 mb-4 flex-wrap">
            <StatCard
              className="min-w-[200px]"
              tone={!covReliable ? "default" : cov > 0.5 ? "success" : cov > 0.2 ? "warning" : "danger"}
              label={<>{discData ? `Охват рынка: ${selected?.name ?? ""}` : "Средний охват рынка"}{covTotal ? ` (из ${covTotal})` : ""}</>}
              value={covReliable ? `${(cov * 100).toFixed(1)}%` : "н/д"}
              hint={<>
                <span className="font-semibold">
                  {covReliable ? (level || "").toUpperCase() : "мало релевантных навыков"}
                </span>
                {/* Пара к строгому охвату сверху: без неё один процент
                читается как полная оценка. Только при достоверном знаменателе. */}
                {covReliable && wCov != null && (
                  <span
                    className="block text-subtle-foreground mt-0.5"
                    title="Взвешенный охват (по спросу): насколько востребовано то из топа рынка, что реально покрыто. Может быть выше или ниже строгого"
                  >
                    взвешенный охват: {(wCov * 100).toFixed(1)}%
                  </span>
                )}
                {/* Откуда знаменатель: ручная привязка, автоподбор по навыкам
                    дисциплины или слабая оценка по названию. */}
                {discData?.market_top_source && (
                  <span className="block text-subtle-foreground mt-0.5">
                    {discData.market_top_source === "explicit"
                      ? "знаменатель: ручная привязка"
                      : discData.market_top_source === "derived"
                        ? "знаменатель: авто по навыкам"
                        : "знаменатель: по профилю дисциплины"}
                  </span>
                )}
              </>}
            />
            <StatCard label="Gaps" value={gaps} tone="danger" className="min-w-[140px]" />
            <StatCard label="Disciplines" value={analysis.total_disciplines} tone="primary" className="min-w-[140px]" />
          </div>
          </>)})()}

              {/* Общие блоки направления - только пока дисциплина не выбрана.
                  При выбранной дисциплине её детали ниже (AnalysisPanel). */}
              {!selected && (<>
              <DirectionInsights analysis={analysis} />
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

        {/* Discipline detail: пошаговый поток */}
        {selected && (
          <>
            {/* Хлебные крошки: направление → дисциплина */}
            <div className="flex items-center gap-2 mb-4 flex-wrap">
              <button
                onClick={() => { setSelected(null); setShowAnalysis(true); setAnalysisMode("coverage"); }}
                className="text-xs text-primary hover:text-primary/80 border border-border rounded-md px-2.5 py-1 bg-muted cursor-pointer"
                title="Вернуться к сводке направления"
              >
                ← Направление {selectedDir}
              </button>
              <span aria-hidden="true" className="text-subtle-foreground">/</span>
              <h2 className="text-xl m-0 text-foreground">
                {selected.name}
              </h2>
            </div>

            {/* Шаг 1: покрытие (готовый анализ дисциплины) */}
            {analysisMode === "coverage" && disciplineView === "coverage" && (
              /* Analysis panel for this discipline */
              <AnalysisPanel disciplineName={selected.name} dirCode={selectedDir} />
            )}

            {/* Шаг 2: компетенции и их ЗУН */}
            {analysisMode === "coverage" && disciplineView === "zun" && (<>
            <ZunSection
              selected={selected}
              compForm={compForm}
              setCompForm={setCompForm}
              compErr={compErr}
              setCompErr={setCompErr}
              compCode={compCode}
              setCompCode={setCompCode}
              compName={compName}
              setCompName={setCompName}
              compSaving={compSaving}
              saveCompetency={saveCompetency}
              compSearch={compSearch}
              setCompSearch={setCompSearch}
              compFilter={compFilter}
              setCompFilter={setCompFilter}
              compCounts={compCounts}
              compsVisible={compsVisible}
              compFiltersActive={compFiltersActive}
              openComp={openComp}
              setOpenComp={setOpenComp}
              zunForm={zunForm}
              setZunForm={setZunForm}
              zunMsg={zunMsg}
              setZunMsg={setZunMsg}
              zunSaving={zunSaving}
              addZunEntry={addZunEntry}
              ksaEditing={ksaEditing}
              setKsaEditing={setKsaEditing}
              saveKsaEdit={saveKsaEdit}
              delKsa={delKsa}
            />
            </>)}

            {/* Шаг 3: рекомендации по дисциплине */}
            {analysisMode === "coverage" && disciplineView === "recs" && (
            <DisciplineRecsSection
              selected={selected}
              recs={recs}
              discRecCount={discRecCount}
              recsVisible={recsVisible}
              recFiltersActive={recFiltersActive}
              recComp={recComp}
              setRecComp={setRecComp}
              recTypeF={recTypeF}
              setRecTypeF={setRecTypeF}
              showAddForm={showAddForm}
              setShowAddForm={setShowAddForm}
              recType={recType}
              setRecType={setRecType}
              selectedCompetency={selectedCompetency}
              setSelectedCompetency={setSelectedCompetency}
              suggestion={suggestion}
              setSuggestion={setSuggestion}
              recError={recError}
              setRecError={setRecError}
              seedMsg={seedMsg}
              seedLoading={seedLoading}
              seedAutoRecs={seedAutoRecs}
              addRec={addRec}
              deleteRec={deleteRec}
            />
            )}
          </>
        )}
      </div>
      {/* Ключ контекста чата - КОД направления (selectedDir, напр. 09.03.02):
          бэкенд llm_chat строит контекст из data/result/teacher/<код>/_summary.json
          и krm_disciplines_<код>.json, человеческое имя давало пустой контекст.
          Подпись в шапке (contextLabel) остаётся человекочитаемой. */}
      <LlmChatPanel
        role="teacher"
        discipline={selectedDir}
        contextLabel={selected?.name ?? null}
      />
    </div>
  );
}
