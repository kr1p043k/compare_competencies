import { useState, useEffect, useRef, useMemo } from "react";
import { api } from "../api";
import { Badge, Button, Skeleton } from "../../design";
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
import { plural } from "./teacher-dashboard/helpers";
import type {
  Direction,
  DirectionAnalysis,
  Discipline,
  DisciplineDetail,
  Recommendation,
  ScopeChain,
} from "./teacher-dashboard/types";
import { TeacherHeader } from "./teacher-dashboard/TeacherHeader";
import { DisciplineList } from "./teacher-dashboard/DisciplineList";
import { DirectionInsights } from "./teacher-dashboard/DirectionInsights";
import { DirectionKpis } from "./teacher-dashboard/DirectionKpis";
import { DisciplineRecsSection } from "./teacher-dashboard/DisciplineRecsSection";
import { DataOpsDrawer } from "./teacher-dashboard/DataOpsDrawer";
import { useRpdProcessing } from "./teacher-dashboard/useRpdProcessing";
import { ZunSection } from "./teacher-dashboard/ZunSection";

/**
 * Экран преподавателя: шапка с контекстом, список дисциплин слева и рабочая
 * область справа. Служебные операции (состав анализа, пересчёт, РПД) вынесены
 * в дровер «Данные и РПД», чтобы не занимать место над списком.
 */
export function TeacherDashboard() {
  const [disciplines, setDisciplines] = useState<Discipline[]>([]);
  const [selected, setSelected] = useState<DisciplineDetail | null>(null);
  const [scopeSaving, setScopeSaving] = useState<string | null>(null);
  // Фильтр видимости («Показать в списке»): скрывает строки уже
  // загруженного списка по году набора, данных не перезагружает.
  const [yearFilter, setYearFilter] = useState<0 | 2024 | 2025>(0);
  // Набор («План» в шапке): по его плану ЗАГРУЖАЮТСЯ данные. Это НЕ фильтр
  // видимости: у наборов разные планы и разные KRM, поэтому дисциплины,
  // охваты и анализ запрашиваются заново.
  const [intakeYear, setIntakeYear] = useState<2024 | 2025>(2025);
  // Арифметика «в наборе -> в учёте -> в анализе»: без неё три разных числа
  // дисциплин выглядят как ошибка, а не как разные выборки.
  const [scopeChain, setScopeChain] = useState<ScopeChain | null>(null);
  const [metaStale, setMetaStale] = useState(false);
  const [scopeMsg, setScopeMsg] = useState("");
  const [dataOpsOpen, setDataOpsOpen] = useState(false);

  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [suggestion, setSuggestion] = useState("");
  const [recType, setRecType] = useState("modify");
  const [search, setSearch] = useState("");
  // Фильтры и сортировка списка дисциплин (поиск — в DisciplineList).
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
  const [analysisMode, setAnalysisMode] = useState<"coverage" | "trends" | "taxonomy">("coverage");
  // Пошаговый вид дисциплины: покрытие → компетенции и ЗУН → рекомендации.
  const [disciplineView, setDisciplineView] = useState<"coverage" | "zun" | "recs">("coverage");
  // Аккордеон компетенций: развёрнута максимум одна, по умолчанию все свёрнуты.
  const [openComp, setOpenComp] = useState<string | null>(null);

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
      setCompForm(false);
      setCompCode("");
      setCompName("");
      setCompErr({});
      await loadDiscipline(selected.name);
    } catch (e) {
      let msg = errMessage(e) || "неизвестная ошибка";
      try {
        const j = JSON.parse(msg);
        msg = j.detail || msg;
      } catch (err) {
        reportError("app/components/TeacherDashboard.tsx", err);
      }
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

  // Для поллинга run-analysis: не даём долгому циклу писать в размонтированный
  // компонент или в чужое направление после переключения.
  const mountedRef = useRef(true);
  const selectedDirRef = useRef(selectedDir);
  const intakeYearRef = useRef(intakeYear);
  useEffect(() => () => {
    mountedRef.current = false;
  }, []);
  useEffect(() => {
    selectedDirRef.current = selectedDir;
  }, [selectedDir]);
  useEffect(() => {
    intakeYearRef.current = intakeYear;
  }, [intakeYear]);

  const refreshAnalysis = () => {
    api<DirectionAnalysis>(`/teacher/analysis?dir_code=${selectedDirRef.current}&year=${intakeYearRef.current}`)
      .then(setAnalysis)
      .catch(() => setAnalysis(null));
  };

  const rpd = useRpdProcessing(selectedDir, refreshAnalysis);

  // Кнопка «Запустить анализ» из пустого состояния AnalysisPanel (window-событие).
  // Держим актуальный обработчик в ref, чтобы слушатель, повешенный один раз,
  // видел текущее направление.
  const runAnalysisRef = useRef<() => void>(() => {});
  useEffect(() => {
    const handler = () => runAnalysisRef.current();
    window.addEventListener("run-direction-analysis", handler);
    return () => window.removeEventListener("run-direction-analysis", handler);
  }, []);

  useEffect(() => {
    Promise.all([
      api<Recommendation[]>("/teacher/krm/recommendations"),
      api<Direction[]>("/teacher/krm/directions"),
    ])
      .then(([r, dirs]) => {
        setRecs(r);
        const list = (Array.isArray(dirs) ? dirs : [dirs]) as Direction[];
        setDirections(list);
        if (list.length > 0 && !list.some((d) => d.dir_code === selectedDir)) {
          setSelectedDir(list[0].dir_code);
        }
        setLoading(false);
      })
      .catch((e) => {
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
      .then((d) => {
        if (!cancelled) setDisciplines(d);
      })
      .catch(() => {
        if (!cancelled) setDisciplines([]);
      });
    api<ScopeChain>(`/teacher/krm/scope-summary?dir_code=${selectedDir}&year=${intakeYear}`)
      .then((s) => {
        if (!cancelled) setScopeChain(s);
      })
      .catch(() => {
        if (!cancelled) setScopeChain(null);
      });
    api<DirectionAnalysis>(`/teacher/analysis?dir_code=${selectedDir}&year=${intakeYear}`)
      .then((a) => {
        if (!cancelled) setAnalysis(a);
      })
      .catch(() => {
        if (!cancelled) setAnalysis(null);
      });
    api<{ stale?: boolean }>(`/teacher/analysis/meta?dir_code=${selectedDir}`)
      .then((m) => {
        if (!cancelled) setMetaStale(!!m?.stale);
      })
      .catch(() => {
        if (!cancelled) setMetaStale(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedDir, intakeYear]);

  async function saveScopeBulk(included: boolean) {
    setScopeSaving("__all__");
    setScopeMsg("");
    try {
      const names = disciplines.map((d) => d.name);
      await api(`/teacher/zun/scope`, {
        method: "PUT",
        body: JSON.stringify({ dir_code: selectedDir, changes: names.map((n) => ({ discipline_name: n, included })) }),
      });
      setDisciplines((prev) => prev.map((d) => ({ ...d, in_scope: included, scope_source: "custom" })));
      setScopeMsg("Состав обновлён для всех. Перезапустите анализ, чтобы средние пересчитались.");
    } catch (e) {
      setScopeMsg("Ошибка состава: " + (errMessage(e) || "неизвестная ошибка"));
    } finally {
      setScopeSaving(null);
    }
  }

  async function saveScope(name: string, included: boolean) {
    setScopeSaving(name);
    setScopeMsg("");
    try {
      await api(`/teacher/zun/scope`, {
        method: "PUT",
        body: JSON.stringify({ dir_code: selectedDir, changes: [{ discipline_name: name, included }] }),
      });
      setDisciplines((prev) =>
        prev.map((d) => (d.name === name ? { ...d, in_scope: included, scope_source: "custom" } : d)),
      );
      setScopeMsg("Состав обновлён. Перезапустите анализ, чтобы средние пересчитались.");
    } catch (e) {
      let msg = errMessage(e) || "неизвестная ошибка";
      try {
        const j = JSON.parse(msg);
        msg = j.detail || msg;
      } catch (err) {
        reportError("app/components/TeacherDashboard.tsx", err);
      }
      setScopeMsg("Ошибка состава: " + msg);
    } finally {
      setScopeSaving(null);
    }
  }

  async function runDirectionAnalysis() {
    if (runLoading) return;
    setRunLoading(true);
    setRunMsg("Запуск анализа...");
    const dir = selectedDirRef.current;
    try {
      const started = await api<{ run_id?: string }>(`/teacher/krm/run-analysis?dir_code=${dir}`, { method: "POST" });
      const runId = started?.run_id;
      const t0 = Date.now();
      if (!runId) throw new Error("Сервер не вернул ID задачи");
      for (let i = 0; i < 360; i++) {
        await new Promise((res) => setTimeout(res, 5000));
        // Компонент размонтирован или направление сменили - выходим,
        // не затирая чужое состояние.
        if (!mountedRef.current || selectedDirRef.current !== dir) return;
        const spent = Math.round((Date.now() - t0) / 1000);
        let s: { status: string; error?: string } | null = null;
        try {
          s = await api<{ status: string; error?: string }>(`/teacher/rpd/status/${runId}`);
        } catch {
          setRunMsg(`Этап: пересчёт анализа… нет связи (${spent} c)`);
          continue;
        }
        if (!s) continue;
        if (s.status === "completed") {
          const a = await api<DirectionAnalysis>(`/teacher/analysis?dir_code=${dir}`).catch(() => null);
          if (a) setAnalysis(a);
          setRunMsg("Готово");
          break;
        }
        if (s.status === "failed" || s.status === "cancelled") {
          setRunMsg(s.status === "cancelled" ? "Отменено пользователем" : `Ошибка анализа: ${s.error || "unknown"}`);
          break;
        }
        setRunMsg(`Этап: пересчёт анализа… ${spent} c`);
        if (i === 359) setRunMsg("Превышено ожидание - проверьте результат позже");
      }
    } catch (e) {
      setRunMsg(`Ошибка запуска: ${errMessage(e) || String(e)}`);
    } finally {
      setRunLoading(false);
    }
  }
  runAnalysisRef.current = runDirectionAnalysis;

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
      const r = await api<{ seeded: number }>(`/teacher/krm/recommendations/seed/auto?dir_code=${selectedDir}`, {
        method: "POST",
      });
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
      const data = await api<DisciplineDetail>(
        `/teacher/krm/disciplines/${encodeURIComponent(name)}?dir_code=${selectedDir}&year=${intakeYear}`,
      );
      setSelected(data);
      // Каждая дисциплина начинается с первого шага, свёрнутого аккордеона
      // и чистых фильтров.
      setDisciplineView("coverage");
      setAnalysisMode("coverage");
      setOpenComp(null);
      setCompSearch("");
      setCompFilter("all");
      setRecComp("");
      setRecTypeF("");
    } catch (err) {
      reportError("app/components/TeacherDashboard.tsx", err);
    }
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
      reportError("app/components/TeacherDashboard.tsx", err);
      setRecError(err instanceof Error ? err.message : String(err));
      return false;
    }
  }

  async function deleteRec(id: number) {
    try {
      await api(`/teacher/krm/recommendations/${id}`, { method: "DELETE" });
      setRecs((prev) => prev.filter((r) => r.id !== id));
    } catch (err) {
      reportError("app/components/TeacherDashboard.tsx", err);
    }
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
  const discFiltersActive = !!search.trim() || yearFilter !== 0 || discFilter !== "all" || discSort !== "plan";
  // Шаг «Компетенции и ЗУН»: поиск + наличие ЗУН.
  const compsVisible = useMemo(
    () => (selected ? filterCompetencies(selected.competencies, { search: compSearch, filter: compFilter }) : []),
    [selected, compSearch, compFilter],
  );
  const compCounts = useMemo(
    () =>
      selected ? competencyFilterCounts(filterCompetencies(selected.competencies, { search: compSearch })) : null,
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
        <div className="w-full shrink-0 space-y-2.5 overflow-hidden border-b border-border p-4 md:w-[340px] md:min-w-[340px] md:border-b-0 md:border-r">
          <Skeleton className="h-8 w-full" />
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
    <div className="flex h-[calc(100vh-100px)] flex-col overflow-hidden rounded-2xl border border-border bg-card font-sans text-foreground">
      <TeacherHeader
        directions={directions}
        selectedDir={selectedDir}
        onDirChange={(v) => {
          setSelectedDir(v);
          setSelected(null);
        }}
        intakeYear={intakeYear}
        onYearChange={(y) => {
          setIntakeYear(y);
          setSelected(null);
        }}
        analysis={analysis}
        scopeChain={scopeChain}
        stale={metaStale}
        onOpenData={() => setDataOpsOpen(true)}
        onRerun={runDirectionAnalysis}
        runLoading={runLoading}
      />

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <aside className="flex w-full shrink-0 flex-col overflow-hidden border-b border-border md:w-[340px] md:min-w-[340px] md:border-b-0 md:border-r">
          <DisciplineList
            filtered={filtered}
            disciplines={disciplines}
            analysisByName={analysisByName}
            selected={selected}
            search={search}
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
        </aside>

        <main className="min-h-0 flex-1 overflow-auto p-6" id="teacher-main">
          {selected ? (
            <>
              {/* Хлебные крошки: направление -> дисциплина */}
              <div className="mb-4 flex flex-wrap items-center gap-2">
                <Button variant="secondary" size="sm" onClick={() => setSelected(null)} title="Вернуться к сводке направления">
                  ← Направление {selectedDir}
                </Button>
                <span aria-hidden="true" className="text-subtle-foreground">/</span>
                <h2 className="m-0 text-xl text-foreground">{selected.name}</h2>
                <span className="text-xs text-muted-foreground">
                  {selected.competencies.length} {plural(selected.competencies.length, "компетенция", "компетенции", "компетенций")}
                  {analysisByName.get(selected.name) && (
                    <> · {analysisByName.get(selected.name)!.gaps} {plural(analysisByName.get(selected.name)!.gaps, "пробел", "пробела", "пробелов")}</>
                  )}
                </span>
                {metaStale && (
                  <Badge tone="warning" title="Данные устарели: пересчитайте анализ">
                    устарело
                  </Badge>
                )}
              </div>

              <div className="mb-4 flex flex-wrap gap-1.5" role="tablist" aria-label="Разделы дисциплины">
                {([
                  ["coverage", "Покрытие"],
                  ["zun", `Компетенции и ЗУН (${selected.competencies.length})`],
                  ["recs", `Рекомендации (${discRecCount})`],
                ] as const).map(([id, label]) => (
                  <button
                    key={id}
                    role="tab"
                    aria-selected={disciplineView === id}
                    onClick={() => setDisciplineView(id)}
                    className={`cursor-pointer rounded-md border-0 px-3.5 py-1.5 text-xs ${
                      disciplineView === id
                        ? "bg-primary text-primary-foreground font-bold"
                        : "bg-muted text-primary font-normal"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {disciplineView === "coverage" && (
                <AnalysisPanel disciplineName={selected.name} dirCode={selectedDir} />
              )}

              {disciplineView === "zun" && (
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
              )}

              {disciplineView === "recs" && (
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
          ) : (
            <div>
              <div className="mb-4 flex flex-wrap items-center gap-1.5">
                <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Разделы направления">
                  {([
                    ["coverage", "Обзор"],
                    ["trends", "Тренды компетенций"],
                    ["taxonomy", "Таксономия"],
                  ] as const).map(([id, label]) => (
                    <button
                      key={id}
                      role="tab"
                      aria-selected={analysisMode === id}
                      onClick={() => setAnalysisMode(id)}
                      disabled={id === "coverage" && !analysis}
                      title={id === "coverage" && !analysis ? "Появится после запуска анализа направления" : undefined}
                      className={`cursor-pointer rounded-md border-0 px-3.5 py-1.5 text-xs disabled:cursor-not-allowed disabled:opacity-50 ${
                        analysisMode === id
                          ? "bg-primary font-bold text-primary-foreground"
                          : "bg-muted font-normal text-primary"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <div className="ml-auto">
                  <MetricsGlossary />
                </div>
              </div>

              {analysisMode === "coverage" && analysis && (
                <>
                  <DirectionKpis analysis={analysis} />
                  <DirectionInsights analysis={analysis} />
                </>
              )}
              {analysisMode === "coverage" && !analysis && (
                <div className="mt-16 text-center text-sm text-subtle-foreground">
                  <p>Анализ направления ещё не считался.</p>
                  <Button className="mt-3" loading={runLoading} onClick={runDirectionAnalysis}>
                    Запустить анализ
                  </Button>
                  {runMsg && <div className="mt-2 text-xs text-warning">{runMsg}</div>}
                </div>
              )}
              {analysisMode === "trends" && <CompetencyTrendsPanel dirCode={selectedDir} />}
              {analysisMode === "taxonomy" && (
                <div className="mt-2">
                  <TaxonomyBrowser showSuggest />
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      <DataOpsDrawer
        open={dataOpsOpen}
        onClose={() => setDataOpsOpen(false)}
        dirCode={selectedDir}
        scopeChain={scopeChain}
        scopeSaving={scopeSaving}
        scopeMsg={scopeMsg}
        onScopeBulk={saveScopeBulk}
        stale={metaStale}
        runLoading={runLoading}
        runMsg={runMsg}
        onRun={runDirectionAnalysis}
        rpd={rpd}
      />

      {/* Ключ контекста чата - КОД направления (selectedDir, напр. 09.03.02):
          бэкенд llm_chat строит контекст из data/result/teacher/<код>/_summary.json
          и krm_disciplines_<код>.json, человеческое имя давало пустой контекст.
          Подпись в шапке (contextLabel) остаётся человекочитаемой. */}
      <LlmChatPanel role="teacher" discipline={selectedDir} contextLabel={selected?.name ?? null} />
    </div>
  );
}
