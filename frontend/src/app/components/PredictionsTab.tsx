import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  TrendingUp, TrendingDown, BarChart3, Sparkles,
  ChevronDown, ChevronUp, AlertCircle, CalendarDays,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./ui/tabs";

interface ForecastItem {
  skill: string;
  current_frequency: number;
  predicted_growth: number;
  predicted_change_pct: number;
  confidence: number;
  next_year_frequency: number;
  method: string;
  trend_direction: string;
  forecast_steps?: number[];
  history?: number[];
  uncertainty_upper?: number;
  uncertainty_lower?: number;
  data_points?: number;
  mape?: number;
  forecast_months?: number;
}

interface ObservedItem {
  skill: string;
  observed_change_pct: number;
  first_frequency: number;
  last_frequency: number;
  points: number;
}

export function PredictionsTab() {
  const [activeTab, setActiveTab] = useState("growing");
  const [forecasts, setForecasts] = useState<ForecastItem[]>([]);
  const [observed, setObserved] = useState<ObservedItem[]>([]);
  const [observedMeta, setObservedMeta] = useState<{ required_points?: number; max_points?: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSkill, setSelectedSkill] = useState<ForecastItem | null>(null);
  const [vacanciesCount, setVacanciesCount] = useState<number>(0);
  const [dataFrom, setDataFrom] = useState<string | null>(null);
  const [dataTo, setDataTo] = useState<string | null>(null);
  const [months, setMonths] = useState(3);
  const [sort, setSort] = useState("growth");
  const [snapshotsCount, setSnapshotsCount] = useState<number | null>(null);
  const [engineInfo, setEngineInfo] = useState<{ engine?: string; prophet_models?: number } | null>(null);
  const [fitStatus, setFitStatus] = useState<{ state?: string; done?: number; total?: number; eta_seconds?: number | null } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    loadForecasts("growing");
  }, []);

  // Countdown Prophet-фита: пока движок genetic – опрашиваем лёгкий статус.
  useEffect(() => {
    if (engineInfo && engineInfo.engine !== "genetic") return;
    let stop = false;
    const poll = async () => {
      try {
        const res = await fetch("/api/forecast/engine");
        if (!res.ok || stop) return;
        const data = await res.json();
        setFitStatus({ state: data.state, done: data.done, total: data.total, eta_seconds: data.eta_seconds });
        if (data.engine === "prophet" || data.state === "ready" || data.state === "failed") {
          setEngineInfo({ engine: data.engine, prophet_models: data.prophet_models ?? 0 });
          stop = true;
        }
      } catch { /* следующий тик */ }
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => { stop = true; clearInterval(id); };
  }, [engineInfo?.engine]);

  const loadForecasts = async (direction: string, monthsOverride?: number, sortOverride?: string) => {
    const effMonths = monthsOverride ?? months;
    const effSort = sortOverride ?? sort;
    // Быстрые переключения (1->3 мес) не копят висящие тяжёлые запросы.
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    try {
      // Падающие – измеренное падение спроса за окно, а не прогноз:
      // /forecast/observed возвращает факты по снимкам.
      const url = direction === "declining"
        ? `/api/forecast/observed?n=25&months=${effMonths}&sort=${effSort}`
        : `/api/forecast/top?n=25&months=${effMonths}&direction=${direction}&sort=${effSort}`;
      const res = await fetch(url, { signal: ctrl.signal });
      if (!res.ok) {
        if (res.status === 400) {
          const err = await res.json().catch(() => ({}));
          throw new Error((err as any).detail || "Не удалось загрузить прогнозы. Попробуйте позже.");
        }
        throw new Error("Не удалось загрузить прогнозы. Попробуйте позже.");
      }
      const data = await res.json();
      if (direction === "declining") {
        setObserved((data.forecasts || []).filter((f: ObservedItem) => (f.skill || "").trim()));
        setObservedMeta({ required_points: data.required_points, max_points: data.max_points });
      } else {
        setForecasts((data.forecasts || []).filter((f: ForecastItem) => (f.skill || "").trim()));
      }
      setVacanciesCount(data.vacancies_count || 0);
      setDataFrom(data.data_from || null);
      setDataTo(data.data_to || null);
      setSnapshotsCount(typeof data.snapshots_count === "number" ? data.snapshots_count : null);
      if (typeof data.engine === "string") {
        setEngineInfo({ engine: data.engine, prophet_models: data.prophet_models ?? 0 });
      }
      if (data.requested_months && data.months !== data.requested_months) {
        setMonths(data.months);
      }
    } catch (e: any) {
      if (e?.name === "AbortError") return;
      setError(e.message);
    } finally {
      if (abortRef.current === ctrl) setLoading(false);
    }
  };

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    loadForecasts(tab);
  };

  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={handleTabChange} className="space-y-4">
        <TabsList className="inline-flex h-12 items-center justify-center rounded-lg bg-gray-100 dark:bg-slate-800 p-1">
          <TabsTrigger value="growing" className="inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm dark:data-[state=active]:bg-slate-700 dark:data-[state=active]:text-slate-100">
            <TrendingUp className="size-4 text-green-600" />
            Растущие
          </TabsTrigger>
          <TabsTrigger value="declining" className="inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm dark:data-[state=active]:bg-slate-700 dark:data-[state=active]:text-slate-100">
            <TrendingDown className="size-4 text-red-600" />
            Падающие
          </TabsTrigger>
        </TabsList>

          <TabsContent value="growing" className="space-y-4">
            <Card className="border border-gray-200 dark:border-slate-700 shadow-sm">
              <CardHeader className="border-b border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-900">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 bg-green-600 rounded-lg">
                    <TrendingUp className="size-5 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-xl font-semibold text-gray-900 dark:text-slate-100">Топ растущих навыков</CardTitle>
                    <CardDescription>Прогноз популярности · {vacanciesCount ? `${vacanciesCount} вакансий` : "–"} · {dataFrom && dataTo ? `${dataFrom}–${dataTo}` : "–"}{engineInfo?.engine ? ` · Движок: ${engineInfo.engine === "prophet" ? `Prophet (${engineInfo.prophet_models} моделей)` : "Genetic (Prophet ещё считается)"}` : ""}</CardDescription>
                  </div>
                  <div className="ml-auto flex items-center gap-2">
                    <CalendarDays className="size-4 text-gray-400 dark:text-slate-500" />
                    <Select value={String(months)} onValueChange={(v) => { const m = Number(v); setMonths(m); loadForecasts(activeTab, m); }}>
                      <SelectTrigger className="w-28 h-9 text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1">1 месяц</SelectItem>
                        <SelectItem value="3">3 месяца</SelectItem>
                        <SelectItem value="6">6 месяцев</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={sort} onValueChange={(v) => { setSort(v); loadForecasts(activeTab, undefined, v); }}>
                      <SelectTrigger className="w-36 h-9 text-sm" title="Сортировка списка">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="growth">По росту</SelectItem>
                        <SelectItem value="popular">Популярные</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
            </CardHeader>
            <CardContent className="p-6">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Sparkles className="size-6 text-blue-500 animate-pulse" />
                  <span className="ml-3 text-gray-600 dark:text-slate-400">Загрузка прогнозов...</span>
                </div>
              ) : error ? (
                <div className="flex items-center gap-3 py-8 text-red-600">
                  <AlertCircle className="size-5" />
                  <span>{error}</span>
                </div>
              ) : (<div className="space-y-2">
              {fitStatus?.state === "fitting" && (
                <div className="text-xs text-gray-500 dark:text-slate-400">
                  Prophet считается: {fitStatus.done ?? 0}/{fitStatus.total ?? "?"} моделей
                  {fitStatus.eta_seconds != null ? ` · осталось ~${fitStatus.eta_seconds < 120 ? `${Math.max(1, Math.round(fitStatus.eta_seconds))} с` : `${Math.round(fitStatus.eta_seconds / 60)} мин`}` : ""}
                </div>
              )}
              {forecasts.length === 0 && (
                <div className="py-8 text-center text-gray-500 dark:text-slate-400 text-sm">
                  <p>No forecasts yet – not enough history snapshots ({snapshotsCount}).</p>
                  <p className="mt-1">Run the nightly pipeline a few times to accumulate trend snapshots.</p>
                </div>
              )}
              {forecasts.map((f, i) => (<ForecastRow key={f.skill} item={f} rank={i + 1} expanded={selectedSkill?.skill === f.skill} months={months} onToggle={() => setSelectedSkill(selectedSkill?.skill === f.skill ? null : f)} />))}
              </div>)}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="declining" className="space-y-4">
          <Card className="border border-gray-200 dark:border-slate-700 shadow-sm">
            <CardHeader className="border-b border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-900">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-10 h-10 bg-red-600 rounded-lg">
                  <TrendingDown className="size-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-xl font-semibold text-gray-900 dark:text-slate-100">Падающие навыки</CardTitle>
                  <CardDescription>Измеренное падение спроса за период · {vacanciesCount ? `${vacanciesCount} вакансий` : "–"} · {dataFrom && dataTo ? `${dataFrom}–${dataTo}` : "–"}</CardDescription>
                </div>
                <div className="ml-auto flex items-center gap-2">
                  <CalendarDays className="size-4 text-gray-400 dark:text-slate-500" />
                  <Select value={String(months)} onValueChange={(v) => { const m = Number(v); setMonths(m); loadForecasts(activeTab, m); }}>
                    <SelectTrigger className="w-28 h-9 text-sm" title="Окно измерения">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1 месяц</SelectItem>
                      <SelectItem value="3">3 месяца</SelectItem>
                      <SelectItem value="6">6 месяцев</SelectItem>
                      <SelectItem value="12">12 месяцев</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-6">
              {fitStatus?.state === "fitting" && (
                <div className="mb-3 text-xs text-gray-500 dark:text-slate-400">
                  Prophet считается: {fitStatus.done ?? 0}/{fitStatus.total ?? "?"} моделей
                  {fitStatus.eta_seconds != null ? ` · осталось ~${fitStatus.eta_seconds < 120 ? `${Math.max(1, Math.round(fitStatus.eta_seconds))} с` : `${Math.round(fitStatus.eta_seconds / 60)} мин`}` : ""}
                </div>
              )}
              {observed.length === 0 && (
                <div className="py-8 text-center text-gray-500 dark:text-slate-400 text-sm">
                  <p>Нет данных: для окна нужно снимков: {observedMeta?.required_points ?? "–"}, есть максимум: {observedMeta?.max_points ?? snapshotsCount ?? "–"}.</p>
                  <p className="mt-1">Уменьшите окно или дождитесь новых снимков пайплайна.</p>
                </div>
              )}
              {observed.map((f, i) => (<ObservedRow key={f.skill} item={f} rank={i + 1} />))}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function ForecastRow({ item, rank, expanded, onToggle, months }: { item: ForecastItem; rank: number; expanded: boolean; onToggle: () => void; months?: number }) {
  const changePct = item.predicted_change_pct ?? (item.predicted_growth * 100);
  const changePctSign = changePct > 0 ? "+" : "";
  const methodColors: Record<string, string> = { prophet: "bg-purple-100 dark:bg-purple-950/30 text-purple-700 dark:text-purple-300", ets: "bg-blue-100 dark:bg-blue-950/30 text-blue-700 dark:text-blue-300", linear: "bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300", genetic: "bg-amber-100 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300", trend: "bg-sky-100 dark:bg-sky-950/30 text-sky-700 dark:text-sky-300", insufficient_data: "bg-gray-200 dark:bg-slate-700 text-gray-500 dark:text-slate-400" };
  const insufficient = item.method === "insufficient_data" || (item.data_points !== undefined && item.data_points < 3);
  const horizon = item.forecast_months ?? (insufficient ? 0 : (months || 12));
  const confPct = Math.round((item.confidence ?? 0) * 100);

  return (
    <div className="border border-gray-100 dark:border-slate-800 rounded-lg overflow-hidden">
      <button onClick={onToggle} className="w-full flex items-center gap-3 p-3 hover:bg-gray-50 dark:hover:bg-slate-800 transition-colors text-left">
        <span className="w-6 h-6 rounded-full bg-gray-100 dark:bg-slate-800 flex items-center justify-center text-xs font-medium text-gray-500 dark:text-slate-400">{rank}</span>
        <span className="flex-1 font-medium text-gray-900 dark:text-slate-100">{item.skill}</span>
        <div className="flex items-center gap-2">
          <span className={`text-sm font-semibold ${insufficient ? "text-gray-400 dark:text-slate-500" : changePct > 0 ? "text-green-600" : "text-red-600"}`}>
            {insufficient ? "–" : `${changePctSign}${changePct.toFixed(1)}%`}
          </span>
          <Badge className={`text-xs border-0 ${methodColors[item.method] || "bg-gray-100 dark:bg-slate-800"}`}>
            {insufficient ? "нет данных" : (item.method || "trend")}
          </Badge>
          {!insufficient && <div className={`w-2 h-2 rounded-full ${changePct > 0 ? "bg-green-50 dark:bg-green-950/30" : "bg-red-50 dark:bg-red-950/30"}`} />}
        </div>
        {expanded ? <ChevronUp className="size-4 text-gray-400 dark:text-slate-500" /> : <ChevronDown className="size-4 text-gray-400 dark:text-slate-500" />}
      </button>
      {expanded && (
        <div className="px-3 pb-3 pt-0 border-t border-gray-100 dark:border-slate-800">
          <div className="grid grid-cols-3 gap-4 mt-3 mb-3">
            <div className="text-center p-2 bg-gray-50 dark:bg-slate-900 rounded-lg"><div className="text-xs text-gray-500 dark:text-slate-400">Сейчас</div><div className="text-lg font-semibold text-gray-900 dark:text-slate-100">{item.current_frequency.toFixed(0)}</div></div>
            <div className="text-center p-2 bg-gray-50 dark:bg-slate-900 rounded-lg"><div className="text-xs text-gray-500 dark:text-slate-400">Через {horizon === 1 ? "месяц" : `${horizon} мес`}</div><div className="text-lg font-semibold text-gray-900 dark:text-slate-100">{insufficient ? "–" : item.next_year_frequency.toFixed(0)}</div></div>
            <div className="text-center p-2 bg-gray-50 dark:bg-slate-900 rounded-lg"><div className="text-xs text-gray-500 dark:text-slate-400">Уверенность</div><div className="text-lg font-semibold text-gray-900 dark:text-slate-100">{insufficient ? "–" : `${confPct}%`}</div></div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-gray-500 dark:text-slate-400 mb-2">
            {item.data_points !== undefined && (
              <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-slate-800">точек истории: {item.data_points}</span>
            )}
            {item.mape !== undefined && item.mape > 0 && (
              <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-slate-800">hold-out MAPE: {item.mape.toFixed(2)}</span>
            )}
            {insufficient && (
              <span className="px-2 py-0.5 rounded bg-amber-100 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300">недостаточно данных для прогноза</span>
            )}
            {((confPct < 30) || (confPct < 50) || ((item.data_points ?? 99) < 4)) && !insufficient && (
              <span className="px-2 py-0.5 rounded bg-amber-100 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300">{confPct < 30 ? "низкая достоверность" : "низкая достоверность (мало данных)"}</span>
            )}
          </div>
          {item.uncertainty_upper && item.uncertainty_lower && (
            <div className="text-xs text-gray-400 dark:text-slate-500 text-center mb-2">
              Доверительный интервал 80%: [{item.uncertainty_lower.toFixed(2)} – {item.uncertainty_upper.toFixed(2)}]
            </div>
          )}
          {item.forecast_steps && item.forecast_steps.length > 1 && (
            <div className="mt-2"><MiniChart data={item.forecast_steps} /></div>
          )}
        </div>
      )}
    </div>
  );
}

function ObservedRow({ item, rank }: { item: ObservedItem; rank: number }) {
  return (
    <div className="border border-gray-100 dark:border-slate-800 rounded-lg overflow-hidden">
      <div className="w-full flex items-center gap-3 p-3">
        <span className="w-6 h-6 rounded-full bg-gray-100 dark:bg-slate-800 flex items-center justify-center text-xs font-medium text-gray-500 dark:text-slate-400">{rank}</span>
        <span className="flex-1 font-medium text-gray-900 dark:text-slate-100">{item.skill}</span>
        <span className="text-sm font-semibold text-red-600">
          {item.observed_change_pct.toFixed(1)}%
        </span>
        <span className="text-[11px] text-gray-500 dark:text-slate-400 tabular-nums">
          {item.first_frequency.toFixed(0)} → {item.last_frequency.toFixed(0)} · точек: {item.points}
        </span>
      </div>
    </div>
  );
}

function MiniChart({ data }: { data: number[] }) {
  if (!data.length) return null;
  if (data.length < 2) {
    // Single point: dot instead of NaN polyline (verified: i/0 = NaN)
    return (
      <svg viewBox="0 0 400 60" className="w-full h-12" preserveAspectRatio="none">
        <circle cx="200" cy="30" r="4" fill="#3b82f6" />
      </svg>
    );
  }
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const w = 400;
  const h = 60;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-12" preserveAspectRatio="none">
      <polyline points={points} fill="none" stroke="#3b82f6" strokeWidth="2" />
    </svg>
  );
}
