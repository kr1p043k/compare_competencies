import { useState, useEffect } from "react";
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
}

export function PredictionsTab() {
  const [activeTab, setActiveTab] = useState("growing");
  const [forecasts, setForecasts] = useState<ForecastItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSkill, setSelectedSkill] = useState<ForecastItem | null>(null);
  const [vacanciesCount, setVacanciesCount] = useState<number>(0);
  const [dataFrom, setDataFrom] = useState<string | null>(null);
  const [dataTo, setDataTo] = useState<string | null>(null);
  const [months, setMonths] = useState(12);

  useEffect(() => {
    loadForecasts("growing");
  }, []);

  const loadForecasts = async (direction: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/forecast/top?n=25&months=${months}&direction=${direction}`);
      if (!res.ok) throw new Error("Failed to load forecasts");
      const data = await res.json();
      setForecasts(data.forecasts || []);
      setVacanciesCount(data.vacancies_count || 0);
      setDataFrom(data.data_from || null);
      setDataTo(data.data_to || null);
      if (data.requested_months && data.months !== data.requested_months) {
        setMonths(data.months);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    loadForecasts(tab);
  };

  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={handleTabChange} className="space-y-4">
        <TabsList className="inline-flex h-12 items-center justify-center rounded-lg bg-gray-100 p-1">
          <TabsTrigger value="growing" className="inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm">
            <TrendingUp className="size-4 text-green-600" />
            Растущие
          </TabsTrigger>
          <TabsTrigger value="declining" className="inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium data-[state=active]:bg-white data-[state=active]:shadow-sm">
            <TrendingDown className="size-4 text-red-600" />
            Падающие
          </TabsTrigger>
        </TabsList>

          <TabsContent value="growing" className="space-y-4">
            <Card className="border border-gray-200 shadow-sm">
              <CardHeader className="border-b border-gray-200 bg-gray-50">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 bg-green-600 rounded-lg">
                    <TrendingUp className="size-5 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-xl font-semibold text-gray-900">Топ растущих навыков</CardTitle>
                    <CardDescription>Прогноз популярности · {vacanciesCount ? `${vacanciesCount} вакансий` : "—"} · {dataFrom && dataTo ? `${dataFrom}–${dataTo}` : "—"}</CardDescription>
                  </div>
                  <div className="ml-auto flex items-center gap-2">
                    <CalendarDays className="size-4 text-gray-400" />
                    <Select value={String(months)} onValueChange={(v) => { setMonths(Number(v)); loadForecasts(activeTab); }}>
                      <SelectTrigger className="w-28 h-9 text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1">1 месяц</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
            </CardHeader>
            <CardContent className="p-6">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Sparkles className="size-6 text-blue-500 animate-pulse" />
                  <span className="ml-3 text-gray-600">Загрузка прогнозов...</span>
                </div>
              ) : error ? (
                <div className="flex items-center gap-3 py-8 text-red-600">
                  <AlertCircle className="size-5" />
                  <span>{error}</span>
                </div>
              ) : (<div className="space-y-2">
              {forecasts.map((f, i) => (<ForecastRow key={f.skill} item={f} rank={i + 1} expanded={selectedSkill?.skill === f.skill} months={months} onToggle={() => setSelectedSkill(selectedSkill?.skill === f.skill ? null : f)} />))}
              </div>)}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="declining" className="space-y-4">
          <Card className="border border-gray-200 shadow-sm">
            <CardHeader className="border-b border-gray-200 bg-gray-50">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-10 h-10 bg-red-600 rounded-lg">
                  <TrendingDown className="size-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-xl font-semibold text-gray-900">Падающие навыки</CardTitle>
                  <CardDescription>Навыки с отрицательным прогнозом роста · {vacanciesCount ? `${vacanciesCount} вакансий` : "—"} · {dataFrom && dataTo ? `${dataFrom}–${dataTo}` : "—"}</CardDescription>
                </div>
                <div className="ml-auto flex items-center gap-2">
                  <CalendarDays className="size-4 text-gray-400" />
                  <Select value={String(months)} onValueChange={(v) => { setMonths(Number(v)); loadForecasts(activeTab); }}>
                    <SelectTrigger className="w-28 h-9 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1 месяц</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-6">
              {forecasts.map((f, i) => (<ForecastRow key={f.skill} item={f} rank={i + 1} expanded={selectedSkill?.skill === f.skill} onToggle={() => setSelectedSkill(selectedSkill?.skill === f.skill ? null : f)} />))}
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
  const methodColors: Record<string, string> = { prophet: "bg-purple-100 text-purple-700", ets: "bg-blue-100 text-blue-700", linear: "bg-gray-100 text-gray-700", genetic: "bg-amber-100 text-amber-700" };

  return (
    <div className="border border-gray-100 rounded-lg overflow-hidden">
      <button onClick={onToggle} className="w-full flex items-center gap-3 p-3 hover:bg-gray-50 transition-colors text-left">
        <span className="w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center text-xs font-medium text-gray-500">{rank}</span>
        <span className="flex-1 font-medium text-gray-900">{item.skill}</span>
        <div className="flex items-center gap-2">
          <span className={`text-sm font-semibold ${changePct > 0 ? "text-green-600" : "text-red-600"}`}>
            {changePctSign}{changePct.toFixed(1)}%
          </span>
          <Badge className={`text-xs border-0 ${methodColors[item.method] || "bg-gray-100"}`}>{item.method}</Badge>
          <div className={`w-2 h-2 rounded-full ${changePct > 0 ? "bg-green-500" : "bg-red-500"}`} />
        </div>
        {expanded ? <ChevronUp className="size-4 text-gray-400" /> : <ChevronDown className="size-4 text-gray-400" />}
      </button>
      {expanded && (
        <div className="px-3 pb-3 pt-0 border-t border-gray-100">
          <div className="grid grid-cols-3 gap-4 mt-3 mb-3">
            <div className="text-center p-2 bg-gray-50 rounded-lg"><div className="text-xs text-gray-500">Сейчас</div><div className="text-lg font-semibold text-gray-900">{item.current_frequency.toFixed(0)}</div></div>
            <div className="text-center p-2 bg-gray-50 rounded-lg"><div className="text-xs text-gray-500">Через {months === 1 ? "месяц" : `${months || 12} мес`}</div><div className="text-lg font-semibold text-gray-900">{item.next_year_frequency.toFixed(0)}</div></div>
            <div className="text-center p-2 bg-gray-50 rounded-lg"><div className="text-xs text-gray-500">Уверенность</div><div className="text-lg font-semibold text-gray-900">{(item.confidence * 100).toFixed(0)}%</div></div>
          </div>
          {item.uncertainty_upper && item.uncertainty_lower && (
            <div className="text-xs text-gray-400 text-center mb-2">
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

function MiniChart({ data }: { data: number[] }) {
  if (!data.length) return null;
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
