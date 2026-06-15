import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";
import { apiFetch } from "../../lib/auth";
import {
  BookOpen, ChevronDown, ChevronRight, Plus, Trash2,
  RefreshCw, Search, GraduationCap, Lightbulb, Target,
  TrendingUp, TrendingDown, Minus,
} from "lucide-react";
import { AnalysisPanel } from "./AnalysisPanel";

interface CompetencyTrendItem {
  competency_id: string;
  code: string;
  name: string;
  direction: "rising" | "falling" | "stable";
  change_pct: number;
  skill_count: number;
  snapshot_date: string;
  skills: { name: string; direction: string; change_pct: number }[];
}

interface Discipline {
  name: string;
  competencies_count: number;
  skills_count: number;
}

interface Competency {
  code: string;
  skills: string[];
}

interface DisciplineDetail {
  name: string;
  competencies: Competency[];
}

interface Recommendation {
  id: number;
  discipline: string;
  competency: string;
  suggestion: string;
  type: string;
}

interface KrmStats {
  total_disciplines: number;
  total_competencies: number;
  total_skills: number;
}

const KRM_API = "/api/teacher/krm";

export function TeacherDashboard() {
  const [disciplines, setDisciplines] = useState<Discipline[]>([]);
  const [selected, setSelected] = useState<DisciplineDetail | null>(null);
  const [expandedComp, setExpandedComp] = useState<string | null>(null);
  const [stats, setStats] = useState<KrmStats | null>(null);
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [suggestion, setSuggestion] = useState("");
  const [recType, setRecType] = useState("modify");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<"skills" | "analysis" | "trends">("skills");
  const [skillTrendMap, setSkillTrendMap] = useState<Record<string, { direction: string; change_pct: number }>>({});
  const [trends, setTrends] = useState<CompetencyTrendItem[]>([]);
  const [trendsFilter, setTrendsFilter] = useState<string>("all");
  const [trendsLoading, setTrendsLoading] = useState(false);
  const [expandedTrend, setExpandedTrend] = useState<string | null>(null);

  const loadAll = async () => {
    setLoading(true);
    try {
      const [s, d, r] = await Promise.all([
        apiFetch(`${KRM_API}/stats`).then(r => r.ok ? r.json() : null),
        apiFetch(`${KRM_API}/disciplines`).then(r => r.ok ? r.json() : []),
        apiFetch(`${KRM_API}/recommendations`).then(r => r.ok ? r.json() : []),
      ]);
      if (s) setStats(s);
      setDisciplines(d);
      setRecs(r);
    } catch (e) {
      console.error("Failed to load KRM data", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadAll(); }, []);

  const fetchTrends = async (direction?: string) => {
    setTrendsLoading(true);
    try {
      const params = new URLSearchParams({ limit: "100" });
      if (direction && direction !== "all") params.set("direction", direction);
      const res = await apiFetch(`/api/competency-trends?${params}`);
      if (res.ok) setTrends((await res.json()).trends);
    } catch (e) {
      console.error("Failed to load competency trends", e);
    } finally {
      setTrendsLoading(false);
    }
  };

  const fetchSkillTrends = async () => {
    try {
      const res = await apiFetch("/api/competency-trends?limit=500");
      if (!res.ok) return;
      const data = await res.json();
      const map: Record<string, { direction: string; change_pct: number }> = {};
      for (const t of data.trends || []) {
        for (const s of t.skills || []) {
          map[s.name] = { direction: s.direction, change_pct: s.change_pct };
        }
      }
      setSkillTrendMap(map);
    } catch (e) {
      console.error("Failed to load skill trends", e);
    }
  };

  useEffect(() => {
    if (viewMode === "trends") fetchTrends(trendsFilter);
    if (viewMode === "skills") fetchSkillTrends();
  }, [viewMode, trendsFilter]);

  const loadDiscipline = async (name: string) => {
    setViewMode("skills");
    const res = await apiFetch(`${KRM_API}/disciplines/${encodeURIComponent(name)}`);
    if (res.ok) setSelected(await res.json());
  };

  const addRecommendation = async () => {
    if (!selected || !suggestion.trim()) return;
    const res = await apiFetch(`${KRM_API}/recommendations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        discipline: selected.name,
        competency: expandedComp || "",
        suggestion: suggestion.trim(),
        type: recType,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      setRecs(prev => [...prev, {
        id: data.id,
        discipline: selected.name,
        competency: expandedComp || "",
        suggestion: suggestion.trim(),
        type: recType,
      }]);
      setSuggestion("");
    }
  };

  const deleteRecommendation = async (id: number) => {
    await apiFetch(`${KRM_API}/recommendations/${id}`, { method: "DELETE" });
    setRecs(prev => prev.filter(r => r.id !== id));
  };

  const filtered = disciplines.filter(d =>
    d.name.toLowerCase().includes(search.toLowerCase())
  );
  const activeRecs = recs.filter(r => r.discipline === selected?.name);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <RefreshCw className="size-5 mr-2 animate-spin" />
        Loading KRM data...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
            <GraduationCap className="size-5 text-indigo-600" />
            KRM 09.03.02
          </h2>
          <p className="text-sm text-gray-500">
            {stats
              ? `${stats.total_disciplines} дисциплин, ${stats.total_competencies} компетенций, ${stats.total_skills} навыков`
              : "Рабочие программы дисциплин"}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={loadAll} disabled={loading}>
          <RefreshCw className="size-4 mr-2" />
          Обновить
        </Button>
      </div>

      <div className="grid grid-cols-[380px_1fr] gap-6">
        <Card className="border border-gray-200 shadow-sm h-[calc(100vh-280px)] flex flex-col">
          <CardHeader className="border-b border-gray-200 bg-gray-50 px-4 py-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-gray-400" />
              <Input
                placeholder="Поиск дисциплин..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="pl-9 h-9 text-sm"
              />
            </div>
          </CardHeader>
          <CardContent className="flex-1 overflow-auto p-0">
            {filtered.length === 0 && (
              <div className="p-4 text-sm text-gray-400 text-center">Нет дисциплин</div>
            )}
            {filtered.map(d => (
              <button
                key={d.name}
                onClick={() => loadDiscipline(d.name)}
                className="w-full text-left px-4 py-3 border-b border-gray-100 transition-colors hover:bg-indigo-50"
              >
                <div className="text-sm font-medium text-gray-900 truncate">{d.name}</div>
                <div className="text-xs text-gray-400 mt-0.5">
                  {d.competencies_count} компетенций / {d.skills_count} навыков
                </div>
              </button>
            ))}
          </CardContent>
        </Card>

        <Card className="border border-gray-200 shadow-sm h-[calc(100vh-280px)] flex flex-col">
          <CardHeader className="border-b border-gray-200 bg-gray-50 px-5 py-4">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base text-gray-900">
                  {viewMode === "trends" ? "Тренды компетенций" : selected?.name || "Teacher Dashboard"}
                </CardTitle>
                <CardDescription className="text-xs">
                  {viewMode === "trends" ? `${trends.length} компетенций` : selected ? `${selected.competencies.length} компетенций` : ""}
                </CardDescription>
              </div>
              <div className="flex gap-1">
                {selected && (
                  <>
                    <Button
                      variant={viewMode === "skills" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setViewMode("skills")}
                      className={viewMode === "skills" ? "bg-indigo-600" : ""}
                    >
                      <BookOpen className="size-3.5 mr-1" />
                      Skills
                    </Button>
                    <Button
                      variant={viewMode === "analysis" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setViewMode("analysis")}
                      className={viewMode === "analysis" ? "bg-indigo-600" : ""}
                    >
                      <Target className="size-3.5 mr-1" />
                      Analysis
                    </Button>
                  </>
                )}
                <Button
                  variant={viewMode === "trends" ? "default" : "outline"}
                  size="sm"
                  onClick={() => { setViewMode("trends"); setSelected(null); }}
                  className={viewMode === "trends" ? "bg-indigo-600" : ""}
                >
                  <TrendingUp className="size-3.5 mr-1" />
                  Trends
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="flex-1 overflow-auto p-4 space-y-3">
            {viewMode === "trends" ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Direction</span>
                  {["all", "rising", "falling", "stable"].map(d => (
                    <button
                      key={d}
                      onClick={() => setTrendsFilter(d)}
                      className={`text-xs px-3 py-1 rounded-full border transition-colors ${
                        trendsFilter === d
                          ? "bg-indigo-600 text-white border-indigo-600"
                          : "bg-white text-gray-600 border-gray-300 hover:bg-gray-50"
                      }`}
                    >
                      {d === "all" ? "All" : d === "rising" ? "↑ Rising" : d === "falling" ? "↓ Falling" : "→ Stable"}
                    </button>
                  ))}
                </div>
                {trendsLoading ? (
                  <div className="text-center text-gray-400 text-sm py-8">
                    <RefreshCw className="size-5 mx-auto mb-2 animate-spin" />
                    Loading trends...
                  </div>
                ) : trends.length === 0 ? (
                  <div className="text-center text-gray-400 text-sm py-8">
                    <TrendingUp className="size-8 mx-auto mb-2 opacity-30" />
                    No competency trends data. Run pipeline first.
                  </div>
                ) : (
                  <div className="divide-y divide-gray-100 border rounded-lg">
                    {trends.map((t) => {
                      const isExpanded = expandedTrend === t.competency_id;
                      return (
                        <div key={t.competency_id} className="border-b border-gray-100 last:border-0">
                          <button
                            onClick={() => setExpandedTrend(isExpanded ? null : t.competency_id)}
                            className="w-full flex items-center justify-between px-3 py-2.5 text-sm hover:bg-gray-50 transition-colors text-left"
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              {isExpanded ? <ChevronDown className="size-3.5 text-gray-400 shrink-0" /> : <ChevronRight className="size-3.5 text-gray-400 shrink-0" />}
                              <span className="font-mono text-xs font-semibold text-indigo-700 shrink-0">{t.code}</span>
                              {t.name && t.name !== t.code && (
                                <span className="text-gray-600 truncate text-xs">{t.name}</span>
                              )}
                            </div>
                            <div className="flex items-center gap-3 shrink-0">
                              <span className="text-xs text-gray-400">{t.skill_count} skills</span>
                              <span className={`font-semibold text-xs flex items-center gap-0.5 ${
                                t.direction === "rising" ? "text-green-600" :
                                t.direction === "falling" ? "text-red-600" : "text-gray-500"
                              }`}>
                                {t.direction === "rising" ? <TrendingUp className="size-3" /> :
                                 t.direction === "falling" ? <TrendingDown className="size-3" /> :
                                 <Minus className="size-3" />}
                                {t.change_pct > 0 ? "+" : ""}{t.change_pct.toFixed(1)}%
                              </span>
                            </div>
                          </button>
                          {isExpanded && t.skills.length > 0 && (
                            <div className="px-3 pb-3 border-t border-gray-100 pt-2 space-y-1">
                              {t.skills.map((s, i) => (
                                <div key={i} className="flex items-center justify-between text-xs py-0.5">
                                  <span className="text-gray-700">{s.name}</span>
                                  <span className={`font-medium flex items-center gap-0.5 ${
                                    s.direction === "rising" ? "text-green-600" :
                                    s.direction === "falling" ? "text-red-600" : "text-gray-400"
                                  }`}>
                                    {s.direction === "rising" ? "↑" : s.direction === "falling" ? "↓" : "→"}
                                    {s.change_pct > 0 ? "+" : ""}{s.change_pct}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          )}
                          {isExpanded && t.skills.length === 0 && (
                            <div className="px-3 pb-3 text-xs text-gray-400 italic border-t border-gray-100 pt-2">
                              No skills linked to this competency
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : !selected ? (
              <div className="flex-1 flex items-center justify-center text-gray-400 min-h-[400px]">
                <div className="text-center">
                  <BookOpen className="size-12 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">Выберите дисциплину из списка</p>
                </div>
              </div>
            ) : viewMode === "analysis" ? (
              <AnalysisPanel disciplineName={selected.name} />
            ) : (
              <>
                {selected.competencies.map(comp => {
                  const isOpen = expandedComp === comp.code;
                  return (
                    <div key={comp.code} className="border border-gray-200 rounded-lg overflow-hidden">
                      <button
                        onClick={() => setExpandedComp(isOpen ? null : comp.code)}
                        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
                      >
                        <div className="flex items-center gap-2">
                          {isOpen ? <ChevronDown className="size-4 text-gray-400" /> : <ChevronRight className="size-4 text-gray-400" />}
                          <span className="font-mono text-sm font-semibold text-indigo-700">{comp.code}</span>
                          <Badge variant="secondary" className="text-xs">{comp.skills.length} навыков</Badge>
                        </div>
                      </button>

                      {isOpen && (
                        <div className="px-4 py-3 space-y-1">
                          {comp.skills.length === 0 && (
                            <p className="text-sm text-gray-400 italic">Навыки не извлечены</p>
                          )}
                          {comp.skills.map((s, i) => {
                            const tr = skillTrendMap[s];
                            return (
                              <div key={i} className="flex items-center justify-between text-sm py-0.5 border-b border-gray-100 last:border-0">
                                <span className="text-gray-700">{s}</span>
                                {tr ? (
                                  <span className={`font-medium text-xs flex items-center gap-0.5 ${
                                    tr.direction === "rising" ? "text-green-600" :
                                    tr.direction === "falling" ? "text-red-600" : "text-gray-400"
                                  }`}>
                                    {tr.direction === "rising" ? "↑" : tr.direction === "falling" ? "↓" : "→"}
                                    {tr.change_pct > 0 ? "+" : ""}{tr.change_pct}%
                                  </span>
                                ) : (
                                  <span className="text-xs text-gray-300">—</span>
                                )}
                              </div>
                            );
                          })}

                          <div className="border-t border-gray-200 pt-3 mt-3">
                            <Label className="text-xs text-gray-500 mb-1 block">Рекомендация преподавателя</Label>
                            <Textarea
                              placeholder="Предложение по изменению..."
                              value={suggestion}
                              onChange={e => setSuggestion(e.target.value)}
                              rows={2}
                              className="text-sm resize-none"
                            />
                            <div className="flex items-center gap-2 mt-2">
                              <select
                                value={recType}
                                onChange={e => setRecType(e.target.value)}
                                className="h-8 text-xs border border-gray-300 rounded-md px-2 bg-white"
                              >
                                <option value="modify">modify</option>
                                <option value="add">add</option>
                                <option value="remove">remove</option>
                              </select>
                              <Button size="sm" onClick={addRecommendation} disabled={!suggestion.trim()}>
                                <Plus className="size-3 mr-1" />
                                Добавить
                              </Button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}

                {activeRecs.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1">
                      <Lightbulb className="size-4 text-amber-500" />
                      Рекомендации
                    </h4>
                    {activeRecs.map(r => (
                      <div key={r.id} className="flex items-start gap-2 p-3 bg-amber-50 border border-amber-200 rounded-lg mb-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge variant="outline" className="text-xs">{r.type}</Badge>
                            <span className="font-mono text-xs text-gray-500">{r.competency}</span>
                          </div>
                          <p className="text-sm text-gray-700">{r.suggestion}</p>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => deleteRecommendation(r.id)} className="text-red-500 hover:text-red-700 shrink-0">
                          <Trash2 className="size-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
