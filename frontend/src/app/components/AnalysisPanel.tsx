import { useState, useEffect } from "react";
import { apiFetch } from "../../lib/auth";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { AlertCircle, TrendingUp, TrendingDown, Lightbulb, Target, ChevronDown, ChevronRight, Undo2 } from "lucide-react";
import { CompetencyTree } from "./CompetencyTree";

interface CompetencyCov {
  code: string;
  total_skills: number;
  matched_skills: number;
  coverage: number;
  weighted_coverage?: number;
  strong_coverage?: number;
  matched?: string[];
  gaps?: string[];
}

interface Rec {
  type: string;
  priority: string;
  message: string;
  skill?: string;
  llm_reason?: string;
}

interface DisciplineAnalysis {
  direction: string;
  direction_name: string;
  discipline: string;
  metrics: {
    total_rpd_skills: number;
    market_matched: number;
    gaps: number;
    coverage_ratio: number;
    weighted_coverage?: number;
    strong_coverage?: number;
    coverage_level: string;
    top_market_matched_skills: { skill: string; frequency: number; match_type: string }[];
    gaps_in_curriculum: string[];
    emerging_market_skills_not_in_rpd: { skill: string; frequency: number }[];
  };
  competencies: CompetencyCov[];
  recommendations: Rec[];
  llm_enhanced?: boolean;
  hidden_foundational?: { skill: string; type: string; manual?: boolean }[];
}

export function AnalysisPanel({ disciplineName, dirCode = "09.03.02" }: { disciplineName: string; dirCode?: string }) {
  const [data, setData] = useState<DisciplineAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [showHidden, setShowHidden] = useState(false);

  const reload = () => {
    setLoading(true);
    apiFetch(`/api/teacher/analysis/${encodeURIComponent(disciplineName)}?dir_code=${encodeURIComponent(dirCode)}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => setData(d))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disciplineName, dirCode]);

  const markFoundational = async (skill: string) => {
    await apiFetch("/api/teacher/krm/foundational", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skill }),
    });
    reload();
  };

  const unmarkFoundational = async (skill: string) => {
    await apiFetch(`/api/teacher/krm/foundational/${encodeURIComponent(skill)}`, { method: "DELETE" });
    reload();
  };

  if (loading) return (
    <Card className="border border-gray-200 dark:border-slate-700 shadow-sm">
      <CardContent className="p-6 text-center text-gray-500 dark:text-slate-400 text-sm">Загрузка анализа...</CardContent>
    </Card>
  );
  if (!data) return (
    <Card className="border border-gray-200 dark:border-slate-700 shadow-sm">
      <CardContent className="p-6 text-center text-gray-400 dark:text-slate-500 text-sm">
        <AlertCircle className="size-8 mx-auto mb-2 opacity-40" />
        Анализ не найден. Запустите teacher analysis через пайплайн.
        <div>
          <button
            onClick={() => window.dispatchEvent(new CustomEvent("run-direction-analysis"))}
            className="mt-3 px-4 py-2 text-sm font-medium text-white bg-violet-600 rounded-lg hover:bg-violet-700 cursor-pointer border-0"
          >
            Запустить анализ
          </button>
        </div>
      </CardContent>
    </Card>
  );

  const { metrics, competencies, recommendations } = data;
  const cov = metrics.coverage_ratio;
  const wcov = metrics.weighted_coverage;
  const scov = metrics.strong_coverage;

  return (
    <div className="space-y-4 mt-6" id="d-panel">
      <Card id="d-cover" className="border border-gray-200 dark:border-slate-700 shadow-sm" style={{ scrollMarginTop: 8 }}>
        <CardHeader className="border-b border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-900 py-3">
          <div className="flex items-center gap-2">
            <Target className="size-4 text-blue-600" />
            <CardTitle className="text-sm font-semibold text-gray-900 dark:text-slate-100">Анализ покрытия рынком</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            {scov !== undefined && (
              <div>
                <div className="text-xs text-gray-500 dark:text-slate-400">Сильное покрытие</div>
                <div className="text-2xl font-bold text-emerald-600">
                  {(scov * 100).toFixed(1)}%
                </div>
              </div>
            )}
            <div>
              <div className="text-xs text-gray-500 dark:text-slate-400">Покрытие с учётом смежных</div>
              <div className="text-2xl font-bold">
                {(cov * 100).toFixed(1)}%
              </div>
            </div>
            {wcov !== undefined && (
              <div>
                <div className="text-xs text-gray-500 dark:text-slate-400">Взвешенное покрытие</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-slate-100">
                  {(wcov * 100).toFixed(1)}%
                </div>
              </div>
            )}
            <div>
              <div className="text-xs text-gray-500 dark:text-slate-400">Навыков в РПД</div>
              <div className="text-lg font-semibold text-gray-900 dark:text-slate-100">{metrics.total_rpd_skills}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 dark:text-slate-400">Совпало с рынком</div>
              <div className="text-lg font-semibold text-green-600">{metrics.market_matched}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 dark:text-slate-400">Пробелы</div>
              <div className="text-lg font-semibold text-red-600">{metrics.gaps}</div>
            </div>
          </div>

          {recommendations.length > 0 && (
            <div className="space-y-2 mb-4" id="d-recs" style={{ scrollMarginTop: 8 }}>
              <div className="text-xs font-semibold text-gray-700 dark:text-slate-300 uppercase tracking-wider">Рекомендации</div>
              {data.llm_enhanced && (
                <div className="text-[11px] font-medium text-amber-700 dark:text-amber-300" title="Часть рекомендаций дополнена языковой моделью">
                  Дополнено LLM
                </div>
              )}
              {recommendations.map((r, i) => (
                <div key={i} className="p-3 rounded-lg border text-sm">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant={r.priority === "high" ? "destructive" : r.priority === "medium" ? "default" : "secondary"} className="text-xs">
                      {r.priority}
                    </Badge>
                    <span className="text-xs text-gray-500 dark:text-slate-400">{r.type}</span>
                    {r.llm_reason && (
                      <Badge className="bg-amber-100 text-amber-800 border border-amber-300 dark:bg-amber-950/30 dark:text-amber-200 dark:border-amber-700 text-xs" title="Есть обоснование языковой модели — см. ниже">
                        LLM
                      </Badge>
                    )}
                    {r.skill && r.type === "review_content" && (
                      <button
                        onClick={() => markFoundational(r.skill as string)}
                        title="Пометить фундаментальным – убрать со страницы"
                        className="ml-auto text-[11px] px-2 py-0.5 rounded border border-gray-300 dark:border-slate-600 text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-100 hover:border-gray-400 dark:hover:border-slate-500 bg-transparent cursor-pointer transition-colors"
                      >
                        в фундаментальные
                      </button>
                    )}
                  </div>
                  <div className="text-gray-700 dark:text-slate-300">{r.message}</div>
                  {r.llm_reason && (
                    <details className="mt-2 rounded-md border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/20 px-2.5 py-1.5">
                      <summary className="text-xs font-medium text-amber-900 dark:text-amber-100 cursor-pointer hover:text-amber-700 dark:hover:text-amber-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 rounded transition-colors duration-200">
                        Обоснование LLM
                      </summary>
                      <div className="text-xs text-amber-800 dark:text-amber-200 mt-1 whitespace-pre-line">{r.llm_reason}</div>
                    </details>
                  )}
                </div>
              ))}
              {(data.hidden_foundational?.length || 0) > 0 && (
                <div className="rounded-lg border border-dashed border-gray-300 dark:border-slate-600 text-sm overflow-hidden">
                  <button
                    onClick={() => setShowHidden(v => !v)}
                    aria-expanded={showHidden}
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-500 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-100 hover:bg-gray-50 dark:hover:bg-slate-800/60 bg-transparent border-0 cursor-pointer transition-colors text-left"
                  >
                    {showHidden
                      ? <ChevronDown className="size-3.5 shrink-0" />
                      : <ChevronRight className="size-3.5 shrink-0" />}
                    <span className="font-medium">Скрытые</span>
                    <span className="px-1.5 py-px rounded-full bg-gray-100 dark:bg-slate-800 text-[11px] tabular-nums">
                      {data.hidden_foundational!.length}
                    </span>
                  </button>
                  {showHidden && (
                    <div className="px-3 pb-2 pt-1 space-y-0.5 border-t border-dashed border-gray-300 dark:border-slate-600">
                      {data.hidden_foundational!.map((h, i) => (
                        <div key={i} className="flex items-center gap-2 py-1 text-xs text-gray-600 dark:text-slate-400">
                          <span className="flex-1 truncate" title={h.skill}>{h.skill}</span>
                          {h.manual && (
                            <button
                              onClick={() => unmarkFoundational(h.skill)}
                              title="Вернуть в рекомендации"
                              className="shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] text-gray-500 dark:text-slate-400 hover:text-violet-700 dark:hover:text-violet-300 hover:bg-violet-50 dark:hover:bg-violet-950/40 bg-transparent border-0 cursor-pointer transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500"
                            >
                              <Undo2 className="size-3" />
                              Вернуть
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {metrics.top_market_matched_skills.length > 0 && (
            <div className="mb-4" id="d-top" style={{ scrollMarginTop: 8 }}>
              <div className="text-xs font-semibold text-gray-700 dark:text-slate-300 uppercase tracking-wider mb-2">Топ совпадений с рынком</div>
              <div className="flex flex-wrap gap-1.5">
                {metrics.top_market_matched_skills.map((s, i) => {
                  const mtColors: Record<string,string> = {
                    exact: "bg-green-50 dark:bg-green-950/30 text-green-700 dark:text-green-300 border-green-300 dark:border-green-700",
                    fuzzy: "bg-yellow-50 dark:bg-yellow-950/30 text-yellow-700 dark:text-yellow-300 border-yellow-300 dark:border-yellow-700",
                    semantic: "bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-300 border-blue-300 dark:border-blue-700",
                  };
                  const cls = mtColors[s.match_type] || "bg-gray-50 dark:bg-slate-900 text-gray-500 dark:text-slate-400";
                  return (
                    <Badge key={i} variant="outline" className={`${cls} border text-xs`}>
                      {s.skill}
                      <span className="text-[10px] opacity-50">{s.match_type}</span>
                    </Badge>
                  );
                })}
              </div>
            </div>
          )}

          {metrics.gaps_in_curriculum.length > 0 && (
            <div className="mb-4" id="d-gaps" style={{ scrollMarginTop: 8 }}>
              <div className="text-xs font-semibold text-gray-700 dark:text-slate-300 uppercase tracking-wider mb-2">
                <TrendingDown className="inline size-3 mr-1 text-red-600" />
                Навыки РПД без спроса на рынке
              </div>
              <div className="flex flex-wrap gap-1.5">
                {metrics.gaps_in_curriculum.slice(0, 10).map((g, i) => (
                  <Badge key={i} variant="secondary" className="bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-950/30 border-red-200 dark:border-red-800">
                    {g.length > 35 ? g.slice(0, 35) + "…" : g}
                  </Badge>
                ))}
                {metrics.gaps_in_curriculum.length > 10 && (
                  <Badge variant="outline" className="text-gray-400 dark:text-slate-500">+{metrics.gaps_in_curriculum.length - 10} ещё</Badge>
                )}
              </div>
            </div>
          )}

          {competencies.length > 0 && (
            <div id="d-comps" style={{ scrollMarginTop: 8 }}>
              <div className="text-xs font-semibold text-gray-700 dark:text-slate-300 uppercase tracking-wider mb-2">Покрытие по компетенциям</div>
              <CompetencyTree competencies={competencies} />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
