import { useState } from "react";
import { request } from "../../lib/api";
import { useApi, errorText } from "../../lib/useApi";
import { AlertCircle, TrendingDown, Target, ChevronDown, ChevronRight, Undo2 } from "lucide-react";
import { Badge, Button, Card, StatCard, type BadgeTone } from "../../design";
import { CompetencyTree } from "./CompetencyTree";
import { ShowMore } from "./ui/show-more";

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
    top_market_matched_skills: { skill: string; matched_as?: string; frequency: number; match_type: string }[];
    gaps_in_curriculum: string[];
    emerging_market_skills_not_in_rpd: { skill: string; frequency: number }[];
  };
  competencies: CompetencyCov[];
  recommendations: Rec[];
  llm_enhanced?: boolean;
  hidden_foundational?: { skill: string; type: string; manual?: boolean }[];
}

/** Тон бейджа приоритета рекомендации. */
function priorityTone(priority: string): BadgeTone {
  if (priority === "high") return "danger";
  if (priority === "medium") return "primary";
  return "neutral";
}

/** Тон бейджа типа совпадения навыка с рынком. */
function matchTone(matchType: string): BadgeTone {
  if (matchType === "exact") return "success";
  if (matchType === "fuzzy") return "warning";
  if (matchType === "semantic") return "info";
  return "neutral";
}

export function AnalysisPanel({ disciplineName, dirCode = "09.03.02" }: { disciplineName: string; dirCode?: string }) {
  const [showHidden, setShowHidden] = useState(false);
  const [expandedGaps, setExpandedGaps] = useState(false);

  const query = useApi<DisciplineAnalysis>(
    `/api/teacher/analysis/${encodeURIComponent(disciplineName)}?dir_code=${encodeURIComponent(dirCode)}`,
    [disciplineName, dirCode],
  );
  const data = query.data;
  const loading = query.loading;

  const reload = () => {
    setExpandedGaps(false);
    query.refetch();
  };

  const markFoundational = async (skill: string) => {
    await request("/api/teacher/krm/foundational", {
      method: "POST",
      body: { skill },
    });
    reload();
  };

  const unmarkFoundational = async (skill: string) => {
    await request(`/api/teacher/krm/foundational/${encodeURIComponent(skill)}`, { method: "DELETE" });
    reload();
  };

  if (loading) return (
    <Card>
      <div className="p-6 text-center text-muted-foreground text-sm">Загрузка анализа...</div>
    </Card>
  );
  if (!data) return (
    <Card>
      <div className="p-6 text-center text-subtle-foreground text-sm">
        <AlertCircle className="size-8 mx-auto mb-2 opacity-40" />
        {query.error && (
          <div className="mb-2 text-destructive">{errorText(query.error)}</div>
        )}
        Анализ не найден. Запустите teacher analysis через пайплайн.
        <div>
          <Button
            className="mt-3"
            onClick={() => window.dispatchEvent(new CustomEvent("run-direction-analysis"))}
          >
            Запустить анализ
          </Button>
        </div>
      </div>
    </Card>
  );

  const { metrics, competencies, recommendations } = data;
  const cov = metrics.coverage_ratio;
  const wcov = metrics.weighted_coverage;
  const scov = metrics.strong_coverage;

  return (
    <div className="space-y-4 mt-6" id="d-panel">
      <Card id="d-cover" className="scroll-mt-2">
        <div className="flex items-center gap-2 px-6 py-3 border-b border-border bg-muted">
          <Target className="size-4 text-primary" />
          <h4 className="m-0 leading-none text-sm font-semibold text-foreground">Анализ покрытия рынком</h4>
        </div>
        <div className="p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            {scov !== undefined && (
              <StatCard label="Сильное покрытие" value={`${(scov * 100).toFixed(1)}%`} tone="success" />
            )}
            <StatCard label="Покрытие с учётом смежных" value={`${(cov * 100).toFixed(1)}%`} />
            {wcov !== undefined && (
              <StatCard label="Взвешенное покрытие" value={`${(wcov * 100).toFixed(1)}%`} />
            )}
            <StatCard label="Навыков в РПД" value={metrics.total_rpd_skills} />
            <StatCard label="Совпало с рынком" value={metrics.market_matched} tone="success" />
            <StatCard label="Пробелы" value={metrics.gaps} tone="danger" />
          </div>

          {recommendations.length > 0 && (
            <div className="space-y-2 mb-4 scroll-mt-2" id="d-recs" >
              <div className="text-xs font-semibold text-foreground uppercase tracking-wider">Рекомендации</div>
              {data.llm_enhanced && (
                <div className="text-xs font-medium text-warning" title="Часть рекомендаций дополнена языковой моделью">
                  Дополнено LLM
                </div>
              )}
              {recommendations.map((r, i) => (
                <div key={i} className="p-3 rounded-lg border text-sm">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge tone={priorityTone(r.priority)} className="text-xs">
                      {r.priority}
                    </Badge>
                    <span className="text-xs text-muted-foreground">{r.type}</span>
                    {r.llm_reason && (
                      <Badge tone="warning" className="text-xs" title="Есть обоснование языковой модели - см. ниже">
                        LLM
                      </Badge>
                    )}
                    {r.skill && r.type === "review_content" && (
                      <button
                        onClick={() => markFoundational(r.skill as string)}
                        title="Пометить фундаментальным - убрать со страницы"
                        className="ml-auto text-xs px-2 py-0.5 rounded border border-border-strong text-muted-foreground hover:text-foreground hover:border-border-strong bg-transparent cursor-pointer transition-colors"
                      >
                        в фундаментальные
                      </button>
                    )}
                  </div>
                  <div className="text-foreground">{r.message}</div>
                  {r.llm_reason && (
                    <details className="mt-2 rounded-md border border-warning/30 bg-warning-soft px-2.5 py-1.5">
                      <summary className="text-xs font-medium text-warning cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded transition-colors duration-200">
                        Обоснование LLM
                      </summary>
                      <div className="text-xs text-warning mt-1 whitespace-pre-line">{r.llm_reason}</div>
                    </details>
                  )}
                </div>
              ))}
              {(data.hidden_foundational?.length || 0) > 0 && (
                <div className="rounded-lg border border-dashed border-border-strong text-sm overflow-hidden">
                  <button
                    onClick={() => setShowHidden(v => !v)}
                    aria-expanded={showHidden}
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/40 bg-transparent border-0 cursor-pointer transition-colors text-left"
                  >
                    {showHidden
                      ? <ChevronDown className="size-3.5 shrink-0" />
                      : <ChevronRight className="size-3.5 shrink-0" />}
                    <span className="font-medium">Скрытые</span>
                    <span className="px-1.5 py-px rounded-full bg-muted text-xs tabular-nums">
                      {data.hidden_foundational!.length}
                    </span>
                  </button>
                  {showHidden && (
                    <div className="px-3 pb-2 pt-1 space-y-0.5 border-t border-dashed border-border">
                      {data.hidden_foundational!.map((h, i) => (
                        <div key={i} className="flex items-center gap-2 py-1 text-xs text-muted-foreground">
                          <span className="flex-1 truncate" title={h.skill}>{h.skill}</span>
                          {h.manual && (
                            <button
                              onClick={() => unmarkFoundational(h.skill)}
                              title="Вернуть в рекомендации"
                              className="shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs text-muted-foreground hover:text-primary hover:bg-primary-soft bg-transparent border-0 cursor-pointer transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
            <div className="mb-4 scroll-mt-2" id="d-top" >
              <div className="text-xs font-semibold text-foreground uppercase tracking-wider mb-2">Топ совпадений с рынком</div>
              <div className="flex flex-wrap gap-1.5">
                {metrics.top_market_matched_skills.map((s, i) => (
                  <Badge
                    key={i}
                    tone={matchTone(s.match_type)}
                    className="text-xs"
                    title={[
                      `Совпало с рыночным навыком${s.matched_as && s.matched_as !== s.skill ? ` «${s.matched_as}»` : ""}`,
                      `тип совпадения: ${s.match_type}`,
                      `вакансий с навыком: ${s.frequency}`,
                    ].join("; ")}
                  >
                    {s.skill}
                    <span className="text-xs opacity-50">{s.match_type}</span>
                    <span className="text-xs opacity-60" title={`Вакансий с этим навыком: ${s.frequency}`}>
                      {s.frequency} вк.
                    </span>
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {metrics.gaps_in_curriculum.length > 0 && (
            <div className="mb-4 scroll-mt-2" id="d-gaps" >
              <div className="text-xs font-semibold text-foreground uppercase tracking-wider mb-2">
                <TrendingDown className="inline size-3 mr-1 text-destructive" />
                Навыки РПД без спроса на рынке
              </div>
              <div className="flex flex-wrap gap-1.5">
                {(expandedGaps ? metrics.gaps_in_curriculum : metrics.gaps_in_curriculum.slice(0, 10)).map((g, i) => (
                  <Badge key={i} tone="danger" className="text-xs">
                    {g.length > 35 && !expandedGaps ? g.slice(0, 35) + "…" : g}
                  </Badge>
                ))}
              </div>
              <ShowMore total={metrics.gaps_in_curriculum.length} shown={10}
                        expanded={expandedGaps} onToggle={() => setExpandedGaps(!expandedGaps)} />
            </div>
          )}

          {competencies.length > 0 && (
            <div id="d-comps" className="scroll-mt-2">
              <div className="text-xs font-semibold text-foreground uppercase tracking-wider mb-2">Покрытие по компетенциям</div>
              <CompetencyTree competencies={competencies} />
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
