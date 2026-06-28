import { useState, useEffect } from "react";
import { apiFetch } from "../../lib/auth";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { AlertCircle, TrendingUp, TrendingDown, Lightbulb, Target } from "lucide-react";
import { CompetencyTree } from "./CompetencyTree";

interface CompetencyCov {
  code: string;
  total_skills: number;
  matched_skills: number;
  coverage: number;
}

interface Rec {
  type: string;
  priority: string;
  message: string;
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
    coverage_level: string;
    top_market_matched_skills: { skill: string; frequency: number; match_type: string }[];
    gaps_in_curriculum: string[];
    emerging_market_skills_not_in_rpd: { skill: string; frequency: number }[];
  };
  competencies: CompetencyCov[];
  recommendations: Rec[];
}

export function AnalysisPanel({ disciplineName, dirCode = "09.03.02" }: { disciplineName: string; dirCode?: string }) {
  const [data, setData] = useState<DisciplineAnalysis | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    apiFetch(`/api/teacher/analysis/${disciplineName}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => setData(d))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [disciplineName, dirCode]);

  if (loading) return (
    <Card className="border border-gray-200 shadow-sm">
      <CardContent className="p-6 text-center text-gray-500 text-sm">Загрузка анализа...</CardContent>
    </Card>
  );
  if (!data) return (
    <Card className="border border-gray-200 shadow-sm">
      <CardContent className="p-6 text-center text-gray-400 text-sm">
        <AlertCircle className="size-8 mx-auto mb-2 opacity-40" />
        Анализ не найден. Запустите teacher analysis через пайплайн.
      </CardContent>
    </Card>
  );

  const { metrics, competencies, recommendations } = data;
  const cov = metrics.coverage_ratio;

  return (
    <div className="space-y-4 mt-6">
      <Card className="border border-gray-200 shadow-sm">
        <CardHeader className="border-b border-gray-200 bg-gray-50 py-3">
          <div className="flex items-center gap-2">
            <Target className="size-4 text-blue-600" />
            <CardTitle className="text-sm font-semibold text-gray-900">Market Coverage Analysis</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div>
              <div className="text-xs text-gray-500">Coverage</div>
              <div className="text-2xl font-bold">
                {(cov * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">RPD Skills</div>
              <div className="text-lg font-semibold text-gray-900">{metrics.total_rpd_skills}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Market Matched</div>
              <div className="text-lg font-semibold text-green-600">{metrics.market_matched}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Gaps</div>
              <div className="text-lg font-semibold text-red-600">{metrics.gaps}</div>
            </div>
          </div>

          {recommendations.length > 0 && (
            <div className="space-y-2 mb-4">
              <div className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Recommendations</div>
              {recommendations.map((r, i) => (
                <div key={i} className="p-3 rounded-lg border text-sm">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant={r.priority === "high" ? "destructive" : r.priority === "medium" ? "default" : "secondary"} className="text-xs">
                      {r.priority}
                    </Badge>
                    <span className="text-xs text-gray-500">{r.type}</span>
                  </div>
                  <div className="text-gray-700">{r.message}</div>
                </div>
              ))}
            </div>
          )}

          {metrics.top_market_matched_skills.length > 0 && (
            <div className="mb-4">
              <div className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">Top Market-Matched Skills</div>
              <div className="flex flex-wrap gap-1.5">
                {metrics.top_market_matched_skills.map((s, i) => {
                  const mtColors: Record<string,string> = {
                    exact: "bg-green-50 text-green-700 border-green-300",
                    fuzzy: "bg-yellow-50 text-yellow-700 border-yellow-300",
                    semantic: "bg-blue-50 text-blue-700 border-blue-300",
                  };
                  const cls = mtColors[s.match_type] || "bg-gray-50 text-gray-500";
                  return (
                    <Badge key={i} variant="outline" className={`${cls} border text-xs`}>
                      {s.skill}
                      <span className="opacity-60 mx-1">×{s.frequency}</span>
                      <span className="text-[10px] opacity-50">{s.match_type}</span>
                    </Badge>
                  );
                })}
              </div>
            </div>
          )}

          {metrics.emerging_market_skills_not_in_rpd.length > 0 && (
            <div className="mb-4">
              <div className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">
                <TrendingUp className="inline size-3 mr-1 text-blue-600" />
                Emerging Market Skills
              </div>
              <div className="flex flex-wrap gap-1.5">
                {metrics.emerging_market_skills_not_in_rpd.map((s, i) => (
                  <Badge key={i} variant="secondary" className="bg-blue-50 text-blue-700 hover:bg-blue-100 border-blue-200">
                    {s.skill} <span className="opacity-50 ml-1">×{s.frequency}</span>
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {metrics.gaps_in_curriculum.length > 0 && (
            <div className="mb-4">
              <div className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">
                <TrendingDown className="inline size-3 mr-1 text-red-600" />
                RPD Skills Not Found on Market
              </div>
              <div className="flex flex-wrap gap-1.5">
                {metrics.gaps_in_curriculum.slice(0, 10).map((g, i) => (
                  <Badge key={i} variant="secondary" className="bg-red-50 text-red-700 hover:bg-red-100 border-red-200">
                    {g.length > 35 ? g.slice(0, 35) + "…" : g}
                  </Badge>
                ))}
                {metrics.gaps_in_curriculum.length > 10 && (
                  <Badge variant="outline" className="text-gray-400">+{metrics.gaps_in_curriculum.length - 10} more</Badge>
                )}
              </div>
            </div>
          )}

          {competencies.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">Per-Competency Coverage</div>
              <CompetencyTree competencies={competencies} />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
