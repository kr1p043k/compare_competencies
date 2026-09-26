import { motion } from "motion/react";
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { ShowMore } from "./ui/show-more";
import { Badge } from "./ui/badge";
import {
  TrendingUp,
  Target,
  Award,
  Clock,
  BookOpen,
  ArrowUp,
  CheckCircle2,
  AlertCircle,
  Zap,
  ChevronDown,
  ChevronUp,
  Layers,
} from "lucide-react";

interface DomainEntry {
  domain: string;
  required_skills: string[];
  user_has: number;
  total_required: number;
  coverage: number;
  importance: number;
}

interface GapEntry {
  skill: string;
  gap_j: number;
  gap_m: number;
  gap_s: number;
  demand_j: number;
  demand_m: number;
  demand_s: number;
  cluster_relevance: number;
  user_level: number;
  importance: number;
  category: string;
}

interface RecommendationData {
  summary: {
    match_score: number;
    confidence: number;
    market_coverage_score: number;
    skill_coverage: number;
    domain_coverage_score: number;
    readiness_score: number;
    avg_gap: number;
    coverage: number;
    coverage_details: {
      covered_skills_count: number;
      total_market_skills: number;
    };
    market_skill_coverage: number;
    coverage_strict?: number;
    coverage_weighted?: number;
    coverage_strict_scope?: string;
  };
  closest_roles: Array<{
    role: string;
    semantic_similarity: number;
    similarity_explanation: string;
    skills_covered: string;
    cluster_level?: string;
    coverage_percent: number;
    coverage_explanation: string;
  }>;
  recommendations: Array<{
    rank: number;
    skill: string;
    importance_score: number;
    priority: string;
    category: string;
    why_important: string;
    how_to_learn: string;
    expected_timeframe: string;
    expected_outcome: string;
    is_soft_skill: boolean;
    market_frequency_percent: number;
  }>;
  domain_coverage?: Record<string, DomainEntry>;
  gaps?: Record<string, GapEntry>;
}

interface RecommendationsReportProps {
  data: RecommendationData;
}

export const INITIAL_SKILLS = 12;

function formatCovered(s: string): string {
  const m = /^(\d+)\/(\d+)$/.exec((s || "").trim());
  return m ? `знаете ${m[1]} из ${m[2]}` : s;
}

function stripRoleTag(s: string): string {
  return (s || "").replace(/^\[[^\]]+\]\s*/, "");
}

function ruPriority(p: string): string {
  const m: Record<string, string> = { HIGH: "Высокий", MEDIUM: "Средний", LOW: "Низкий" };
  return m[(p || "").toUpperCase()] || p;
}

const GAP_STATUS_RU: Record<string, { label: string; cls: string }> = {
  missing: { label: "нет", cls: "bg-red-100 dark:bg-red-950/30 text-red-800 dark:text-red-200 border-red-300 dark:border-red-700" },
  weak: { label: "слабый", cls: "bg-amber-100 dark:bg-amber-950/30 text-amber-800 dark:text-amber-200 border-amber-300 dark:border-amber-700" },
  strong: { label: "сильный", cls: "bg-green-100 dark:bg-green-950/30 text-green-800 dark:text-green-200 border-green-300 dark:border-green-700" },
};

function GapStatusBadge({ category }: { category: string }) {
  const key = (category || "").toLowerCase();
  const meta = GAP_STATUS_RU[key] || {
    label: category || "–",
    cls: "bg-slate-100 text-slate-600 dark:text-slate-400 border-slate-300 dark:border-slate-600",
  };
  return <Badge className={`${meta.cls} border text-xs`}>{meta.label}</Badge>;
}

function DomainCard({ name, entry }: { name: string; entry: DomainEntry }) {
  const [expanded, setExpanded] = useState(false);
  const skills = entry.required_skills || [];
  const show = expanded ? skills : skills.slice(0, INITIAL_SKILLS);
  const remaining = skills.length - INITIAL_SKILLS;

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 bg-gradient-to-r from-slate-50 to-white dark:from-slate-800 dark:to-slate-900">
        <div className="flex items-center gap-3">
          <Layers className="size-5 text-slate-600 dark:text-slate-400" />
          <h4 className="font-bold text-slate-900 dark:text-slate-100">{entry.domain || name}</h4>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-slate-600 dark:text-slate-400">
            <span className="text-slate-600 dark:text-slate-400">ваши <span className={`font-semibold ${entry.user_has > 0 ? "text-green-600" : "text-red-500"}`}>{entry.user_has}</span></span>
            <span className="text-slate-400"> из {entry.total_required}</span>
          </span>
          <span className={`font-semibold ${entry.coverage >= 0.3 ? "text-green-600" : entry.coverage >= 0.1 ? "text-orange-500" : "text-red-500"}`}>
            {(entry.coverage * 100).toFixed(1)}%
          </span>
        </div>
      </div>
      <div className="px-5 pb-4">
        <div className="flex flex-wrap gap-1.5">
          {show.map((sk) => (
            <Badge key={sk} variant="outline" className="text-xs bg-white dark:bg-slate-950">{sk}</Badge>
          ))}
        </div>
        {remaining > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-2 flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium cursor-pointer"
          >
            {expanded ? (
              <><ChevronUp className="size-3.5" />Свернуть</>
            ) : (
              <><ChevronDown className="size-3.5" />Ещё {remaining} навыков</>
            )}
          </button>
        )}
      </div>
    </div>
  );
}

function GapsCard({ skill, entry }: { skill: string; entry: GapEntry }) {
  const [expanded, setExpanded] = useState(false);
  const gapAvg = (entry.gap_j + entry.gap_m + entry.gap_s) / 3;
  const demandAvg = (entry.demand_j + entry.demand_m + entry.demand_s) / 3;
  // Для незнакомых навыков спрос == разрыву по построению (demand = вес рынка,
  // gap = вес − 0) — не показываем дубль, чтобы не вводить в заблуждение.
  const showDemand = Math.abs(gapAvg - demandAvg) > 0.005;
  const gapColor = gapAvg > 0.7 ? "text-red-600" : gapAvg > 0.4 ? "text-orange-500" : "text-yellow-600";

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <span className="font-medium text-slate-900 dark:text-slate-100">{entry.skill || skill}</span>
            <GapStatusBadge category={entry.category} />
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-slate-500">разрыв: <span className={`font-semibold ${gapColor}`}>{(gapAvg * 100).toFixed(0)}%</span></span>
          {showDemand && (
            <span className="text-slate-500">спрос: <span className="font-semibold text-blue-600">{(demandAvg * 100).toFixed(0)}%</span></span>
          )}
        </div>
      </div>
      {expanded && (
        <div className="mt-3 space-y-2">
          <p className="text-xs text-slate-500 italic">
              разрыв – насколько навыка не хватает до требуемого (0% = нет разрыва),
              спрос – востребованность навыка на рынке. Ниже – детали по уровням.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-slate-600 dark:text-slate-400">
            <div title="Разрыв на уровне Junior">gap_j: {(entry.gap_j * 100).toFixed(0)}%</div>
            <div title="Разрыв на уровне Middle">gap_m: {(entry.gap_m * 100).toFixed(0)}%</div>
            <div title="Разрыв на уровне Senior">gap_s: {(entry.gap_s * 100).toFixed(0)}%</div>
            <div title="Востребованность на уровне Junior">demand_j: {(entry.demand_j * 100).toFixed(0)}%</div>
            <div title="Востребованность на уровне Middle">demand_m: {(entry.demand_m * 100).toFixed(0)}%</div>
            <div title="Востребованность на уровне Senior">demand_s: {(entry.demand_s * 100).toFixed(0)}%</div>
            <div title="Ваш текущий уровень">user_level: {(entry.user_level * 100).toFixed(0)}%</div>
            <div title="Общая важность навыка">importance: {(entry.importance * 100).toFixed(0)}%</div>
          </div>
        </div>
      )}
      <button
        onClick={() => setExpanded(!expanded)}
        className="mt-1 flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium cursor-pointer"
      >
        {expanded ? <><ChevronUp className="size-3" />свернуть</> : <><ChevronDown className="size-3" />подробнее</>}
      </button>
    </div>
  );
}

export function RecommendationsReport({ data }: RecommendationsReportProps) {
  const [gapFilter, setGapFilter] = useState<string>("all");
  const [showAllRecs, setShowAllRecs] = useState(false);
  const [showAllGaps, setShowAllGaps] = useState(false);
  if (!data || !data.summary) {
    return (
      <div className="py-8 text-center text-gray-500 dark:text-slate-400 text-sm">
        <p>No recommendations yet – run the analysis first.</p>
      </div>
    );
  }
  const getPriorityColor = (priority: string) => {
    switch (priority.toUpperCase()) {
      case "HIGH":
        return "bg-red-100 text-red-800 dark:bg-red-950/20 dark:text-red-300 border-red-300 dark:border-red-700";
      case "MEDIUM":
        return "bg-orange-100 text-orange-800 dark:bg-orange-950/20 dark:text-orange-300 border-orange-300 dark:border-orange-700";
      case "LOW":
        return "bg-blue-100 text-blue-800 dark:bg-blue-950/20 dark:text-blue-300 border-blue-300 dark:border-blue-700";
      default:
        return "bg-slate-100 text-slate-800 dark:bg-slate-950/20 dark:text-slate-300 border-slate-300 dark:border-slate-600";
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority.toUpperCase()) {
      case "HIGH":
        return <AlertCircle className="size-4" />;
      case "MEDIUM":
        return <Zap className="size-4" />;
      case "LOW":
        return <CheckCircle2 className="size-4" />;
      default:
        return null;
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600 dark:text-green-400";
    if (score >= 60) return "text-blue-600 dark:text-blue-400";
    if (score >= 40) return "text-orange-600 dark:text-orange-400";
    return "text-red-600 dark:text-red-400";
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="border-2 border-orange-200 dark:border-orange-800 bg-gradient-to-br from-orange-50 to-amber-50 dark:from-orange-950/20 dark:to-amber-950/20">
          <CardHeader className="pb-3">
            <CardDescription className="flex items-center gap-2">
              <CheckCircle2 className="size-4" />
                <span title="Композитный индекс 0–100. Readiness = 0.45 × market + 0.30 × strong% − 0.25 × weak%: покрытие рынка, доля сильных и слабых навыков">Готовность</span>
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold ${getScoreColor(data.summary.readiness_score)}`}>
              {data.summary.readiness_score.toFixed(1)}%
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
              Готовность к работе
            </p>
          </CardContent>
        </Card>

        <Card className="border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-sky-50 dark:from-blue-950/20 dark:to-sky-950/20">
          <CardHeader className="pb-3">
            <CardDescription className="flex items-center gap-2">
              <Target className="size-4" />
                <span title="Средневзвешенная оценка по трём метрикам. Match Score = (Market Coverage + Skill Coverage + Readiness) / 3">Соответствие рынку</span>
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold ${getScoreColor(data.summary.match_score)}`}>
              {data.summary.match_score.toFixed(1)}%
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
              Общее соответствие рынку
            </p>
          </CardContent>
        </Card>

        <Card className="border-2 border-green-200 dark:border-green-800 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20">
          <CardHeader className="pb-3">
            <CardDescription className="flex items-center gap-2">
              <TrendingUp className="size-4" />
                <span title="Доля навыков студента от всех навыков на рынке. Market Coverage = (Σ весов навыков студента) / (Σ весов всех навыков) × 100">Покрытие рынка</span>
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold ${getScoreColor(data.summary.market_coverage_score)}`}>
              {data.summary.market_coverage_score.toFixed(1)}%
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
              Покрытие навыков рынка
            </p>
          </CardContent>
        </Card>

        <Card className="border-2 border-purple-200 dark:border-purple-800 bg-gradient-to-br from-purple-50 to-fuchsia-50 dark:from-purple-950/20 dark:to-fuchsia-950/20">
          <CardHeader className="pb-3">
            <CardDescription className="flex items-center gap-2">
              <Award className="size-4" />
                <span title="Строгое: доля навыков студента среди рыночных (бинарно, без весов). Взвешенное ниже — с учётом спроса, может быть выше строгого">Навыки профиля (строго)</span>
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold ${getScoreColor(data.summary.coverage_strict ?? data.summary.skill_coverage)}`}>
              {(data.summary.coverage_strict ?? data.summary.skill_coverage).toFixed(1)}%
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
              Взвешенное по спросу: {(data.summary.coverage_weighted ?? data.summary.skill_coverage).toFixed(1)}%
            </p>
          </CardContent>
        </Card>

      </div>

      {/* Closest Roles */}
      <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
        <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 dark:from-blue-950/30 to-purple-600 rounded-lg">
              <Target className="size-5 text-white" />
            </div>
            Ближайшие роли
          </CardTitle>
          <CardDescription>
            Роли, которые лучше всего соответствуют вашему профилю. Сходство – похожесть навыков на требования вакансий; полного соответствия не гарантирует.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="space-y-4">
            {data.closest_roles.map((role, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="border-2 border-slate-100 dark:border-slate-800 rounded-xl p-4"
              >
                <div className="flex items-start justify-between gap-4 mb-3">
                  <h4 className="font-bold text-slate-900 dark:text-white flex-1">{stripRoleTag(role.role)}</h4>
                  <div className="flex gap-2 flex-shrink-0">
                    {role.cluster_level && (
                      <Badge className="bg-violet-100 text-violet-800 dark:bg-violet-950/20 dark:text-violet-300 border border-violet-300 dark:border-violet-700">
                        {role.cluster_level}
                      </Badge>
                    )}
                    <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-950/20 dark:text-blue-300 border border-blue-300 dark:border-blue-700">
                      {role.semantic_similarity.toFixed(1)}% сходство
                    </Badge>
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-950/20 dark:text-green-300 border border-green-300 dark:border-green-700">
                      {formatCovered(role.skills_covered)}
                    </Badge>
                  </div>
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  {role.coverage_explanation}
                </p>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recommendations */}
      <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
        <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-green-500 dark:from-green-950/30 to-emerald-600 rounded-lg">
              <BookOpen className="size-5 text-white" />
            </div>
            Рекомендации по навыкам
          </CardTitle>
          <CardDescription>
            Топ-10 навыков для изучения, отсортированные по важности
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="space-y-4">
            {data.recommendations.slice(0, showAllRecs ? 10 : 3).map((rec, index) => (
              <motion.div
                key={rec.rank}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="border-2 border-slate-100 dark:border-slate-800 rounded-xl p-5 hover:border-slate-200 dark:hover:border-slate-700 transition-colors"
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0">
                    <div className="size-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      {rec.rank}
                    </div>
                  </div>
                  <div className="flex-1 space-y-3">
                    <div className="flex items-start justify-between gap-4">
                      <h4 className="font-bold text-lg text-slate-900 dark:text-white capitalize">
                        {rec.skill}
                      </h4>
                      <div className="flex gap-2 flex-shrink-0 flex-wrap justify-end">
                        <Badge className={`${getPriorityColor(rec.priority)} border`}>
                          <span className="flex items-center gap-1">
                            {getPriorityIcon(rec.priority)}
                            {ruPriority(rec.priority)}
                          </span>
                        </Badge>
                        {rec.is_soft_skill && (
                          <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-950/20 dark:text-purple-300 border border-purple-300 dark:border-purple-700">
                            Софт-скилл
                          </Badge>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-400">
                      <span className="flex items-center gap-1" title="Композитный скор важности (gap + спрос + релевантность), а не доля вакансий">
                        <TrendingUp className="size-4" />
                        {rec.market_frequency_percent.toFixed(1)}% важность
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="size-4" />
                        {rec.expected_timeframe}
                      </span>
                    </div>

                    <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
                      <p className="text-xs font-semibold text-blue-900 dark:text-blue-100 mb-1">
                        💡 Почему важно:
                      </p>
                      <p className="text-sm text-blue-800 dark:text-blue-200 whitespace-pre-line">
                        {rec.why_important}
                      </p>
                    </div>

                    {rec.how_to_learn && (<div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-3 border border-green-200 dark:border-green-800">
                      <p className="text-xs font-semibold text-green-900 dark:text-green-100 mb-1">
                        📚 Как изучать:
                      </p>
                      <p className="text-sm text-green-800 dark:text-green-200">{rec.how_to_learn}</p>
                    </div>)}

                    {rec.expected_outcome && (<div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-3 border border-purple-200 dark:border-purple-800">
                      <p className="text-xs font-semibold text-purple-900 dark:text-purple-100 mb-1 flex items-center gap-1">
                        <ArrowUp className="size-3" />
                        Ожидаемый результат:
                      </p>
                      <p className="text-sm text-purple-800 dark:text-purple-200">{rec.expected_outcome}</p>
                    </div>)}
                  </div>
                </div>
              </motion.div>
            ))}
            <ShowMore total={Math.min(10, data.recommendations.length)} shown={3} expanded={showAllRecs} onToggle={() => setShowAllRecs((v) => !v)} />
          </div>
        </CardContent>
      </Card>

      {/* Domain Coverage */}
      {data.domain_coverage && Object.keys(data.domain_coverage).length > 0 && (
        <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
          <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
            <CardTitle className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-indigo-500 dark:from-indigo-950/30 to-blue-600 rounded-lg">
                <Layers className="size-5 text-white" />
              </div>
              Покрытие доменов
            </CardTitle>
            <CardDescription>
              Какие домены навыков покрыты вашим профилем
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-3">
            {Object.entries(data.domain_coverage).map(([name, entry]) => (
              <motion.div
                key={name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <DomainCard name={name} entry={entry} />
              </motion.div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Gaps */}
      {data.gaps && Object.keys(data.gaps).length > 0 && (
        <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
          <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
            <CardTitle className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-rose-500 to-pink-600 rounded-lg">
                <AlertCircle className="size-5 text-white" />
              </div>
              Пробелы
            </CardTitle>
            <CardDescription>
              Навыки, по которым у вас наибольшие пробелы
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-2">
            <div className="flex gap-2 mb-3">
              {[["all", "Все"], ["missing", "Нет"], ["weak", "Слабые"], ["strong", "Сильные"]].map(([v, label]) => (
                <button
                  key={v}
                  onClick={() => setGapFilter(v)}
                  className={`px-3 py-1 text-xs rounded-full border font-medium cursor-pointer ${gapFilter === v ? "bg-blue-600 text-white border-blue-600" : "bg-white dark:bg-slate-950 text-slate-600 dark:text-slate-400 border-slate-300 dark:border-slate-600 hover:border-blue-400"}`}
                >
                  {label}
                </button>
              ))}
            </div>
            {(() => {
              const items = Object.entries(data.gaps)
                .map(([skill, entry]) => ({ skill, entry, avg: (entry.gap_j + entry.gap_m + entry.gap_s) / 3 }))
                .filter((g) => gapFilter === "all" || (g.entry.category || "").toLowerCase() === gapFilter)
                .sort((a, b) => b.avg - a.avg);
              const shown = showAllGaps ? items : items.slice(0, 8);
              return (
                <>
                  {shown.map(({ skill, entry }) => (
                    <motion.div
                      key={skill}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <GapsCard skill={skill} entry={entry} />
                    </motion.div>
                  ))}
                  <ShowMore total={items.length} shown={8} expanded={showAllGaps} onToggle={() => setShowAllGaps((v) => !v)} />
                </>
              );
            })()}
          </CardContent>
        </Card>
      )}
    </motion.div>
  );
}
