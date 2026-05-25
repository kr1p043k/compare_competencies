import { motion } from "motion/react";
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
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
  };
  closest_roles: Array<{
    role: string;
    semantic_similarity: number;
    similarity_explanation: string;
    skills_covered: string;
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

function DomainCard({ name, entry }: { name: string; entry: DomainEntry }) {
  const [expanded, setExpanded] = useState(false);
  const skills = entry.required_skills || [];
  const show = expanded ? skills : skills.slice(0, INITIAL_SKILLS);
  const remaining = skills.length - INITIAL_SKILLS;

  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 bg-gradient-to-r from-slate-50 to-white">
        <div className="flex items-center gap-3">
          <Layers className="size-5 text-slate-600" />
          <h4 className="font-bold text-slate-900">{entry.domain || name}</h4>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-slate-600">
            <span className={`font-semibold ${entry.user_has > 0 ? "text-green-600" : "text-red-500"}`}>{entry.user_has}</span>
            <span className="text-slate-400"> / {entry.total_required}</span>
          </span>
          <span className={`font-semibold ${entry.coverage >= 0.3 ? "text-green-600" : entry.coverage >= 0.1 ? "text-orange-500" : "text-red-500"}`}>
            {(entry.coverage * 100).toFixed(1)}%
          </span>
        </div>
      </div>
      <div className="px-5 pb-4">
        <div className="flex flex-wrap gap-1.5">
          {show.map((sk) => (
            <Badge key={sk} variant="outline" className="text-xs bg-white">{sk}</Badge>
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
  const gapColor = gapAvg > 0.7 ? "text-red-600" : gapAvg > 0.4 ? "text-orange-500" : "text-yellow-600";

  return (
    <div className="border border-slate-200 rounded-lg px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <span className="font-medium text-slate-900">{entry.skill || skill}</span>
          <span className="ml-2 text-xs text-slate-400">{entry.category}</span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-slate-500">gap: <span className={`font-semibold ${gapColor}`}>{(gapAvg * 100).toFixed(0)}%</span></span>
          <span className="text-slate-500">demand: <span className="font-semibold text-blue-600">{((entry.demand_j + entry.demand_m + entry.demand_s) / 3 * 100).toFixed(0)}%</span></span>
        </div>
      </div>
      {expanded && (
        <div className="mt-3 space-y-2">
          <p className="text-xs text-slate-500 italic">
            gap — разрыв между текущим уровнем и требуемым (0% = нет разрыва),
            demand — востребованность навыка на рынке,
            user_level — ваш текущий уровень владения,
            importance — общая важность навыка для карьеры
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-slate-600">
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
  const getPriorityColor = (priority: string) => {
    switch (priority.toUpperCase()) {
      case "HIGH":
        return "bg-red-100 text-red-800 dark:bg-red-950/20 dark:text-red-300 border-red-300";
      case "MEDIUM":
        return "bg-orange-100 text-orange-800 dark:bg-orange-950/20 dark:text-orange-300 border-orange-300";
      case "LOW":
        return "bg-blue-100 text-blue-800 dark:bg-blue-950/20 dark:text-blue-300 border-blue-300";
      default:
        return "bg-slate-100 text-slate-800 dark:bg-slate-950/20 dark:text-slate-300 border-slate-300";
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
        <Card className="border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-sky-50 dark:from-blue-950/20 dark:to-sky-950/20">
          <CardHeader className="pb-3">
            <CardDescription className="flex items-center gap-2">
              <Target className="size-4" />
              Match Score
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
              Market Coverage
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
              Skill Coverage
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold ${getScoreColor(data.summary.skill_coverage)}`}>
              {data.summary.skill_coverage.toFixed(1)}%
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
              Покрытие профиля
            </p>
          </CardContent>
        </Card>

        <Card className="border-2 border-orange-200 dark:border-orange-800 bg-gradient-to-br from-orange-50 to-amber-50 dark:from-orange-950/20 dark:to-amber-950/20">
          <CardHeader className="pb-3">
            <CardDescription className="flex items-center gap-2">
              <CheckCircle2 className="size-4" />
              Readiness
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
      </div>

      {/* Closest Roles */}
      <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
        <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <Target className="size-5 text-white" />
            </div>
            Ближайшие роли
          </CardTitle>
          <CardDescription>
            Роли, которые лучше всего соответствуют вашему профилю
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
                  <h4 className="font-bold text-slate-900 dark:text-white flex-1">{role.role}</h4>
                  <div className="flex gap-2 flex-shrink-0">
                    <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-950/20 dark:text-blue-300 border border-blue-300">
                      {role.semantic_similarity.toFixed(1)}% similarity
                    </Badge>
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-950/20 dark:text-green-300 border border-green-300">
                      {role.skills_covered}
                    </Badge>
                  </div>
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                  {role.similarity_explanation}
                </p>
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
            <div className="p-2 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg">
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
            {data.recommendations.slice(0, 10).map((rec, index) => (
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
                            {rec.priority}
                          </span>
                        </Badge>
                        {rec.is_soft_skill && (
                          <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-950/20 dark:text-purple-300 border border-purple-300">
                            Soft Skill
                          </Badge>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-400">
                      <span className="flex items-center gap-1">
                        <TrendingUp className="size-4" />
                        {rec.market_frequency_percent.toFixed(1)}% спрос
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

                    <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-3 border border-green-200 dark:border-green-800">
                      <p className="text-xs font-semibold text-green-900 dark:text-green-100 mb-1">
                        📚 Как изучать:
                      </p>
                      <p className="text-sm text-green-800 dark:text-green-200">{rec.how_to_learn}</p>
                    </div>

                    <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-3 border border-purple-200 dark:border-purple-800">
                      <p className="text-xs font-semibold text-purple-900 dark:text-purple-100 mb-1 flex items-center gap-1">
                        <ArrowUp className="size-3" />
                        Ожидаемый результат:
                      </p>
                      <p className="text-sm text-purple-800 dark:text-purple-200">{rec.expected_outcome}</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Domain Coverage */}
      {data.domain_coverage && Object.keys(data.domain_coverage).length > 0 && (
        <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
          <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
            <CardTitle className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-lg">
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
              Пробелы (Gaps)
            </CardTitle>
            <CardDescription>
              Навыки, по которым у вас наибольшие пробелы
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-2">
            {Object.entries(data.gaps).map(([skill, entry]) => (
              <motion.div
                key={skill}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <GapsCard skill={skill} entry={entry} />
              </motion.div>
            ))}
          </CardContent>
        </Card>
      )}
    </motion.div>
  );
}
