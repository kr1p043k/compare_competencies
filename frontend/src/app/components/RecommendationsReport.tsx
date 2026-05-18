import { motion } from "motion/react";
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
} from "lucide-react";

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
}

interface RecommendationsReportProps {
  data: RecommendationData;
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
    </motion.div>
  );
}
