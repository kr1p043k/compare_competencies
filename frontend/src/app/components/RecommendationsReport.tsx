import { motion } from "motion/react";
import { useState } from "react";
import {
  Badge,
  type BadgeTone,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  StatCard,
} from "../../design";
import { ShowMore } from "./ui/show-more";
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
  llm_enhanced?: boolean;
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
    importance_base?: number;
    importance_bonus?: number;
    priority: string;
    category: string;
    why_important: string;
    how_to_learn: string;
    expected_timeframe: string;
    expected_outcome: string;
    is_soft_skill: boolean;
    market_frequency_percent: number;
    llm_reason?: string;
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

/** Тон метрики по качеству совпадения: ≥60 — хорошо, 30–59 — средне, ниже — плохо. */
function scoreTone(score: number): "success" | "warning" | "danger" {
  if (score >= 60) return "success";
  if (score >= 30) return "warning";
  return "danger";
}

/** Тон бейджа приоритета: высокий — опасность, средний — внимание, низкий — акцент. */
const PRIORITY_TONE: Record<string, BadgeTone> = {
  HIGH: "danger",
  MEDIUM: "warning",
  LOW: "primary",
};

function priorityTone(priority: string): BadgeTone {
  return PRIORITY_TONE[(priority || "").toUpperCase()] ?? "neutral";
}

const GAP_STATUS_RU: Record<string, { label: string; tone: BadgeTone }> = {
  missing: { label: "нет", tone: "danger" },
  weak: { label: "слабый", tone: "warning" },
  strong: { label: "сильный", tone: "success" },
};

function GapStatusBadge({ category }: { category: string }) {
  const key = (category || "").toLowerCase();
  const meta = GAP_STATUS_RU[key] || { label: category || "-", tone: "neutral" as BadgeTone };
  return <Badge tone={meta.tone} className="text-xs">{meta.label}</Badge>;
}

function DomainCard({ name, entry }: { name: string; entry: DomainEntry }) {
  const [expanded, setExpanded] = useState(false);
  const skills = entry.required_skills || [];
  const show = expanded ? skills : skills.slice(0, INITIAL_SKILLS);
  const remaining = skills.length - INITIAL_SKILLS;

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 bg-muted">
        <div className="flex items-center gap-3">
          <Layers className="size-5 text-muted-foreground" />
          <h4 className="font-bold text-foreground">{entry.domain || name}</h4>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-muted-foreground">
            ваши <span className={`font-semibold ${entry.user_has > 0 ? "text-success" : "text-destructive"}`}>{entry.user_has}</span>
            <span className="text-subtle-foreground"> из {entry.total_required}</span>
          </span>
          <span title="Покрытие домена: <20% критическое, 20-40% частичное, ≥40% уверенное" className={`font-semibold ${entry.coverage >= 0.4 ? "text-success" : entry.coverage >= 0.2 ? "text-warning" : "text-destructive"}`}>
            {(entry.coverage * 100).toFixed(1)}%
          </span>
        </div>
      </div>
      <div className="px-5 pb-4">
        <div className="flex flex-wrap gap-1.5">
          {show.map((sk) => (
            <Badge key={sk} tone="neutral" className="text-xs">{sk}</Badge>
          ))}
        </div>
        {remaining > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-2 flex items-center gap-1 text-xs text-primary hover:text-primary/80 font-medium cursor-pointer"
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
  // gap = вес − 0) - не показываем дубль, чтобы не вводить в заблуждение.
  const showDemand = Math.abs(gapAvg - demandAvg) > 0.005;
  const gapColor = gapAvg > 0.7 ? "text-destructive" : "text-warning";

  return (
    <div className="border border-border rounded-lg px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <span className="font-medium text-foreground">{entry.skill || skill}</span>
            <GapStatusBadge category={entry.category} />
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-muted-foreground">разрыв: <span className={`font-semibold ${gapColor}`}>{(gapAvg * 100).toFixed(0)}%</span></span>
          {showDemand && (
            <span className="text-muted-foreground">спрос: <span className="font-semibold text-primary">{(demandAvg * 100).toFixed(0)}%</span></span>
          )}
        </div>
      </div>
      {expanded && (
        <div className="mt-3 space-y-2">
          <p className="text-xs text-muted-foreground italic">
              разрыв - насколько навыка не хватает до требуемого (0% = нет разрыва),
              спрос - востребованность навыка на рынке. Ниже - детали по уровням.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs text-muted-foreground">
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
        className="mt-1 flex items-center gap-1 text-xs text-primary hover:text-primary/80 font-medium cursor-pointer"
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
    return <EmptyState title="Пока нет рекомендаций - запустите анализ." />;
  }

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

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<CheckCircle2 className="size-4" />}
          label={<span title="Композитный индекс 0-100. Readiness = 0.45 × market + 0.30 × strong% − 0.25 × weak%: покрытие рынка, доля сильных и слабых навыков">Готовность</span>}
          value={`${data.summary.readiness_score.toFixed(1)}%`}
          hint="Готовность к работе"
          tone={scoreTone(data.summary.readiness_score)}
        />
        <StatCard
          icon={<Target className="size-4" />}
          label={<span title="Средневзвешенная оценка по трём метрикам. Match Score = (Market Coverage + Skill Coverage + Readiness) / 3">Соответствие рынку</span>}
          value={`${data.summary.match_score.toFixed(1)}%`}
          hint="Общее соответствие рынку"
          tone={scoreTone(data.summary.match_score)}
        />
        <StatCard
          icon={<TrendingUp className="size-4" />}
          label={<span title="Доля навыков студента от всех навыков на рынке. Market Coverage = (Σ весов навыков студента) / (Σ весов всех навыков) × 100">Покрытие рынка</span>}
          value={`${data.summary.market_coverage_score.toFixed(1)}%`}
          hint="Покрытие навыков рынка"
          tone={scoreTone(data.summary.market_coverage_score)}
        />
        <StatCard
          icon={<Award className="size-4" />}
          label={<span title="Строгое: доля навыков студента среди рыночных (бинарно, без весов). Взвешенное ниже - с учётом спроса, может быть выше строгого">Навыки профиля (строго)</span>}
          value={data.summary.coverage_strict === undefined ? "-" : `${data.summary.coverage_strict.toFixed(1)}%`}
          hint={`Взвешенное по спросу: ${(data.summary.coverage_weighted ?? data.summary.skill_coverage).toFixed(1)}%`}
          tone={data.summary.coverage_strict === undefined ? "default" : scoreTone(data.summary.coverage_strict)}
        />
      </div>

      {/* Closest Roles */}
      <Card>
        <CardHeader
          title={
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary-soft rounded-lg">
                <Target className="size-5 text-primary" />
              </div>
              Ближайшие роли
            </div>
          }
          description="Роли, которые лучше всего соответствуют вашему профилю. Сходство - похожесть навыков на требования вакансий; полного соответствия не гарантирует."
        />
        <CardBody>
          <div className="space-y-4">
            {data.closest_roles.map((role, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="border border-border rounded-lg p-4"
              >
                <div className="flex items-start justify-between gap-4 mb-3">
                  <h4 className="font-bold text-foreground flex-1">{stripRoleTag(role.role)}</h4>
                  <div className="flex gap-2 flex-shrink-0">
                    {role.cluster_level && (
                      <Badge className="text-xs bg-chart-5/10 text-chart-5">
                        {role.cluster_level}
                      </Badge>
                    )}
                    <Badge tone="primary" className="text-xs">
                      {role.semantic_similarity.toFixed(1)}% сходство
                    </Badge>
                    <Badge tone="success" className="text-xs">
                      {formatCovered(role.skills_covered)}
                    </Badge>
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">
                  {role.coverage_explanation}
                </p>
              </motion.div>
            ))}
          </div>
        </CardBody>
      </Card>

      {/* Recommendations */}
      <Card>
        <CardHeader
          title={
            <div className="flex items-center gap-3">
              <div className="p-2 bg-success-soft rounded-lg">
                <BookOpen className="size-5 text-success" />
              </div>
              Рекомендации по навыкам
            </div>
          }
          description="Топ-10 навыков для изучения, отсортированные по важности"
          actions={
            data.llm_enhanced ? (
              <span className="text-xs font-medium text-warning" title="Часть рекомендаций дополнена языковой моделью">
                Дополнено LLM
              </span>
            ) : undefined
          }
        />
        <CardBody>
          <div className="space-y-4">
            {data.recommendations.slice(0, showAllRecs ? 10 : 3).map((rec, index) => (
              <motion.div
                key={rec.rank}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="border border-border rounded-lg p-5 hover:border-border-strong transition-colors"
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0">
                    <div className="size-10 bg-primary text-primary-foreground rounded-full flex items-center justify-center font-bold print:bg-transparent print:text-primary">
                      {rec.rank}
                    </div>
                  </div>
                  <div className="flex-1 space-y-3">
                    <div className="flex items-start justify-between gap-4">
                      <h4 className="font-bold text-lg text-foreground capitalize">
                        {rec.skill}
                      </h4>
                      <div className="flex gap-2 flex-shrink-0 flex-wrap justify-end">
                        <Badge tone={priorityTone(rec.priority)} className="text-xs">
                          {getPriorityIcon(rec.priority)}
                          {ruPriority(rec.priority)}
                        </Badge>
                        {rec.is_soft_skill && (
                          <Badge className="text-xs bg-chart-5/10 text-chart-5">
                            Софт-скилл
                          </Badge>
                        )}
                        {rec.llm_reason && (
                          <Badge
                            title="Есть обоснование языковой модели - см. ниже"
                            tone="warning"
                            className="text-xs"
                          >
                            LLM
                          </Badge>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span className="flex items-center gap-1" title={`База ${(rec.importance_base ?? rec.importance_score).toFixed(3)} + бонусы ${(rec.importance_bonus ?? 0).toFixed(3)} (тренд, домен, роль). Композитный балл, а не доля вакансий`}>
                        <TrendingUp className="size-4" />
                        {(rec.importance_score * 100).toFixed(1)} балл
                        {(rec.importance_bonus ?? 0) > 0.001 && (
                          <span className="text-xs text-subtle-foreground">
                            (база {((rec.importance_base ?? 0) * 100).toFixed(0)} + бонус {((rec.importance_bonus ?? 0) * 100).toFixed(0)})
                          </span>
                        )}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="size-4" />
                        {rec.expected_timeframe}
                      </span>
                    </div>

                    <div className="bg-primary-soft rounded-lg p-3 border border-primary/20">
                      <p className="text-xs font-semibold text-primary mb-1">
                        💡 Почему важно:
                      </p>
                      <p className="text-sm text-foreground whitespace-pre-line">
                        {rec.why_important}
                      </p>
                    </div>

                    {rec.how_to_learn && (<div className="bg-success-soft rounded-lg p-3 border border-success/25">
                      <p className="text-xs font-semibold text-success mb-1">
                        📚 Как изучать:
                      </p>
                      <p className="text-sm text-foreground">{rec.how_to_learn}</p>
                    </div>)}

                    {rec.expected_outcome && (<div className="bg-chart-5/10 rounded-lg p-3 border border-chart-5/25">
                      <p className="text-xs font-semibold text-chart-5 mb-1 flex items-center gap-1">
                        <ArrowUp className="size-3" />
                        Ожидаемый результат:
                      </p>
                      <p className="text-sm text-foreground">{rec.expected_outcome}</p>
                    </div>)}

                    {rec.llm_reason && (
                      <details className="bg-warning-soft rounded-lg p-3 border border-warning/25">
                        <summary className="text-xs font-semibold text-warning cursor-pointer hover:text-warning/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-warning rounded transition-colors duration-200">
                          Обоснование LLM
                        </summary>
                        <p className="text-sm text-foreground mt-1 whitespace-pre-line">
                          {rec.llm_reason}
                        </p>
                      </details>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
            <ShowMore total={Math.min(10, data.recommendations.length)} shown={3} expanded={showAllRecs} onToggle={() => setShowAllRecs((v) => !v)} />
          </div>
        </CardBody>
      </Card>

      {/* Domain Coverage */}
      {data.domain_coverage && Object.keys(data.domain_coverage).length > 0 && (
        <Card>
          <CardHeader
            title={
              <div className="flex items-center gap-3">
                <div className="p-2 bg-chart-5/10 rounded-lg">
                  <Layers className="size-5 text-chart-5" />
                </div>
                Покрытие доменов
              </div>
            }
            description="Какие домены навыков покрыты вашим профилем"
          />
          <CardBody className="space-y-3">
            {Object.entries(data.domain_coverage).map(([name, entry]) => (
              <motion.div
                key={name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <DomainCard name={name} entry={entry} />
              </motion.div>
            ))}
          </CardBody>
        </Card>
      )}

      {/* Gaps */}
      {data.gaps && Object.keys(data.gaps).length > 0 && (
        <Card>
          <CardHeader
            title={
              <div className="flex items-center gap-3">
                <div className="p-2 bg-destructive/10 rounded-lg">
                  <AlertCircle className="size-5 text-destructive" />
                </div>
                Пробелы
              </div>
            }
            description="Навыки, по которым у вас наибольшие пробелы"
          />
          <CardBody className="space-y-2">
            <div className="flex gap-2 mb-3">
              {[["all", "Все"], ["missing", "Нет"], ["weak", "Слабые"], ["strong", "Сильные"]].map(([v, label]) => (
                <button
                  key={v}
                  onClick={() => setGapFilter(v)}
                  className={`px-3 py-1 text-xs rounded-full border font-medium cursor-pointer ${gapFilter === v ? "bg-primary text-primary-foreground border-primary print:bg-transparent print:text-primary print:border-primary" : "bg-card text-muted-foreground border-border hover:border-primary/60"}`}
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
          </CardBody>
        </Card>
      )}
    </motion.div>
  );
}
