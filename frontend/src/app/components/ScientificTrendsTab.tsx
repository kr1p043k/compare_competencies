import { useState, useEffect, useRef } from "react";
import {
  type LucideIcon,
  FlaskConical,
  TrendingUp,
  GitCompare,
  Search,
  Sparkles,
  Target,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import {
  Badge,
  Button,
  Card,
  CardBody,
  EmptyState,
  ErrorState,
  PageHeader,
  inputClass,
} from "../../design";
import { useAuth } from "../../lib/auth";
import { reportError } from "./../../lib/errors";
import { request } from "../../lib/api";
import { Label } from "./ui/label";
import { Progress } from "./ui/progress";
import { cn } from "./ui/utils";

interface TrendCompetency {
  code: string;
  description: string;
  keywords: string;
  trend_source: string;
  status?: string;
  coverage_percent?: number;
  skills_count?: number;
  disciplines?: string[];
  near_skills?: GapSkill[];
  missing_topic_skills?: string[];
  suggested_skills?: GapSkill[];
  reason?: string;
  recommendation?: string;
}

interface TrendResponse {
  topic: string;
  found_trends: string[];
  trends_source?: "market_skills" | string;
  recommended_competencies: TrendCompetency[];
  rationale: string;
}

interface GapSkill {
  skill: string;
  similarity: number;
  source?: string;
}

interface GapItem {
  code: string;
  status: string;
  coverage_percent: number;
  reason: string;
  recommendation: string;
  disciplines?: string[];
  skills_count?: number;
  near_skills?: GapSkill[];
  missing_topic_skills?: string[];
  suggested_skills?: GapSkill[];
}

interface GapResponse {
  overall_score: number;
  detailed_analysis: GapItem[];
  summary: string;
}

interface ParsedTrend {
  title: string;
  summary: string;
  keywords: string[];
}

function cleanTrendValue(s: string): string {
  return s.trim().replace(/^["']|["']$/g, "");
}

function parseTrendText(blob: string): ParsedTrend[] {
  const results: ParsedTrend[] = [];
  const blockRe = /\{([^{}]*)\}/g;
  let m: RegExpExecArray | null;
  while ((m = blockRe.exec(blob)) !== null) {
    const body = m[1];
    const titleM = body.match(/title:\s*(.*?)\s*,\s*summary:/i);
    const summaryM = body.match(/summary:\s*(.*?)\s*,\s*keywords:/i);
    const kwM = body.match(/keywords:\s*\[(.*?)\]/i);
    let title = titleM ? cleanTrendValue(titleM[1]) : "";
    if (!title) {
      const first = body.split(",")[0];
      title = cleanTrendValue(first.replace(/^title:\s*/i, ""));
    }
    const summary = summaryM ? cleanTrendValue(summaryM[1]) : "";
    const keywords = kwM
      ? kwM[1].split(",").map((k) => cleanTrendValue(k)).filter(Boolean)
      : [];
    if (title) results.push({ title, summary, keywords });
  }
  return results;
}

function uniqueTrends(list: ParsedTrend[]): ParsedTrend[] {
  const seen = new Set<string>();
  const out: ParsedTrend[] = [];
  for (const t of list) {
    const key = t.title.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      out.push(t);
    }
  }
  return out;
}

function bearerHeaders(): Record<string, string> {
  // Сессия в HttpOnly-cookie: токена в JS больше нет, браузер приложит сам.
  return {};
}

interface AcademicError extends Error {
  status?: number;
}

async function academicCall(path: string, body: unknown): Promise<unknown> {
  const res = await request<Response>(path, { raw: true,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...bearerHeaders(),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `Ошибка (${res.status})`;
    try {
      const j = await res.json();
      if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch (err) { reportError("app/components/ScientificTrendsTab.tsx:135", err); }
    const err = new Error(detail) as AcademicError;
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/** Иконка-квадрат + заголовок раздела внутри Card: тонированная плашка
 *  сверху, как в унаследованных экранах, но только на токенах. */
function SectionHeader({
  icon: Icon,
  squareClass,
  title,
  description,
}: {
  icon: LucideIcon;
  squareClass: string;
  title: string;
  description: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-3 border-b border-border bg-muted px-6 py-6">
      <div
        className={cn(
          "flex size-10 shrink-0 items-center justify-center rounded-lg",
          squareClass,
        )}
      >
        <Icon className="size-5" />
      </div>
      <div>
        <h4 className="text-xl font-semibold leading-none text-foreground">{title}</h4>
        <p className="mt-1 text-sm text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}

export function ScientificTrendsTab() {
  const { role } = useAuth();
  const isStudent = role === "student";
  const [topic, setTopic] = useState("");
  const [loading, setLoading] = useState<"trends" | "gap" | null>(null);
  const [error, setError] = useState("");
  const [trend, setTrend] = useState<TrendResponse | null>(null);
  const [gap, setGap] = useState<GapResponse | null>(null);
  const [krmCount, setKrmCount] = useState<number | null>(null);
  const [gapTopic, setGapTopic] = useState("");
  const gapProgressTimer = useRef<number | null>(null);
  const [gapProgress, setGapProgress] = useState(0);

  useEffect(() => {
    return () => {
      if (gapProgressTimer.current !== null) window.clearInterval(gapProgressTimer.current);
    };
  }, []);

  const startGapProgress = () => {
    setGapProgress(0);
    if (gapProgressTimer.current !== null) window.clearInterval(gapProgressTimer.current);
    gapProgressTimer.current = window.setInterval(() => {
      setGapProgress((p) => {
        // Плавный рост до 90%, пока ждём ответа (сервис ЮФУ не отдаёт реальный прогресс)
        const target = 90;
        if (p >= target) return p;
        const remaining = target - p;
        return Math.min(target, p + Math.max(1, remaining * 0.02));
      });
    }, 1500);
  };

  const stopGapProgress = () => {
    if (gapProgressTimer.current !== null) {
      window.clearInterval(gapProgressTimer.current);
      gapProgressTimer.current = null;
    }
    setGapProgress(0);
  };

  const fetchTrends = async () => {
    if (!topic.trim()) { setError("Введите тему для поиска"); return; }
    setError(""); setLoading("trends");
    try {
      const data = await academicCall("/api/academic/get-competencies", {
        topic: topic.trim(),
        broad_top_k: 10,
      });
      setTrend(data as TrendResponse);
    } catch (e) {
      const err = e as AcademicError;
      setError(err.message);
    } finally {
      setLoading(null);
    }
  };

  const analyzeGap = async () => {
    if (!topic.trim()) { setError("Введите тему для анализа разрыва"); return; }
    setError("");
    try {
      const krm = await request<{ codes: string[]; count: number }>(
        "/api/academic/krm-competencies",
      );
      setKrmCount(krm.count);
      setGapTopic(topic.trim());
      setLoading("gap");
      startGapProgress();
      const data = await academicCall("/api/academic/analyze-gap-local", {
        topic: topic.trim(),
        broad_top_k: 10,
        final_top_k: 5,
      });
      setGap(data as GapResponse);
    } catch (e) {
      const err = e as AcademicError;
      setError(err.message);
    } finally {
      stopGapProgress();
      setLoading(null);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        icon={FlaskConical}
        title="Научные тренды"
        description="Навыки и компетенции из научных публикаций и исследовательских вакансий."
      />
      {/* Запрос */}
      <Card>
        <SectionHeader
          icon={Search}
          squareClass="bg-primary text-primary-foreground"
          title="Академический анализ"
          description="Компетенции и разрывы по научной тематике (локальный анализ)"
        />
        <CardBody className="p-6 space-y-4">
          <div className="space-y-2">
            <Label className="text-sm font-medium text-foreground">Тема / научный запрос</Label>
            <input
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") fetchTrends(); }}
              placeholder="Например: нейросетевые методы обработки изображений"
              className={cn(inputClass, "h-11")}
            />
          </div>
          <div className="flex gap-3 flex-wrap">
            <Button
              onClick={fetchTrends}
              disabled={loading !== null}
              className="h-11"
              iconLeft={loading === "trends"
                ? <span className="mr-2 inline-block size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                : <Sparkles className="size-4 mr-2" />}
            >
              Рекомендуемые компетенции
            </Button>
            <Button
              onClick={analyzeGap}
              disabled={loading !== null || isStudent}
              title={isStudent ? "Анализ разрыва доступен преподавателям" : undefined}
              variant="secondary"
              className="h-11"
              iconLeft={loading === "gap"
                ? <span className="mr-2 inline-block size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                : <Target className="size-4 mr-2" />}
            >
              Анализ разрыва (вся КРМ)
            </Button>
          </div>
          {krmCount !== null && (
            <p className="text-xs text-muted-foreground">
              В анализе разрыва учтено компетенций КРМ: <b>{krmCount}</b>
            </p>
          )}

          {loading === "gap" && (
            <div className="space-y-2">
              <Progress value={gapProgress} className="h-2.5" />
              <p className="text-xs text-muted-foreground">
                Анализ разрыва выполняется, это может занять несколько минут… ({Math.round(gapProgress)}%)
              </p>
            </div>
          )}
          {error && (
            <ErrorState title="Ошибка" description={error} />
          )}
        </CardBody>
      </Card>

      {/* Рекомендуемые компетенции */}
      {trend && (
        <Card>
          <SectionHeader
            icon={TrendingUp}
            squareClass="bg-success-soft text-success"
            title="Рекомендуемые компетенции"
            description={`Тема: ${trend.topic}`}
          />
          <CardBody className="p-6 space-y-5">
            {(() => {
              const trendsList = uniqueTrends(
                trend.found_trends.flatMap((t) => parseTrendText(t))
              );
              if (trendsList.length === 0) return null;
              return (
                <div>
                  <h4 className="text-sm font-semibold text-foreground mb-2">{trend.trends_source === "market_skills" ? "Связанные рыночные навыки (не публикации)" : "Обнаруженные тренды"}</h4>
                  <div className="space-y-3">
                    {trendsList.map((t) => (
                      <div key={t.title} className="rounded-lg border border-border p-4">
                        <p className="text-sm font-semibold text-foreground">{t.title}</p>
                        {t.summary && <p className="text-xs text-muted-foreground mt-1">{t.summary}</p>}
                        {t.keywords.length > 0 && (
                          <div className="flex gap-1.5 flex-wrap mt-2">
                            {t.keywords.map((k) => (
                              <Badge key={k}>{k}</Badge>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {trend.recommended_competencies.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-2">Рекомендуемые компетенции</h4>
                <div className="space-y-3">
                  {(() => {
                    const seenSources = new Set<string>();
                    return trend.recommended_competencies.map((c) => {
                      const cardTrends = uniqueTrends(
                        parseTrendText(c.trend_source || "")
                      ).filter((t) => {
                        const key = t.title.toLowerCase();
                        if (seenSources.has(key)) return false;
                        seenSources.add(key);
                        return true;
                      });
                      return (
                        <div key={c.code} className="rounded-lg border border-border p-4">
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <span className="font-mono text-sm font-semibold text-primary">{c.code}</span>
                            <div className="flex items-center gap-2">
                              {c.coverage_percent !== undefined && (
                                <Badge
                                  tone={
                                    c.coverage_percent >= 80
                                      ? "neutral"
                                      : c.coverage_percent >= 40
                                        ? "primary"
                                        : "danger"
                                  }
                                >
                                  {c.status ?? "gap"} · {c.coverage_percent}%
                                </Badge>
                              )}
                              {c.skills_count !== undefined && (
                                <span className="text-xs text-subtle-foreground">{c.skills_count} навыков</span>
                              )}
                            </div>
                          </div>

                          {c.disciplines && c.disciplines.length > 0 && (
                            <div className="flex items-center gap-1.5 flex-wrap mt-1">
                              {c.disciplines.slice(0, 3).map((d) => (
                                <Badge key={d}>{d}</Badge>
                              ))}
                              {c.disciplines.length > 3 && (
                                <span className="text-xs text-subtle-foreground">+{c.disciplines.length - 3}</span>
                              )}
                            </div>
                          )}

                          {c.near_skills && c.near_skills.length > 0 && (
                            <div className="flex items-center gap-1.5 flex-wrap mt-2">
                              {c.near_skills.map((n) => (
                                <Badge key={n.skill} tone="success">
                                  {n.skill.length > 40 ? n.skill.slice(0, 40) + "…" : n.skill} ({n.similarity.toFixed(2)})
                                </Badge>
                              ))}
                            </div>
                          )}

                          {c.missing_topic_skills && c.missing_topic_skills.length > 0 && (
                            <div className="flex items-center gap-1.5 flex-wrap mt-2">
                              {c.missing_topic_skills.map((m) => (
                                <Badge key={m} tone="warning">
                                  {m.length > 40 ? m.slice(0, 40) + "…" : m}
                                </Badge>
                              ))}
                            </div>
                          )}

                          {c.description && <p className="text-sm text-muted-foreground mt-2">{c.description}</p>}
                          {c.keywords && <p className="text-xs text-muted-foreground mt-1">Ключевые слова: {c.keywords}</p>}
                          {c.reason && !c.description && <p className="text-xs text-muted-foreground mt-1">{c.reason}</p>}

                          {c.suggested_skills && c.suggested_skills.length > 0 && (
                            <div className="mt-2 text-xs text-muted-foreground">
                              <span className="font-medium">Рекомендуемые навыки:</span>
                              <ul className="mt-1 space-y-0.5">
                                {c.suggested_skills.map((s) => (
                                  <li key={s.skill + s.source}>
                                    <span className="font-mono text-primary">{s.skill}</span>{" "}
                                    <span className="text-subtle-foreground">
                                      ({s.similarity.toFixed(2)}
                                      {s.source === "competency" ? ", близок к компетенции" : ", по теме"})
                                    </span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {cardTrends.length > 0 && (
                            <div className="flex gap-1.5 flex-wrap mt-2">
                              {cardTrends.map((t) => (
                                <Badge key={t.title} tone="primary" title={t.summary || t.title}>
                                  {t.title}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    });
                  })()}
                </div>
              </div>
            )}

            {trend.rationale && (
              <div className="rounded-lg border border-primary/20 bg-primary-soft p-4 text-sm text-primary-soft-foreground">
                <span className="font-semibold">Обоснование: </span>{trend.rationale}
              </div>
            )}
          </CardBody>
        </Card>
      )}

      {/* Анализ разрыва */}
      {gap && (
        <Card>
          <SectionHeader
            icon={GitCompare}
            squareClass="bg-destructive/10 text-destructive"
            title="Анализ разрыва компетенций"
            description={`Тема: ${gapTopic}`}
          />
          <CardBody className="p-6 space-y-5">
            <div className="flex items-center gap-3">
              <span className="text-3xl font-bold text-foreground">
                {Math.round(gap.overall_score * 100)}%
              </span>
              <span className="text-sm text-muted-foreground">общее покрытие</span>
            </div>

            {gap.summary && (
              <p className="text-sm text-foreground rounded-lg bg-muted border border-border p-4">
                {gap.summary}
              </p>
            )}

            <div className="space-y-2">
              {gap.detailed_analysis.map((item) => {
                const covered = item.coverage_percent >= 80;
                return (
                  <div key={item.code} className="rounded-lg border border-border p-4">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-sm font-semibold text-foreground">{item.code}</span>
                      <div className="flex items-center gap-2">
                        {covered ? <CheckCircle2 className="size-4 text-success" /> : <AlertCircle className="size-4 text-warning" />}
                        <Badge tone={covered ? "neutral" : "danger"}>
                          {item.status} · {item.coverage_percent}%
                        </Badge>
                      </div>
                    </div>

                    {item.disciplines && item.disciplines.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap mt-2">
                        <span className="text-xs text-subtle-foreground">Дисциплины:</span>
                        {item.disciplines.slice(0, 3).map((d) => (
                          <Badge key={d}>{d}</Badge>
                        ))}
                        {item.disciplines.length > 3 && (
                          <span className="text-xs text-subtle-foreground">+{item.disciplines.length - 3}</span>
                        )}
                      </div>
                    )}

                    {item.reason && <p className="text-xs text-muted-foreground mt-1">{item.reason}</p>}

                    {item.near_skills && item.near_skills.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap mt-2">
                        <span className="text-xs text-subtle-foreground">Близкие к теме:</span>
                        {item.near_skills.map((n) => (
                          <Badge key={n.skill} tone="success">
                            {n.skill.length > 40 ? n.skill.slice(0, 40) + "…" : n.skill} ({n.similarity.toFixed(2)})
                          </Badge>
                        ))}
                      </div>
                    )}

                    {item.missing_topic_skills && item.missing_topic_skills.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap mt-2">
                        <span className="text-xs text-subtle-foreground">Чего не хватает:</span>
                        {item.missing_topic_skills.map((m) => (
                          <Badge key={m} tone="warning">
                            {m.length > 40 ? m.slice(0, 40) + "…" : m}
                          </Badge>
                        ))}
                      </div>
                    )}

                    {item.recommendation && (
                      <p className="text-sm text-primary-soft-foreground mt-2 bg-primary-soft rounded p-2">
                        Рекомендация: {item.recommendation}
                      </p>
                    )}

                    {item.suggested_skills && item.suggested_skills.length > 0 && (
                      <div className="mt-2 text-xs text-muted-foreground">
                        <span className="font-medium">Рекомендуемые навыки:</span>
                        <ul className="mt-1 space-y-0.5">
                          {item.suggested_skills.map((s) => (
                            <li key={s.skill + s.source}>
                              <span className="font-mono text-primary">{s.skill}</span>{" "}
                              <span className="text-subtle-foreground">
                                ({s.similarity.toFixed(2)}
                                {s.source === "competency" ? ", близок к вашим навыкам" : ", по теме"})
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </CardBody>
        </Card>
      )}

      {/* Пустое состояние */}
      {!trend && !gap && !error && (
        <EmptyState
          icon={<TrendingUp className="size-12" />}
          title="Задайте тему"
          description="Анализ вернёт рекомендуемые компетенции по научным трендам и разрыв относительно компетенций КРМ"
        />
      )}
    </div>
  );
}
