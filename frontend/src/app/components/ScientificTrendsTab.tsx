import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Badge } from "./ui/badge";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { Progress } from "./ui/progress";
import { TrendingUp, GitCompare, Search, Sparkles, Target, AlertCircle, CheckCircle2 } from "lucide-react";

interface TrendCompetency {
  code: string;
  description: string;
  keywords: string;
  trend_source: string;
}

interface TrendResponse {
  topic: string;
  found_trends: string[];
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

const HUB_SSO_URL = "https://hub.sfedu.ru/dashboard/go-to-gap-analyzer/";
const HUB_REDIRECT_DELAY_S = 6;

function bearerHeaders(): Record<string, string> {
  const stored = localStorage.getItem("auth");
  if (stored) {
    try {
      const token = (JSON.parse(stored) as { token?: string }).token;
      if (token) return { Authorization: `Bearer ${token}` };
    } catch {}
  }
  return {};
}

interface AcademicError extends Error {
  status?: number;
}

async function academicCall(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(path, {
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
    } catch {}
    const err = new Error(detail) as AcademicError;
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export function ScientificTrendsTab() {
  const [topic, setTopic] = useState("");
  const [gapSource, setGapSource] = useState<"yufu" | "local">("yufu");
  const [loading, setLoading] = useState<"trends" | "gap" | null>(null);
  const [error, setError] = useState("");
  const [trend, setTrend] = useState<TrendResponse | null>(null);
  const [gap, setGap] = useState<GapResponse | null>(null);
  const [krmCount, setKrmCount] = useState<number | null>(null);
  const [gapTopic, setGapTopic] = useState("");
  const [ssoBlocked, setSsoBlocked] = useState(false);
  const [redirectIn, setRedirectIn] = useState<number | null>(null);
  const redirectTimer = useRef<number | null>(null);
  const ssoBlockedRef = useRef(false);
  const gapProgressTimer = useRef<number | null>(null);
  const [gapProgress, setGapProgress] = useState(0);

  const startHubRedirect = () => {
    if (redirectTimer.current !== null) return;
    setRedirectIn(HUB_REDIRECT_DELAY_S);
    redirectTimer.current = window.setInterval(() => {
      setRedirectIn((sec) => {
        if (sec === null || sec <= 1) {
          if (redirectTimer.current !== null) {
            window.clearInterval(redirectTimer.current);
            redirectTimer.current = null;
          }
          window.location.href = HUB_SSO_URL;
          return 0;
        }
        return sec - 1;
      });
    }, 1000);
  };

  useEffect(() => {
    return () => {
      if (redirectTimer.current !== null) window.clearInterval(redirectTimer.current);
      if (gapProgressTimer.current !== null) window.clearInterval(gapProgressTimer.current);
    };
  }, []);

  const handleSsoBlocked = (message: string) => {
    setError(message);
    if (!ssoBlockedRef.current) {
      ssoBlockedRef.current = true;
      setSsoBlocked(true);
      startHubRedirect();
    }
  };

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
    setError(""); setSsoBlocked(false); ssoBlockedRef.current = false; setLoading("trends");
    try {
      const data = await academicCall("/api/academic/get-competencies", {
        topic: topic.trim(),
        broad_top_k: 10,
        final_top_k: 5,
      });
      setTrend(data as TrendResponse);
    } catch (e) {
      const err = e as AcademicError;
      if (err.status === 419 || err.status === 403) handleSsoBlocked(err.message);
      else setError(err.message);
    } finally {
      setLoading(null);
    }
  };

  const analyzeGap = async () => {
    if (!topic.trim()) { setError("Введите тему для анализа разрыва"); return; }
    setError(""); setSsoBlocked(false); ssoBlockedRef.current = false;
    try {
      const res = await fetch("/api/academic/krm-competencies", { headers: bearerHeaders() });
      if (!res.ok) throw new Error("Не удалось получить компетенции КРМ");
      const krm = await res.json() as { codes: string[]; count: number };
      setKrmCount(krm.count);
      setGapTopic(topic.trim());
      setLoading("gap");
      startGapProgress();
      const data = gapSource === "local"
        ? await academicCall("/api/academic/analyze-gap-local", {
            topic: topic.trim(),
            broad_top_k: 10,
            final_top_k: 5,
          })
        : await academicCall("/api/academic/analyze-gap", {
            topic: topic.trim(),
            current_competencies: krm.codes.map((code) => ({ code })),
            broad_top_k: 10,
            final_top_k: 5,
          });
      setGap(data as GapResponse);
    } catch (e) {
      const err = e as AcademicError;
      if (err.status === 419 || err.status === 403) handleSsoBlocked(err.message);
      else setError(err.message);
    } finally {
      stopGapProgress();
      setLoading(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Запрос */}
      <Card className="border border-gray-200 shadow-sm">
        <CardHeader className="border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 bg-indigo-600 rounded-lg">
              <Search className="size-5 text-white" />
            </div>
            <div>
              <CardTitle className="text-xl font-semibold text-gray-900">
                Академический анализ
              </CardTitle>
              <CardDescription className="text-sm text-gray-600">
                Компетенции и разрывы по научной тематике (сервис ЮФУ)
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-6 space-y-4">
          <div className="space-y-2">
            <Label className="text-sm font-medium text-gray-900">Тема / научный запрос</Label>
            <Input
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") fetchTrends(); }}
              placeholder="Например: нейросетевые методы обработки изображений"
              className="h-11"
            />
          </div>
          <div className="flex gap-3 flex-wrap">
            <Button onClick={fetchTrends} disabled={loading !== null} className="h-11 bg-indigo-600 hover:bg-indigo-700 text-white">
              {loading === "trends"
                ? <span className="mr-2 inline-block size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                : <Sparkles className="size-4 mr-2" />}
              Рекомендуемые компетенции
            </Button>
            <Button onClick={analyzeGap} disabled={loading !== null} variant="outline" className="h-11 border-gray-300 text-gray-700 hover:bg-gray-50">
              {loading === "gap"
                ? <span className="mr-2 inline-block size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                : <Target className="size-4 mr-2" />}
              Анализ разрыва (вся КРМ)
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Источник анализа разрыва:</span>
            <label className="flex items-center gap-1.5 text-xs cursor-pointer">
              <input
                type="radio"
                name="gapSource"
                checked={gapSource === "yufu"}
                onChange={() => setGapSource("yufu")}
                disabled={loading !== null}
                className="accent-indigo-600"
              />
              Сервис ЮФУ
            </label>
            <label className="flex items-center gap-1.5 text-xs cursor-pointer">
              <input
                type="radio"
                name="gapSource"
                checked={gapSource === "local"}
                onChange={() => setGapSource("local")}
                disabled={loading !== null}
                className="accent-indigo-600"
              />
              Собственный (локальный)
            </label>
          </div>
          {krmCount !== null && (
            <p className="text-xs text-gray-500">
              В анализе разрыва учтено компетенций КРМ: <b>{krmCount}</b>
            </p>
          )}

          {loading === "gap" && (
            <div className="space-y-2">
              <Progress value={gapProgress} className="h-2.5 bg-indigo-100 [&>div]:bg-indigo-600" />
              <p className="text-xs text-gray-500">
                Анализ разрыва выполняется, это может занять несколько минут… ({Math.round(gapProgress)}%)
              </p>
            </div>
          )}
          {error && !ssoBlocked && (
            <Alert variant="destructive">
              <AlertCircle className="size-4" />
              <AlertTitle>Ошибка</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {ssoBlocked && (
            <Alert variant="destructive">
              <AlertCircle className="size-4" />
              <AlertTitle>Доступ к сервису ЮФУ истёк</AlertTitle>
              <AlertDescription className="space-y-3">
                <p>{error || "Токен хаба действителен 1 час. Войдите заново через хаб ЮФУ."}</p>
                <p className="text-sm text-red-700">
                  {redirectIn !== null && redirectIn > 0
                    ? `Перенаправление на хаб ЮФУ через ${redirectIn} с…`
                    : "Перенаправление…"}
                </p>
                <Button
                  onClick={() => { window.location.href = HUB_SSO_URL; }}
                  variant="outline"
                  className="border-red-300 text-red-700 hover:bg-red-50"
                >
                  Войти через хаб ЮФУ
                </Button>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Рекомендуемые компетенции */}
      {trend && (
        <Card className="border border-gray-200 shadow-sm">
          <CardHeader className="border-b border-gray-200 bg-gray-50">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 bg-emerald-600 rounded-lg">
                <TrendingUp className="size-5 text-white" />
              </div>
              <div>
                <CardTitle className="text-xl font-semibold text-gray-900">
                  Рекомендуемые компетенции
                </CardTitle>
                <CardDescription className="text-sm text-gray-600">
                  Тема: {trend.topic}
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-6 space-y-5">
            {(() => {
              const trendsList = uniqueTrends(
                trend.found_trends.flatMap((t) => parseTrendText(t))
              );
              if (trendsList.length === 0) return null;
              return (
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Найденные научные тренды</h4>
                  <div className="space-y-3">
                    {trendsList.map((t) => (
                      <div key={t.title} className="rounded-lg border border-gray-200 p-4">
                        <p className="text-sm font-semibold text-gray-900">{t.title}</p>
                        {t.summary && <p className="text-xs text-gray-600 mt-1">{t.summary}</p>}
                        {t.keywords.length > 0 && (
                          <div className="flex gap-1.5 flex-wrap mt-2">
                            {t.keywords.map((k) => (
                              <span
                                key={k}
                                className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600"
                              >
                                {k}
                              </span>
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
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Рекомендуемые компетенции</h4>
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
                        <div key={c.code} className="rounded-lg border border-gray-200 p-4">
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <span className="font-mono text-sm font-semibold text-indigo-700">{c.code}</span>
                            {cardTrends.length > 0 && (
                              <div className="flex gap-1.5 flex-wrap justify-end">
                                {cardTrends.map((t) => (
                                  <span
                                    key={t.title}
                                    title={t.summary || t.title}
                                    className="text-xs px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-100"
                                  >
                                    {t.title}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                          {c.description && <p className="text-sm text-gray-700">{c.description}</p>}
                          {c.keywords && <p className="text-xs text-gray-500 mt-1">Ключевые слова: {c.keywords}</p>}
                        </div>
                      );
                    });
                  })()}
                </div>
              </div>
            )}

            {trend.rationale && (
              <div className="rounded-lg bg-indigo-50 border border-indigo-100 p-4 text-sm text-indigo-900">
                <span className="font-semibold">Обоснование: </span>{trend.rationale}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Анализ разрыва */}
      {gap && (
        <Card className="border border-gray-200 shadow-sm">
          <CardHeader className="border-b border-gray-200 bg-gray-50">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 bg-rose-600 rounded-lg">
                <GitCompare className="size-5 text-white" />
              </div>
              <div>
                <CardTitle className="text-xl font-semibold text-gray-900">
                  Анализ разрыва компетенций
                </CardTitle>
                <CardDescription className="text-sm text-gray-600">
                  Тема: {gapTopic}
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-6 space-y-5">
            <div className="flex items-center gap-3">
              <span className="text-3xl font-bold text-gray-900">
                {Math.round(gap.overall_score * 100)}%
              </span>
              <span className="text-sm text-gray-500">общее покрытие</span>
            </div>

            {gap.summary && (
              <p className="text-sm text-gray-700 rounded-lg bg-gray-50 border border-gray-200 p-4">
                {gap.summary}
              </p>
            )}

            <div className="space-y-2">
              {gap.detailed_analysis.map((item) => {
                const covered = item.coverage_percent >= 80;
                return (
                  <div key={item.code} className="rounded-lg border border-gray-200 p-4">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-sm font-semibold text-gray-800">{item.code}</span>
                      <div className="flex items-center gap-2">
                        {covered ? <CheckCircle2 className="size-4 text-emerald-600" /> : <AlertCircle className="size-4 text-amber-500" />}
                        <Badge variant={covered ? "secondary" : "destructive"}>
                          {item.status} · {item.coverage_percent}%
                        </Badge>
                      </div>
                    </div>

                    {item.disciplines && item.disciplines.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap mt-2">
                        <span className="text-xs text-gray-400">Дисциплины:</span>
                        {item.disciplines.slice(0, 3).map((d) => (
                          <span key={d} className="text-[11px] px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
                            {d}
                          </span>
                        ))}
                        {item.disciplines.length > 3 && (
                          <span className="text-[11px] text-gray-400">+{item.disciplines.length - 3}</span>
                        )}
                      </div>
                    )}

                    {item.reason && <p className="text-xs text-gray-500 mt-1">{item.reason}</p>}

                    {item.near_skills && item.near_skills.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap mt-2">
                        <span className="text-xs text-gray-400">Близкие к теме:</span>
                        {item.near_skills.map((n) => (
                          <span key={n.skill} className="text-[11px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">
                            {n.skill.length > 40 ? n.skill.slice(0, 40) + "…" : n.skill} ({n.similarity.toFixed(2)})
                          </span>
                        ))}
                      </div>
                    )}

                    {item.missing_topic_skills && item.missing_topic_skills.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap mt-2">
                        <span className="text-xs text-gray-400">Чего не хватает:</span>
                        {item.missing_topic_skills.map((m) => (
                          <span key={m} className="text-[11px] px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-100">
                            {m.length > 40 ? m.slice(0, 40) + "…" : m}
                          </span>
                        ))}
                      </div>
                    )}

                    {item.recommendation && (
                      <p className="text-sm text-indigo-800 mt-2 bg-indigo-50 rounded p-2">
                        Рекомендация: {item.recommendation}
                      </p>
                    )}

                    {item.suggested_skills && item.suggested_skills.length > 0 && (
                      <div className="mt-2 text-xs text-gray-500">
                        <span className="font-medium">Рекомендуемые навыки:</span>
                        <ul className="mt-1 space-y-0.5">
                          {item.suggested_skills.map((s) => (
                            <li key={s.skill + s.source}>
                              <span className="font-mono text-indigo-700">{s.skill}</span>{" "}
                              <span className="text-gray-400">
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
          </CardContent>
        </Card>
      )}

      {/* Пустое состояние */}
      {!trend && !gap && !error && (
        <Card className="border border-gray-200 shadow-sm">
          <CardContent className="p-6">
            <div className="flex flex-col items-center justify-center py-12 text-gray-400">
              <TrendingUp className="size-12 mb-4" />
              <p className="text-lg font-medium">Задайте тему</p>
              <p className="text-sm mt-1">
                Сервис вернёт рекомендуемые компетенции по научным трендам и разрыв относительно компетенций КРМ
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
