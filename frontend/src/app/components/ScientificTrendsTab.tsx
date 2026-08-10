import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Badge } from "./ui/badge";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
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

interface GapItem {
  code: string;
  status: string;
  coverage_percent: number;
  reason: string;
  recommendation: string;
}

interface GapResponse {
  overall_score: number;
  detailed_analysis: GapItem[];
  summary: string;
}

const HUB_SSO_URL = "https://hub.sfedu.ru/dashboard/go-to-gap-analyzer/";

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
  const [loading, setLoading] = useState<"trends" | "gap" | null>(null);
  const [error, setError] = useState("");
  const [trend, setTrend] = useState<TrendResponse | null>(null);
  const [gap, setGap] = useState<GapResponse | null>(null);
  const [krmCount, setKrmCount] = useState<number | null>(null);
  const [gapTopic, setGapTopic] = useState("");
  const [ssoBlocked, setSsoBlocked] = useState(false);

  const fetchTrends = async () => {
    if (!topic.trim()) { setError("Введите тему для поиска"); return; }
    setError(""); setSsoBlocked(false); setLoading("trends");
    try {
      const data = await academicCall("/api/academic/get-competencies", {
        topic: topic.trim(),
        broad_top_k: 10,
        final_top_k: 5,
      });
      setTrend(data as TrendResponse);
    } catch (e) {
      const err = e as AcademicError;
      if (err.status === 419 || err.status === 403) setSsoBlocked(true);
      setError(err.message);
    } finally {
      setLoading(null);
    }
  };

  const analyzeGap = async () => {
    if (!topic.trim()) { setError("Введите тему для анализа разрыва"); return; }
    setError(""); setSsoBlocked(false);
    try {
      const res = await fetch("/api/academic/krm-competencies", { headers: bearerHeaders() });
      if (!res.ok) throw new Error("Не удалось получить компетенции КРМ");
      const krm = await res.json() as { codes: string[]; count: number };
      setKrmCount(krm.count);
      setGapTopic(topic.trim());
      setLoading("gap");
      const data = await academicCall("/api/academic/analyze-gap", {
        topic: topic.trim(),
        current_competencies: krm.codes.map((code) => ({ code })),
        broad_top_k: 10,
        final_top_k: 5,
      });
      setGap(data as GapResponse);
    } catch (e) {
      const err = e as AcademicError;
      if (err.status === 419 || err.status === 403) setSsoBlocked(true);
      setError(err.message);
    } finally {
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
          {krmCount !== null && (
            <p className="text-xs text-gray-500">
              В анализе разрыва учтено компетенций КРМ: <b>{krmCount}</b>
            </p>
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
              const uniqueTrends = [...new Set(trend.found_trends)];
              if (uniqueTrends.length === 0) return null;
              return (
                <div>
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Найденные научные тренды</h4>
                  <div className="flex gap-2 flex-wrap">
                    {uniqueTrends.map((t) => (
                      <Badge key={t} variant="secondary">{t}</Badge>
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
                      const showSource = Boolean(c.trend_source) && !seenSources.has(c.trend_source);
                      if (c.trend_source) seenSources.add(c.trend_source);
                      return (
                        <div key={c.code} className="rounded-lg border border-gray-200 p-4">
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <span className="font-mono text-sm font-semibold text-indigo-700">{c.code}</span>
                            {showSource && <Badge variant="outline">{c.trend_source}</Badge>}
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
                    {item.reason && <p className="text-xs text-gray-500 mt-1">{item.reason}</p>}
                    {item.recommendation && (
                      <p className="text-sm text-indigo-800 mt-2 bg-indigo-50 rounded p-2">
                        Рекомендация: {item.recommendation}
                      </p>
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
