import {
  AlertCircle,
  Award,
  Briefcase,
  Clock,
  Database,
  Download,
  FileText,
  Search,
  Sparkles,
  Zap,
} from "lucide-react";

import { Label } from "./ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { DataViewer } from "./DataViewer";
import { MarketView } from "./MarketView";
import { ProfileView } from "./ProfileView";
import { RecommendationsReport } from "./RecommendationsReport";
import { SummaryReport } from "./SummaryReport";
import { Badge } from "../../design";
import { profileLabel } from "../../lib/profiles";

/**
 * Вкладка «Результаты»: выбор профиля, кнопки загрузки данных, gap-анализ,
 * сводка свежести результата и рендер последнего результата
 * (рекомендации / сводка / рынок / профиль / raw JSON).
 *
 * Разрез из App.tsx без изменения верстки; все обработчики приходят
 * пропсами из App, состояние результата живёт в App.
 */
export function DataTab({
  profile,
  profilesList,
  onProfileChange,
  loading,
  isStudent,
  gapRunning,
  gapMsg,
  onLoadRecommendations,
  onLoadSummary,
  onRunGapAnalysis,
  professionsList,
  targetProfession,
  onTargetProfessionChange,
  onCompareWithProfession,
  resultLoadedAt,
  lastResult,
  onDownloadPdf,
}: {
  profile: string;
  profilesList: string[];
  onProfileChange: (p: string) => void;
  loading: boolean;
  isStudent: boolean;
  gapRunning: boolean;
  gapMsg: string;
  onLoadRecommendations: () => void;
  onLoadSummary: () => void;
  onRunGapAnalysis: () => void;
  professionsList: string[];
  targetProfession: string;
  onTargetProfessionChange: (p: string) => void;
  onCompareWithProfession: () => void;
  resultLoadedAt: Record<string, string>;
  lastResult: any;
  onDownloadPdf: () => void;
}) {
  const lastLoadedIso = resultLoadedAt[profile];
  const lastLoadedDate = lastLoadedIso ? new Date(lastLoadedIso) : null;
  const lastLoadedText = !lastLoadedIso
    ? "ещё не подгружались"
    : lastLoadedDate && !isNaN(lastLoadedDate.getTime())
      ? lastLoadedDate.toLocaleString("ru-RU")
      : lastLoadedIso;
  const generatedAt = (lastResult as { generated_at?: string } | null)?.generated_at;
  let staleDays: number | null = null;
  if (typeof generatedAt === "string") {
    const ageMs = Date.now() - new Date(generatedAt).getTime();
    if (!isNaN(ageMs) && ageMs >= 7 * 864e5) {
      staleDays = Math.floor(ageMs / 864e5);
    }
  }

  return (
    <Card className="gap-5 border border-gray-200 dark:border-slate-700 shadow-sm overflow-hidden">
      <CardHeader className="border-b border-gray-200 dark:border-slate-700 bg-gray-50 dark:bg-slate-900">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 bg-emerald-600 rounded-lg">
            <Database className="size-5 text-white" />
          </div>
          <div>
            <CardTitle className="text-xl font-semibold text-gray-900 dark:text-slate-100">
              Результаты
            </CardTitle>
            <CardDescription className="text-sm text-gray-600 dark:text-slate-400">
              Просмотр профилей, рекомендаций и статистики
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-5 space-y-5">
        <div className="space-y-2">
          <Label className="text-sm font-medium text-gray-900 dark:text-slate-100">
            Профиль компетенций
          </Label>
          <Select value={profile} onValueChange={onProfileChange}>
            <SelectTrigger className="h-11 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {profilesList.map((p) => (
                <SelectItem key={p} value={p}>
                  <div className="flex items-center gap-2">
                    <Award className={`size-4 ${p === profile ? "text-emerald-600" : "text-gray-400 dark:text-slate-500"}`} />
                    <span>{profileLabel(p)}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Каждой кнопке - свой тон дизайн-системы (info / success / нейтральный /
            warning), чтобы действия различались с первого взгляда. Оттенки
            берутся на шаг темнее базовых токенов (--info #0284c7 = sky-600,
            --success #059669 = emerald-600, --warning #d97706 = amber-600):
            шаг 700 даёт контраст белого текста >= 4.5:1 (WCAG AA). */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <Button
            onClick={onLoadRecommendations}
            disabled={loading}
            className="h-11 bg-sky-700 text-white hover:bg-sky-800"
          >
            <Search className="mr-2 size-4" />
            Показать сохранённые
          </Button>
          <Button
            onClick={onLoadRecommendations}
            disabled={loading || isStudent}
            title={isStudent ? "Только для преподавателя" : undefined}
            className="h-11 bg-emerald-700 text-white hover:bg-emerald-800"
          >
            <Sparkles className="mr-2 size-4" />
            Рекомендации
          </Button>
          <Button
            onClick={onLoadSummary}
            disabled={loading || isStudent}
            title={isStudent ? "Только для преподавателя" : undefined}
            variant="outline"
            className="h-11"
          >
            <FileText className="mr-2 size-4" />
            Сводка
          </Button>
          <Button
            onClick={onRunGapAnalysis}
            disabled={loading || gapRunning || isStudent}
            title={isStudent ? "Только для преподавателя" : undefined}
            className="h-11 bg-amber-700 text-white hover:bg-amber-800"
          >
            <Zap className="mr-2 size-4" />
            Запустить gap-анализ
          </Button>
        </div>
        {gapRunning && (
          <p className="text-sm text-amber-700 dark:text-amber-300">{gapMsg || "Выполняется..."}</p>
        )}
        {!isStudent && (
        <div className="space-y-2 rounded-lg border border-gray-200 dark:border-slate-700 p-4">
          <Label className="text-sm font-medium text-gray-900 dark:text-slate-100">
            Целевая профессия для сравнения
          </Label>
          <div className="flex flex-col sm:flex-row gap-3">
            <Select value={targetProfession} onValueChange={onTargetProfessionChange}>
              <SelectTrigger className="h-11 flex-1 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600">
                <SelectValue placeholder="Выберите профессию..." />
              </SelectTrigger>
              <SelectContent>
                {professionsList.map((p) => (
                  <SelectItem key={p} value={p}>
                    {p}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              onClick={onCompareWithProfession}
              disabled={loading || !targetProfession}
              className="h-11 w-full sm:w-auto"
            >
              <Briefcase className="mr-2 size-4" />
              Сравнить с профессией
            </Button>
          </div>
        </div>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <Badge
            tone={lastLoadedIso ? "info" : "neutral"}
            className="text-xs"
            title={`Дата последнего сохранённого прогона (gap-анализа) для профиля ${profileLabel(profile)}`}
          >
            <Clock className="size-3 shrink-0" aria-hidden="true" />
            <span>
              Сохранённый прогон [{profileLabel(profile)}]: {lastLoadedText}
            </span>
          </Badge>
          {staleDays !== null && (
            <Badge
              tone="warning"
              className="text-xs"
              title="Результат старше 7 дней: запустите gap-анализ для обновления"
            >
              данные устарели ({staleDays} дн.)
            </Badge>
          )}
        </div>

        {/* Выгрузка одна - серверный PDF; браузерная печать убрана,
            кнопка прижата к правому краю карточки. */}
        <div className="flex flex-wrap items-center justify-end gap-2 no-print">
          <Button
            onClick={onDownloadPdf}
            disabled={isStudent}
            title={isStudent ? "Только для преподавателя" : undefined}
            variant="outline"
            size="sm"
            className="gap-1.5"
          >
            <Download className="size-3.5" />
            Отчёт (PDF)
          </Button>
        </div>

        {lastResult && (() => {
          const d = lastResult as Record<string, unknown>;
          if (d.recommendations || d.closest_roles) {
            return (
              <>
                {(d as any).focus_mode && (d as any).target_profession && (
                  <Card className="border-2 border-violet-200 dark:border-violet-800 bg-violet-50/50 dark:bg-violet-950/20 mb-4">
                    <CardContent className="pt-5">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="p-2 bg-violet-600 rounded-lg">
                          <Briefcase className="size-5 text-white" />
                        </div>
                        <div>
                          <div className="font-bold text-gray-900 dark:text-slate-100">
                            Фокус: {(d as any).target_profession} · профиль {String((d as any).profile || "")}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-slate-400">
                            {((d as any).target_domains || []).join(", ")}
                          </div>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                        <div className="rounded-lg bg-white/70 dark:bg-slate-950/40 p-3">
                          <div className="text-2xl font-bold text-violet-700 dark:text-violet-300">{Number((d as any).profession_coverage || 0).toFixed(1)}%</div>
                          <div className="text-xs text-gray-500 dark:text-slate-400">покрытие профессии</div>
                        </div>
                        <div className="rounded-lg bg-white/70 dark:bg-slate-950/40 p-3">
                          <div className="text-2xl font-bold text-emerald-700 dark:text-emerald-300">{Number((d as any).coverage_strict ?? (d as any).skill_coverage ?? 0).toFixed(1)}%</div>
                          <div className="text-xs text-gray-500 dark:text-slate-400">навыки: {(d as any).skill_strict_has ?? "-"} из {(d as any).skill_strict_total ?? "-"}</div>
                        </div>
                        <div className="rounded-lg bg-white/70 dark:bg-slate-950/40 p-3">
                          <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">{Number((d as any).readiness_score || 0).toFixed(1)}%</div>
                          <div className="text-xs text-gray-500 dark:text-slate-400">готовность</div>
                        </div>
                        <div className="rounded-lg bg-white/70 dark:bg-slate-950/40 p-3">
                          <div className="text-2xl font-bold text-slate-700 dark:text-slate-200">{Number((d as any).domain_coverage_score || 0).toFixed(1)}%</div>
                          <div className="text-xs text-gray-500 dark:text-slate-400">покрытие доменов</div>
                        </div>
                      </div>
                      {(d as any).krm_note && (
                        <p className="mt-3 text-xs text-amber-700 dark:text-amber-300">{String((d as any).krm_note)}</p>
                      )}
                    </CardContent>
                  </Card>
                )}
                <RecommendationsReport data={lastResult as any} />
              </>
            );
          }
          if (d.evaluations && Array.isArray(d.profiles)) {
            return <SummaryReport data={lastResult as any} />;
          }
          if (Array.isArray((d as any).skills) && typeof (d as any).total === "number") {
            return <MarketView data={lastResult as any} />;
          }
          if (Array.isArray((d as any).skills) && typeof (d as any).profile_name === "string") {
            return <ProfileView data={lastResult as any} />;
          }
          const msg = d.message as string | undefined;
          if (msg && (msg.includes("не найдены") || msg.includes("not found"))) {
            return (
              <Card className="border-2 border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/30">
                <CardContent className="text-center py-12">
                  <AlertCircle className="size-12 text-amber-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-200 mb-2">{msg}</h3>
                  <p className="text-sm text-amber-600 mb-4">Запустите gap-анализ для расчёта покрытия</p>
                  <Button
                    onClick={onRunGapAnalysis}
                    disabled={gapRunning || isStudent}
                    title={isStudent ? "Только для преподавателя" : undefined}
                  >
                    <Zap className="size-4 mr-2" />
                    Запустить gap-анализ
                  </Button>
                  {gapRunning && (
                    <p className="text-sm text-amber-700 dark:text-amber-300 mt-3">{gapMsg || "Выполняется..."}</p>
                  )}
                </CardContent>
              </Card>
            );
          }
          return <DataViewer data={lastResult} />;
        })()}
      </CardContent>
    </Card>
  );
}
