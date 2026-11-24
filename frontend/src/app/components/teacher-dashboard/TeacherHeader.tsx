/**
 * Шапка teacher-кабинета: контекст направления в одну строку.
 *
 * Раньше выбор направления, набор, KPI, блок «Данные и РПД» и scope-бейджи
 * лежали стопкой в сайдбаре над списком дисциплин: до первой дисциплины
 * приходилось прокручивать девять рядов контролов. Теперь контекст живёт
 * здесь, служебные операции - в дровере «Данные и РПД».
 */
import { Database, GitCompareArrows, RefreshCw } from "lucide-react";
import { Badge, Button } from "../../../design";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import { covTextClass, plural } from "./helpers";
import type { Direction, DirectionAnalysis, ScopeChain } from "./types";
import { IntakeComparison } from "./IntakeComparison";

interface TeacherHeaderProps {
  directions: Direction[];
  selectedDir: string;
  onDirChange: (dirCode: string) => void;
  intakeYear: 2024 | 2025;
  onYearChange: (year: 2024 | 2025) => void;
  analysis: DirectionAnalysis | null;
  scopeChain: ScopeChain | null;
  stale: boolean;
  onOpenData: () => void;
  onRerun: () => void;
  runLoading: boolean;
}

export function TeacherHeader({
  directions,
  selectedDir,
  onDirChange,
  intakeYear,
  onYearChange,
  analysis,
  scopeChain,
  stale,
  onOpenData,
  onRerun,
  runLoading,
}: TeacherHeaderProps) {
  // Строгий охват рынка + происхождение знаменателя: процент без источника
  // читается как точный, хотя derived/relevance - вычисленный знаменатель.
  const cov = analysis ? analysis.average_market_top_coverage ?? analysis.average_coverage : null;
  const covReliable = analysis?.average_market_top_reliable !== false;
  const src: Record<string, number> = {};
  let unreliable = 0;
  for (const d of analysis?.disciplines ?? []) {
    const s = d.market_top_source || "n/a";
    src[s] = (src[s] || 0) + 1;
    if (d.market_top_reliable === false) unreliable += 1;
  }
  const srcTitle =
    "источник рынка - " +
    Object.entries(src)
      .sort((a, b) => b[1] - a[1])
      .map(([k, n]) => `${k}: ${n}`)
      .join(", ") +
    (unreliable > 0 ? ` · недостоверно (мало навыков): ${unreliable}` : "");

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-border px-4 py-2.5">
      <div className="flex min-w-0 items-center gap-2">
        <span className="whitespace-nowrap text-sm font-semibold text-foreground">Анализ РПД</span>
        <select
          value={selectedDir}
          onChange={(e) => onDirChange(e.target.value)}
          aria-label="Направление подготовки"
          title="Направление подготовки: от него зависят дисциплины и анализ"
          className="h-8 max-w-[320px] rounded-md border border-border bg-card px-2 text-xs text-foreground outline-none"
        >
          {directions.map((d, idx) => (
            <option key={`${d.dir_code}-${idx}`} value={d.dir_code}>
              {d.dir_code} - {d.name}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-1.5" role="group" aria-label="Набор: загрузка данных">
        <span
          className="text-xs text-muted-foreground"
          title="Учебный план какого набора грузить: от него зависят список дисциплин, охваты и весь анализ. Переключение заново запрашивает данные с сервера."
        >
          План:
        </span>
        {([2025, 2024] as const).map((y) => (
          <button
            key={y}
            onClick={() => onYearChange(y)}
            aria-pressed={intakeYear === y}
            title={`Загрузить список дисциплин и анализ по учебному плану набора ${y}`}
            className={
              "cursor-pointer rounded border px-2 py-0.5 text-xs " +
              (intakeYear === y
                ? "border-primary/60 bg-primary-soft text-primary-soft-foreground"
                : "border-border-strong bg-card text-muted-foreground")
            }
          >
            {y}
          </button>
        ))}
        {/* «Что изменится при смене плана»: ответ на вопрос до переключения. */}
        <Popover>
          <PopoverTrigger asChild>
            <button
              className="inline-flex cursor-pointer items-center gap-1 rounded border border-border-strong bg-card px-2 py-0.5 text-xs text-muted-foreground hover:text-foreground"
              title="Сравнить наборы 2024 и 2025: что добавилось, убрано и как изменился охват"
            >
              <GitCompareArrows className="size-3.5" aria-hidden="true" />
              сравнить
            </button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-[420px] p-3">
            <IntakeComparison dirCode={selectedDir} />
          </PopoverContent>
        </Popover>
      </div>

      {analysis && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
          <span className="flex items-baseline gap-1.5" title={srcTitle}>
            <span className="text-muted-foreground">Охват рынка</span>
            <span
              className={`font-bold tabular ${
                covReliable && cov != null ? covTextClass(cov) : "text-muted-foreground"
              }`}
            >
              {covReliable && cov != null ? `${(cov * 100).toFixed(1)}%` : "н/д"}
            </span>
          </span>
          {analysis.average_market_weight_coverage != null && (
            <span
              className="text-subtle-foreground"
              title="Взвешенный охват: те же топ-навыки рынка, но с весом спроса по каждому навыку (см. глоссарий)"
            >
              по спросу: {(analysis.average_market_weight_coverage * 100).toFixed(1)}%
            </span>
          )}
          {analysis.average_quality_coverage != null && (
            <span
              className="text-subtle-foreground"
              title="Качество: средняя уверенность сопоставления навыков РПД с рынком. Навык может совпасть неточно, поэтому качество обычно ниже строгого охвата (см. глоссарий)"
            >
              качество: {(analysis.average_quality_coverage * 100).toFixed(1)}%
            </span>
          )}
          <span className="text-muted-foreground">
            {analysis.total_gaps_across_all}{" "}
            {plural(analysis.total_gaps_across_all, "пробел", "пробела", "пробелов")}
          </span>
          {scopeChain && (
            <span
              className="text-muted-foreground"
              title={`в наборе ${scopeChain.total} · в учёте ${scopeChain.in_scope} · без данных ${scopeChain.excluded_no_data} · исключено методикой ${scopeChain.excluded_manual}`}
            >
              в анализе {scopeChain.analyzed ?? "-"} из {scopeChain.in_scope}
            </span>
          )}
          {stale && (
            <Badge tone="warning" title="Данные устарели: пересчитайте анализ">
              устарело
            </Badge>
          )}
        </div>
      )}

      <div className="ml-auto flex items-center gap-2">
        <Button
          variant="secondary"
          size="sm"
          iconLeft={<Database className="size-3.5" />}
          onClick={onOpenData}
          title="Состав анализа, пересчёт и загрузка РПД"
        >
          Данные и РПД
        </Button>
        <Button
          variant="primary"
          size="sm"
          iconLeft={<RefreshCw className="size-3.5" />}
          loading={runLoading}
          onClick={onRerun}
          title={
            analysis
              ? `Пересчитать анализ: ${analysis.total_disciplines} ${plural(analysis.total_disciplines, "дисциплина", "дисциплины", "дисциплин")}`
              : "Запустить анализ направления"
          }
        >
          {runLoading ? "Пересчёт..." : "Пересчитать"}
        </Button>
      </div>
    </div>
  );
}
