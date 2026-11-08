/**
 * Человекочитаемые мини-бары метрик пробела навыка.
 *
 * Заменяют сырые ключи API (gap_j, demand_m, user_level, ...) на карточках
 * пробелов: подписи «Разрыв (J/M/S)», «Спрос (J/M/S)», «Ваш уровень»,
 * «Важность», «Кластер». Цвета - только дизайн-токены из theme.css.
 * Tooltip «что это» задан атрибутом title и продублирован в aria-label
 * метра, чтобы подсказка была доступна скринридерам и тестам.
 */

import { cn } from "./utils";

export type GapBarTone = "gap" | "demand" | "level" | "importance" | "cluster";

const TONE_FILL: Record<GapBarTone, string> = {
  gap: "bg-destructive",
  demand: "bg-primary",
  level: "bg-success",
  importance: "bg-warning",
  cluster: "bg-chart-5",
};

const TONE_VALUE: Record<GapBarTone, string> = {
  gap: "text-destructive",
  demand: "text-primary",
  level: "text-success",
  importance: "text-warning",
  cluster: "text-chart-5",
};

/** Доля 0..1 -> целые проценты 0..100 (без float-хвостов вида 66.00000001). */
export function barPercent(value: number): number {
  return Math.min(100, Math.max(0, Math.round(value * 100)));
}

const LEVELS = [
  { suffix: "J", caption: "Junior" },
  { suffix: "M", caption: "Middle" },
  { suffix: "S", caption: "Senior" },
] as const;

/** Tooltip «что это» для каждой метрики. */
export const gapMetricTitles = {
  gapGroup: "Разрыв - насколько навыка не хватает до требуемого уровня (0% - разрыва нет)",
  demandGroup: "Спрос - востребованность навыка на рынке (доля вакансий, где он нужен)",
  gapLevel: (caption: string) => `Разрыв на уровне ${caption}: насколько не хватает до требуемого`,
  demandLevel: (caption: string) => `Востребованность на рынке, уровень ${caption}`,
  userLevel: "Ваш текущий уровень владения навыком",
  importance: "Общая важность навыка",
  cluster: "Релевантность навыка кластеру целевых вакансий",
} as const;

export interface MetricBarProps {
  /** Видимая подпись (например «J» или «Ваш уровень»). */
  label: string;
  /** Значение 0..1. */
  value: number;
  /** Tooltip «что это» = доступное имя метра. */
  title: string;
  tone: GapBarTone;
  className?: string;
}

/** Мини-бар одной метрики: подпись + заливка + значение в одну строку. */
export function MetricBar({ label, value, title, tone, className }: MetricBarProps) {
  const v = barPercent(value);
  return (
    <div className={cn("flex min-w-0 items-center gap-2", className)} title={title}>
      <span className="shrink-0 text-xs text-muted-foreground">{label}</span>
      <span
        role="meter"
        aria-label={title}
        aria-valuenow={v}
        aria-valuemin={0}
        aria-valuemax={100}
        className="flex min-w-0 flex-1 items-center gap-1.5"
      >
        <span className="h-1.5 min-w-4 flex-1 overflow-hidden rounded-full bg-muted">
          <span
            className={cn("block h-full rounded-full", TONE_FILL[tone])}
            style={{ width: `${v}%` }}
          />
        </span>
        <span className={cn("shrink-0 text-xs font-semibold tabular", TONE_VALUE[tone])}>{v}%</span>
      </span>
    </div>
  );
}

interface LevelMet {
  suffix: string;
  value: number;
  title: string;
}

/** Группа метрик по грейдам: заголовок с tooltip + три мини-бара J/M/S. */
function MetricLevelGroup({
  label,
  title,
  levels,
  tone,
  className,
}: {
  label: string;
  title: string;
  levels: LevelMet[];
  tone: GapBarTone;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)} title={title}>
      <span className="text-[11px] font-medium text-muted-foreground">{label}</span>
      <div className="mt-1.5 grid grid-cols-3 gap-2">
        {levels.map((l) => (
          <MetricBar key={l.suffix} label={l.suffix} value={l.value} title={l.title} tone={tone} />
        ))}
      </div>
    </div>
  );
}

export interface GapBreakdownFields {
  /** Разрыв по грейдам, доли 0..1. */
  gapJ?: number;
  gapM?: number;
  gapS?: number;
  /** Спрос рынка по грейдам, доли 0..1. */
  demandJ?: number;
  demandM?: number;
  demandS?: number;
  /** Текущий уровень владения, 0..1. */
  userLevel?: number;
  /** Общий вес навыка в рекомендациях, 0..1. */
  importance?: number;
  /** Релевантность кластеру целевых вакансий, 0..1. */
  cluster?: number;
}

function pickLevels(prefix: "gap" | "demand", fields: GapBreakdownFields): LevelMet[] {
  const values = prefix === "gap"
    ? [fields.gapJ, fields.gapM, fields.gapS]
    : [fields.demandJ, fields.demandM, fields.demandS];
  return LEVELS.flatMap(({ suffix, caption }, i) => {
    const value = values[i];
    if (value == null) return [];
    return [{
      suffix,
      value,
      title: prefix === "gap"
        ? gapMetricTitles.gapLevel(caption)
        : gapMetricTitles.demandLevel(caption),
    }];
  });
}

/**
 * Полная расшифровка метрик навыка. Рисует только пришедшие поля,
 * а подпись групп собирается из реально показанных уровней
 * («Разрыв (J/M/S)», либо «Разрыв (M/S)», если J выведен в заголовке строки).
 */
export function GapBreakdown({
  fields,
  className,
}: {
  fields: GapBreakdownFields;
  className?: string;
}) {
  const gaps = pickLevels("gap", fields);
  const demands = pickLevels("demand", fields);
  const hasAny =
    gaps.length > 0 || demands.length > 0 ||
    fields.userLevel != null || fields.importance != null || fields.cluster != null;
  if (!hasAny) return null;

  return (
    <div className={cn("grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2", className)}>
      {gaps.length > 0 && (
        <MetricLevelGroup
          className="sm:col-span-2"
          label={`Разрыв (${gaps.map((l) => l.suffix).join("/")})`}
          title={gapMetricTitles.gapGroup}
          levels={gaps}
          tone="gap"
        />
      )}
      {demands.length > 0 && (
        <MetricLevelGroup
          className="sm:col-span-2"
          label={`Спрос (${demands.map((l) => l.suffix).join("/")})`}
          title={gapMetricTitles.demandGroup}
          levels={demands}
          tone="demand"
        />
      )}
      {fields.userLevel != null && (
        <MetricBar
          label="Ваш уровень"
          value={fields.userLevel}
          title={gapMetricTitles.userLevel}
          tone="level"
        />
      )}
      {fields.importance != null && (
        <MetricBar
          label="Важность"
          value={fields.importance}
          title={gapMetricTitles.importance}
          tone="importance"
        />
      )}
      {fields.cluster != null && (
        <MetricBar
          label="Кластер"
          value={fields.cluster}
          title={gapMetricTitles.cluster}
          tone="cluster"
        />
      )}
    </div>
  );
}
