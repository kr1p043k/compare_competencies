/**
 * Дизайн-токены для JS/TS-кода: цвета графиков, семантические стили.
 *
 * CSS-переменные живут в `src/styles/theme.css`. Здесь — только то, что нужно
 * вычислять в рантайме (Recharts принимает цвета строками, а не классами).
 */

export const chartColor = {
  primary: "var(--chart-1)",
  sky: "var(--chart-2)",
  emerald: "var(--chart-3)",
  amber: "var(--chart-4)",
  violet: "var(--chart-5)",
} as const;

/** Упорядоченная палитра для серий: сначала контрастные, потом близкие. */
export const chartSeries = [
  chartColor.primary,
  chartColor.sky,
  chartColor.emerald,
  chartColor.amber,
  chartColor.violet,
] as const;

/** Цвет по знаку изменения: рост — зелёный, падение — красный. */
export function trendColor(value: number): string {
  if (value > 0) return "var(--success)";
  if (value < 0) return "var(--destructive)";
  return "var(--muted-foreground)";
}

/** Тональность бейджа по уровню достоверности (0..100). */
export function confidenceTone(percent: number): "success" | "warning" | "danger" {
  if (percent >= 70) return "success";
  if (percent >= 40) return "warning";
  return "danger";
}
