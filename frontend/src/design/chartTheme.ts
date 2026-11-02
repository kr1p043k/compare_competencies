/**
 * Единый вид графиков Recharts: сетка, оси, тултип, легенда.
 *
 * Компоненты передают эти объекты в пропсы, чтобы все диаграммы в продукте
 * выглядели одинаково и не дублировали магические числа по файлам.
 */

const AXIS_STYLE = {
  fontSize: 11,
  fill: "var(--muted-foreground)",
} as const;

export const axisProps = {
  tick: AXIS_STYLE,
  tickLine: false,
  axisLine: false,
} as const;

export const gridProps = {
  stroke: "var(--chart-grid)",
  strokeDasharray: "3 6",
  vertical: false,
} as const;

export const tooltipProps = {
  cursor: { fill: "var(--primary)", fillOpacity: 0.06 },
  contentStyle: {
    background: "var(--popover)",
    border: "1px solid var(--border)",
    borderRadius: 12,
    boxShadow: "var(--shadow-md)",
    fontSize: 12,
    padding: "8px 12px",
    color: "var(--popover-foreground)",
  },
  labelStyle: {
    color: "var(--muted-foreground)",
    fontSize: 11,
    marginBottom: 4,
  },
  itemStyle: { color: "var(--popover-foreground)", fontSize: 12 },
} as const;

export const legendProps = {
  wrapperStyle: { fontSize: 12, color: "var(--muted-foreground)" },
  iconType: "circle",
  iconSize: 8,
} as const;

/** Стандартные размеры полотна: экономит место и не «прыгает» при ресайзе. */
export const chartSize = {
  height: 280,
  margin: { top: 8, right: 12, bottom: 4, left: 4 },
} as const;
