/**
 * Рендер chart-блока ассистента через recharts (ленивый чанк).
 *
 * Данные уже распознаны чистой функцией `extractChartData`
 * (ChatChartBlock.data.ts) - здесь только отрисовка в общем стиле
 * продукта: оси/сетка/тултип из design/chartTheme, цвета из
 * design/tokens. Подписи осей усечены, чтобы панель не расползалась.
 */

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { axisProps, chartSeries, gridProps, legendProps, tooltipProps } from "../../design";
import type { ChartData } from "./ChatChartBlock.data";

const HEIGHT = 210;

function truncate(value: unknown): string {
  const s = typeof value === "number" ? String(value) : String(value ?? "");
  return s.length > 12 ? `${s.slice(0, 11)}…` : s;
}

const tickStyle = {
  ...axisProps.tick,
  fontSize: 10,
};

export function ChatChartBlock({ data }: { data: ChartData }) {
  const { kind, labelKey, seriesKeys, rows, title } = data;

  const grid = <CartesianGrid {...gridProps} />;
  const xAxis = <XAxis dataKey={labelKey} {...axisProps} tick={tickStyle} tickFormatter={truncate} />;
  const yAxis = <YAxis {...axisProps} tick={tickStyle} width={36} />;

  let chart: React.ReactNode;
  if (kind === "pie") {
    chart = (
      <PieChart margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <Tooltip {...tooltipProps} />
        <Pie
          data={rows}
          dataKey={seriesKeys[0]}
          nameKey={labelKey}
          cx="50%"
          cy="50%"
          innerRadius={30}
          outerRadius={72}
          paddingAngle={2}
        >
          {rows.map((_, i) => (
            <Cell key={i} fill={chartSeries[i % chartSeries.length]} />
          ))}
        </Pie>
        <Legend {...legendProps} />
      </PieChart>
    );
  } else if (kind === "line") {
    chart = (
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        {grid}
        {xAxis}
        {yAxis}
        <Tooltip {...tooltipProps} />
        <Legend {...legendProps} />
        {seriesKeys.map((key, i) => (
          <Line
            key={key}
            dataKey={key}
            stroke={chartSeries[i % chartSeries.length]}
            strokeWidth={2}
            dot={false}
          />
        ))}
      </LineChart>
    );
  } else if (kind === "area") {
    chart = (
      <AreaChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        {grid}
        {xAxis}
        {yAxis}
        <Tooltip {...tooltipProps} />
        <Legend {...legendProps} />
        {seriesKeys.map((key, i) => (
          <Area
            key={key}
            dataKey={key}
            stroke={chartSeries[i % chartSeries.length]}
            fill={chartSeries[i % chartSeries.length]}
            fillOpacity={0.2}
          />
        ))}
      </AreaChart>
    );
  } else {
    chart = (
      <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
        {grid}
        {xAxis}
        {yAxis}
        <Tooltip {...tooltipProps} />
        <Legend {...legendProps} />
        {seriesKeys.map((key, i) => (
          <Bar
            key={key}
            dataKey={key}
            fill={chartSeries[i % chartSeries.length]}
            radius={[3, 3, 0, 0]}
          />
        ))}
      </BarChart>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      {title && <div className="mb-1.5 text-xs font-medium text-foreground">{title}</div>}
      <ResponsiveContainer width="100%" height={HEIGHT}>
        {chart}
      </ResponsiveContainer>
    </div>
  );
}

export { extractChartData } from "./ChatChartBlock.data";
