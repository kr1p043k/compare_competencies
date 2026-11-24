/**
 * KPI направления, когда ни одна дисциплина не выбрана.
 *
 * Три карточки вместо россыпи процентов: охват рынка (с парой «по спросу»),
 * пробелы и число дисциплин. Остальные метрики (сильное и взвешенное
 * покрытие) живут в раскрывашке «Все метрики» в AnalysisPanel.
 */
import { StatCard } from "../../../design";
import { plural } from "./helpers";
import type { DirectionAnalysis } from "./types";

interface DirectionKpisProps {
  analysis: DirectionAnalysis;
}

export function DirectionKpis({ analysis }: DirectionKpisProps) {
  const cov = analysis.average_market_top_coverage ?? analysis.average_coverage;
  const reliable = analysis.average_market_top_reliable !== false;
  const wCov = analysis.average_market_weight_coverage;

  return (
    <div className="mb-4 flex flex-wrap gap-3">
      <StatCard
        className="min-w-[220px] flex-1"
        tone={!reliable ? "default" : cov > 0.5 ? "success" : cov > 0.2 ? "warning" : "danger"}
        label="Средний охват рынка"
        value={reliable ? `${(cov * 100).toFixed(1)}%` : "н/д"}
        hint={
          <>
            <span className="font-semibold">
              {reliable ? (analysis.coverage_level || "").toUpperCase() : "мало релевантных навыков"}
            </span>
            {reliable && wCov != null && (
              <span
                className="mt-0.5 block text-subtle-foreground"
                title="Взвешенный охват (по спросу): насколько востребовано то из топа рынка, что реально покрыто. Может быть выше или ниже строгого"
              >
                взвешенный охват: {(wCov * 100).toFixed(1)}%
              </span>
            )}
          </>
        }
      />
      <StatCard
        className="min-w-[150px]"
        label="Пробелы"
        value={analysis.total_gaps_across_all}
        tone="danger"
        hint={`навыков рынка, которые не закрывает ни одна ${plural(analysis.total_disciplines, "дисциплина", "дисциплины", "дисциплин")}`}
      />
      <StatCard
        className="min-w-[150px]"
        label="Дисциплины"
        value={analysis.total_disciplines}
        tone="primary"
        hint="участвуют в анализе"
      />
    </div>
  );
}
