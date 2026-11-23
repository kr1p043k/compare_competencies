/**
 * Сравнение двух наборов (по умолчанию 2024 -> 2025) по готовым сводкам анализа.
 *
 * Данные берутся из `/teacher/intake-comparison`: тот же отчёт, что строит
 * `scripts/compare_intakes.py` из двух `_summary.json`. Карточка отвечает на
 * вопрос «что изменилось в плане и в покрытии при переходе на новый набор»:
 * добавленные/убраны/сохранённые дисциплины и дельта среднего покрытия.
 *
 * Пока сводок за оба набора нет, бэкенд отвечает 404 - это штатное состояние
 * (анализ ещё не прогоняли), поэтому показываем пояснение, а не ошибку.
 */
import { useEffect, useState } from "react";

import { api } from "../../api";
import { errMessage } from "../../../lib/errors";
import { covTextClass, plural } from "./helpers";
import type { IntakeComparisonReport } from "./types";

const card = "rounded-lg border border-border bg-card p-3.5 mb-2.5 shadow-xs";

interface IntakeComparisonProps {
  dirCode: string;
  oldYear?: number;
  newYear?: number;
}

/** Доля -> проценты с одним знаком; пусто -> «-». */
function pct(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%`
    : "-";
}

/** Дельта в процентных пунктах со знаком (для «+16.6 п.п.»). */
function pp(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  const v = value * 100;
  return `${v > 0 ? "+" : ""}${v.toFixed(1)} п.п.`;
}

export function IntakeComparison({
  dirCode,
  oldYear = 2024,
  newYear = 2025,
}: IntakeComparisonProps) {
  const [report, setReport] = useState<IntakeComparisonReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError("");
    setReport(null);
    api<IntakeComparisonReport>(
      `/teacher/intake-comparison?dir_code=${encodeURIComponent(dirCode)}&old=${oldYear}&new=${newYear}`,
    )
      .then((data) => {
        if (alive) setReport(data);
      })
      .catch((err: unknown) => {
        // 404 - нет сводок за один из наборов; это не сбой, а состояние данных.
        if (alive) setError(errMessage(err) ?? "Нет сводок анализа за оба набора");
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [dirCode, oldYear, newYear]);

  const title = `Сравнение наборов: ${oldYear} → ${newYear}`;

  if (loading) {
    return (
      <div className={card} aria-busy="true">
        <div className="text-xs font-semibold text-primary mb-2">{title}</div>
        <div className="text-xs text-subtle-foreground">Загрузка сравнения...</div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className={card}>
        <div className="text-xs font-semibold text-primary mb-1">{title}</div>
        <div className="text-xs text-subtle-foreground">
          {error || "Нет данных для сравнения"}
        </div>
      </div>
    );
  }

  const s = report.summary;
  const delta = s.avg_coverage_delta;
  const added = report.added ?? [];
  const removed = report.removed ?? [];
  const kept = report.kept ?? [];

  return (
    <div className={card} id="t-intake-comparison">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-xs font-semibold text-primary">{title}</div>
        <div className="text-[11px] text-subtle-foreground">
          {plural(added.length, "добавлена", "добавлено", "добавлено")}{" "}
          {added.length} · убрано {removed.length}
        </div>
      </div>

      <div
        className="mt-1.5 flex items-baseline justify-between gap-2"
        title="Среднее покрытие плана рынком по дисциплинам, которые попали в анализ"
      >
        <span className="text-xs text-muted-foreground">Среднее покрытие</span>
        <span className="text-xs font-mono">
          {pct(s.avg_coverage_old)} →{" "}
          <span className={`font-bold ${covTextClass(s.avg_coverage_new)}`}>
            {pct(s.avg_coverage_new)}
          </span>{" "}
          <span className={delta >= 0 ? "text-success" : "text-destructive"}>
            ({pp(delta)})
          </span>
        </span>
      </div>

      <div className="mt-1 flex flex-wrap gap-1.5 text-[11px]">
        <span
          className="rounded bg-muted px-1.5 py-0.5 text-muted-foreground"
          title="Дисциплины, которые есть в обоих наборах"
        >
          сохранено {s.kept}
        </span>
        <span
          className="rounded bg-success-soft px-1.5 py-0.5 text-success"
          title="У общих дисциплин покрытие выросло"
        >
          рост {s.improved}
        </span>
        <span
          className="rounded bg-destructive/10 px-1.5 py-0.5 text-destructive"
          title="У общих дисциплин покрытие упало"
        >
          падение {s.worsened}
        </span>
        {s.unchanged > 0 && (
          <span className="rounded bg-muted px-1.5 py-0.5 text-subtle-foreground">
            без изменений {s.unchanged}
          </span>
        )}
      </div>

      {added.length > 0 && (
        <details className="mt-2">
          <summary className="text-xs font-semibold text-primary cursor-pointer select-none">
            Добавлены в {newYear} ({added.length})
          </summary>
          <ul className="mt-1.5 max-h-48 overflow-auto pr-1 space-y-1">
            {added.map((d) => (
              <li key={d.name} className="flex items-baseline justify-between gap-2 text-xs">
                <span className="text-foreground leading-snug">{d.name}</span>
                <span className={`font-mono shrink-0 ${covTextClass(d.coverage_ratio ?? 0)}`}>
                  {pct(d.coverage_ratio)}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {removed.length > 0 && (
        <details className="mt-1.5">
          <summary className="text-xs font-semibold text-primary cursor-pointer select-none">
            Убраны из плана ({removed.length})
          </summary>
          <ul className="mt-1.5 max-h-48 overflow-auto pr-1 space-y-1">
            {removed.map((d) => (
              <li key={d.name} className="flex items-baseline justify-between gap-2 text-xs">
                <span className="text-muted-foreground leading-snug line-through decoration-border-strong">
                  {d.name}
                </span>
                <span className="font-mono shrink-0 text-subtle-foreground">
                  {pct(d.coverage_ratio)}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {kept.length > 0 && (
        <details className="mt-1.5">
          <summary className="text-xs font-semibold text-primary cursor-pointer select-none">
            Изменилось покрытие ({kept.length})
          </summary>
          <ul className="mt-1.5 max-h-48 overflow-auto pr-1 space-y-1">
            {kept.map((d) => (
              <li key={d.name} className="text-xs">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-foreground leading-snug">
                    {d.name}
                    {d.significant_change && (
                      <span
                        className="ml-1 text-warning"
                        title="Изменение покрытия от 5 п.п. и больше"
                      >
                        [!]
                      </span>
                    )}
                  </span>
                  <span className="font-mono shrink-0">
                    {pct(d.coverage_ratio_old)} →{" "}
                    <span className={covTextClass(d.coverage_ratio_new)}>
                      {pct(d.coverage_ratio_new)}
                    </span>{" "}
                    <span className={d.coverage_delta >= 0 ? "text-success" : "text-destructive"}>
                      ({pp(d.coverage_delta)})
                    </span>
                  </span>
                </div>
                {d.old_name && d.old_name !== d.name && (
                  <div className="text-[11px] text-subtle-foreground">
                    было: {d.old_name}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}

      <div className="mt-2 text-[11px] text-subtle-foreground">
        Сравнение по сводкам анализа обоих наборов; прочерк - дисциплина без анализа.
      </div>
    </div>
  );
}

export default IntakeComparison;