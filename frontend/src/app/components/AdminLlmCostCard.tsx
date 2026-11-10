/**
 * Затраты LLM (admin): /admin/llm/cost - учёт llm_interactions.cost_usd.
 * Только компоненты src/design, токены, light/dark; честные loading/error/empty.
 */
import { useCallback, useEffect, useState } from "react";

import {
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  Skeleton,
  StatCard,
  TableFrame,
  tableHeadClass,
  tableRowClass,
} from "../../design";
import { api } from "../api";
import { reportError } from "../../lib/errors";

interface ModelCost {
  model: string;
  requests: number;
  cost_usd: number;
}
interface LlmCost {
  total_usd: number;
  requests: number;
  by_model: ModelCost[];
  by_day: { day: string; requests: number; cost_usd: number }[];
  reason: string | null;
}

export function AdminLlmCostCard({ days = 30 }: { days?: number }) {
  const [data, setData] = useState<LlmCost | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api<LlmCost>(`/admin/llm/cost?days=${days}`)
      .then(setData)
      .catch((e) => {
        setError(String((e as Error)?.message ?? e));
        reportError("AdminLlmCostCard", e);
      })
      .finally(() => setLoading(false));
  }, [days]);

  useEffect(load, [load]);

  return (
    <Card>
      <CardHeader
        title="Затраты LLM"
        description={`Расходы по модели llm_interactions за ${days} дн.`}
      />
      <CardBody>
        {loading ? (
          <div role="status" aria-live="polite" className="space-y-2">
            <span className="sr-only">Загрузка затрат LLM...</span>
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        ) : error ? (
          <ErrorState title="Не удалось загрузить затраты LLM" description={error} onRetry={load} />
        ) : !data || (data.requests === 0 && data.by_model.length === 0) ? (
          <EmptyState
            title="Нет данных"
            description={data?.reason ?? "За период обращения к LLM не зафиксированы."}
          />
        ) : (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <StatCard label="Всего, $" value={data.total_usd.toFixed(4)} />
              <StatCard label="Запросов" value={String(data.requests)} />
            </div>
            {data.by_model.length > 0 && (
              <TableFrame>
                <table className="w-full text-sm">
                  <thead>
                    <tr className={tableHeadClass}>
                      <th className="px-3 py-2 text-left font-semibold">Модель</th>
                      <th className="px-3 py-2 text-right font-semibold">Запросов</th>
                      <th className="px-3 py-2 text-right font-semibold">$, total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.by_model.map((m) => (
                      <tr key={m.model} className={tableRowClass}>
                        <td className="px-3 py-2 font-mono text-xs text-foreground">{m.model}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                          {m.requests}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-foreground">
                          {m.cost_usd.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableFrame>
            )}
          </div>
        )}
      </CardBody>
    </Card>
  );
}
