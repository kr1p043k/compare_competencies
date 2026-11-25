import { useState } from "react";
import { Activity } from "lucide-react";

import { request } from "../../lib/api";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  StatCard,
  TableFrame,
  tableHeadClass,
  tableRowClass,
} from "../../design";

import { AdminLlmCostCard } from "./AdminLlmCostCard";

/** Вкладка «Проверка» админки: стоимость LLM и состояние сервиса. */
/** Ответ /health: готовность зависимостей сервиса. */
interface HealthPayload {
  status?: string;
  version?: string;
  api_ready?: boolean;
  evaluator?: boolean;
  recommendation_engine?: boolean;
  ltr_model_version?: string | null;
  clusters_versions?: Record<string, string | null>;
}

export function AdminHealthTab() {
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthMsg, setHealthMsg] = useState("");

  const loadHealth = async () => {
    setHealthLoading(true);
    setHealthMsg("");
    try {
      const d = await request<HealthPayload>("/health");
      setHealth(d);
      setHealthMsg("Сервис отвечает");
    } catch (e) {
      setHealthMsg((e as Error)?.message || "Ошибка проверки");
    } finally {
      setHealthLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <AdminLlmCostCard />
      <Card>
        <CardHeader
              headingLevel={2}
          title={
            <span className="flex items-center gap-2">
              <Activity className="size-4" aria-hidden="true" />
              Состояние сервиса
            </span>
          }
        />
        <CardBody className="space-y-3">
          <Button variant="secondary" onClick={loadHealth} disabled={healthLoading}>
            {healthLoading ? "Проверяю..." : "Проверить состояние"}
          </Button>
          {healthMsg && <span className="text-sm text-muted-foreground">{healthMsg}</span>}
          {health && (
            <div className="space-y-3">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                <StatCard label="Статус" value={<Badge tone={health.status === "ok" ? "success" : "danger"}>{String(health.status ?? "-")}</Badge>} />
                <StatCard label="Версия" value={String(health.version ?? "-")} />
                <StatCard label="API (api_ready)" value={<Badge tone={health.api_ready ? "success" : "warning"}>{health.api_ready ? "готов" : "не готов"}</Badge>} />
                <StatCard label="Оценщик (evaluator)" value={<Badge tone={health.evaluator ? "success" : "warning"}>{health.evaluator ? "готов" : "не готов"}</Badge>} />
                <StatCard label="Рекомендации (recommendation_engine)" value={<Badge tone={health.recommendation_engine ? "success" : "warning"}>{health.recommendation_engine ? "готов" : "не готов"}</Badge>} />
                <StatCard label="Модель LTR (ltr_model_version)" value={String(health.ltr_model_version ?? "-")} />
              </div>
              <div>
                <div className="mb-2 text-sm font-medium text-foreground">Версии кластеров (clusters_versions)</div>
                <TableFrame>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className={tableHeadClass}>
                        <th className="px-3 py-2.5 text-left">Уровень</th>
                        <th className="px-3 py-2.5 text-left">Версия</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(Object.entries(health.clusters_versions ?? {}) as [string, string | null][]).map(([lvl, ver]) => (
                        <tr key={lvl} className={tableRowClass}>
                          <td className="px-3 py-2 font-mono text-xs text-foreground">{lvl}</td>
                          <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{ver || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </TableFrame>
              </div>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
