import { useEffect, useState } from "react";
import { Activity, FileText, Shield, Users } from "lucide-react";
import { request } from "../../../lib/api";
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  PageHeader,
  Skeleton,
  StatCard,
} from "../../../design";
import { TwoFactorCard } from "../TwoFactorCard";
import { AdminSecurity } from "../AdminSecurity";
import { AccountCard } from "./AccountCard";
import type { AccountProps } from "./types";

/** Полная страница профиля администратора: учётка + live-сводка + ссылки на админ-разделы. */
export function AdminProfilePage({ displayName, email, onNavigate }: AccountProps & { onNavigate: (tab: string) => void }) {
  const [sum, setSum] = useState<{
    sessions_active?: number | null;
    errors_1h?: { "4xx"?: number; "5xx"?: number };
    freshness?: { vacancies?: number; date_from?: string | null; date_to?: string | null };
  } | null>(null);
  const [sumLoading, setSumLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const d = await request<Record<string, unknown>>("/api/admin/monitoring/summary");
        if (!cancelled) setSum(d as never);
      } catch {
        if (!cancelled) setSum(null);
      } finally {
        if (!cancelled) setSumLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const errorsTotal =
    sum && sum.errors_1h
      ? (sum.errors_1h["4xx"] ?? 0) + (sum.errors_1h["5xx"] ?? 0)
      : null;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Мой профиль"
        description="Учётная запись, дисциплины и уведомления администратора."
      />
      <AccountCard displayName={displayName} email={email} roleLabel="Администратор" />
      <TwoFactorCard />
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Activity className="size-5 text-primary" />
              Живая сводка
            </span>
          }
          description="Активность, ошибки и свежесть данных - прочерк при недоступности"
        />
        <CardBody>
          {sumLoading ? (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-2" aria-label="Загрузка сводки">
              {[0, 1, 2].map((i) => (
                <div key={i} className="rounded-lg border border-border p-3">
                  <Skeleton className="h-6 w-12 mx-auto" />
                  <Skeleton className="mt-2 h-3 w-20 mx-auto" />
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
              <StatCard
                className="p-3"
                label="Сессии активны"
                value={typeof sum?.sessions_active === "number" ? sum.sessions_active : "-"}
              />
              <StatCard
                className="p-3"
                label="Ошибки за час"
                value={errorsTotal === null ? "-" : errorsTotal}
              />
              <StatCard
                className="col-span-2 lg:col-span-1 p-3"
                label={`Вакансии${sum?.freshness?.date_from ? ` · ${sum.freshness.date_from}-${sum.freshness.date_to ?? "…"}` : ""}`}
                value={
                  typeof sum?.freshness?.vacancies === "number" ? sum.freshness.vacancies.toLocaleString("ru-RU") : "-"
                }
              />
            </div>
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Shield className="size-5 text-primary" />
              Администрирование
            </span>
          }
          description="Быстрые ссылки на административные разделы"
        />
        <CardBody className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            className="cursor-pointer"
            onClick={() => onNavigate("admin")}
            iconLeft={<Users className="size-4" />}
          >
            Пользователи
          </Button>
          <Button
            variant="secondary"
            className="cursor-pointer"
            onClick={() => onNavigate("logs")}
            iconLeft={<FileText className="size-4" />}
          >
            Логи
          </Button>
          <Button
            variant="secondary"
            className="cursor-pointer"
            onClick={() => onNavigate("monitoring")}
            iconLeft={<Activity className="size-4" />}
          >
            Мониторинг
          </Button>
        </CardBody>
      </Card>
      <AdminSecurity />
    </div>
  );
}
