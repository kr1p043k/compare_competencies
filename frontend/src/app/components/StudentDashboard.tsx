import { useState, useEffect } from "react";
import { RefreshCw, MapPin, Briefcase, FileText } from "lucide-react";
import { SelfProfileEditor } from "./SelfProfileEditor";
import { StudentKrm } from "./StudentKrm";
import { StudentStartHere } from "./StudentStartHere";
import { LlmChatPanel } from "./LlmChatPanel";
import { request } from "../../lib/api";
import { reportError } from "../../lib/errors";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  SectionTitle,
  Skeleton,
  tableHeadClass,
  tableRowClass,
} from "../../design";
import { cn } from "./ui/utils";

export function StudentDashboard({ onNavigate }: { onNavigate?: (tab: string) => void }) {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  // Ошибка истории теперь доходит до UI: раньше catch писал только в лог,
  // и таблица оставалась пустой без объяснения причины.
  const [error, setError] = useState<string | null>(null);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await request<{ history?: unknown[] }>("/api/student/history?limit=50");
      setHistory(data.history || []);
    } catch (err) {
      setError(reportError("StudentDashboard:history", err));
    } finally { setLoading(false); }
  };

  useEffect(() => {
    loadHistory();
    const handler = () => loadHistory();
    window.addEventListener("student-history-update", handler);
    return () => window.removeEventListener("student-history-update", handler);
  }, []);

  return (
    <div className="space-y-6">
      <StudentStartHere onNavigate={onNavigate} />
      <StudentKrm onNavigate={onNavigate} />
      <SectionTitle
        title="Мои запросы"
        hint="История запусков анализа"
        className="mb-0"
        actions={
          <Button
            variant="secondary"
            size="sm"
            onClick={loadHistory}
            loading={loading}
            iconLeft={<RefreshCw className={loading ? "animate-spin" : ""} />}
          >
            Обновить
          </Button>
        }
      />

      <Card>
        <CardHeader title="История запросов" />
        <CardBody>
          {loading && (
            <div className="space-y-2" role="status" aria-label="Загрузка истории запросов">
              <span className="sr-only">Загрузка истории…</span>
              {[0, 1, 2].map((i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          )}
          {!loading && error && (
            <ErrorState
              title="Не удалось загрузить историю запросов"
              description={error}
              onRetry={() => void loadHistory()}
            />
          )}
          {!loading && !error && history.length === 0 && (
            <EmptyState
              title="История пуста"
              description="Запустите анализ компетенций, чтобы здесь появились записи."
            />
          )}
          {!loading && !error && history.length > 0 && (
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className={cn(tableHeadClass, "sticky top-0 z-10 bg-card text-left")}>
                  <th className="pb-2 pr-3">Время</th>
                  <th className="pb-2 pr-3">Запрос</th>
                  <th className="pb-2 pr-3">Регион</th>
                  <th className="pb-2 pr-3 text-right">Вакансий</th>
                  <th className="pb-2">Результат</th>
                </tr>
              </thead>
                <tbody>
                  {history.toReversed().map((h, i) => (
                    <tr key={i} className={tableRowClass}>
                      <td className="whitespace-nowrap py-2 text-xs text-muted-foreground">
                        {new Date(h.timestamp).toLocaleString()}
                      </td>
                      <td className="py-2">
                        <div className="flex items-center gap-1.5">
                          <Briefcase className="size-3.5 text-muted-foreground" />
                          <span className="font-medium text-foreground">{h.profession || "-"}</span>
                        </div>
                        {h.profile && (
                          <span className="ml-5 text-xs text-subtle-foreground">профиль: {h.profile}</span>
                        )}
                      </td>
                      <td className="py-2">
                        <div className="flex items-center gap-1.5">
                          <MapPin className="size-3.5 text-muted-foreground" />
                          <span className="text-muted-foreground">
                            {h.region === "0" ? "Весь рынок" : h.region || "-"}
                          </span>
                        </div>
                      </td>
                      <td className="py-2 text-right">
                        <Badge tone="neutral" className="text-xs">
                          {h.vacancies_found > 0 ? h.vacancies_found : "-"}
                        </Badge>
                      </td>
                      <td className="py-2">
                        {h.profile ? (
                          <button
                            onClick={() =>
                              window.dispatchEvent(
                                new CustomEvent("navigate-analysis", { detail: { profile: h.profile } }),
                              )
                            }
                            className="cursor-pointer rounded-sm text-xs text-primary underline underline-offset-2 hover:text-primary/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                          >
                            <FileText className="mr-1 inline size-3.5" />
                            Открыть анализ
                          </button>
                        ) : (
                          <span className="text-xs text-muted-foreground">-</span>
                        )}
                      </td>
                    </tr>
                  ))}
                 </tbody>
               </table>
             </div>
           )}
         </CardBody>
      </Card>
      <SelfProfileEditor />
      <LlmChatPanel role="student" />
    </div>
  );
}
