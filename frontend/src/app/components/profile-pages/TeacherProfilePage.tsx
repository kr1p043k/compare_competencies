import { useEffect, useState } from "react";
import { Activity, BookOpen, Loader2 } from "lucide-react";
import { request } from "../../../lib/api";
import { errMessage } from "../../../lib/errors";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  PageHeader,
} from "../../../design";
import { TwoFactorCard } from "../TwoFactorCard";
import { AccountCard } from "./AccountCard";
import type { AccountProps, DisciplineRow } from "./types";

/** Полная страница профиля преподавателя: учётка + зона преподавания + ссылки на анализ. */
export function TeacherProfilePage({ displayName, email, onNavigate }: AccountProps & { onNavigate?: (tab: string) => void }) {
  const [discs, setDiscs] = useState<DisciplineRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    request<DisciplineRow[]>("/api/teacher/krm/disciplines?dir_code=09.03.02")
      .then((d) => { if (!cancelled) setDiscs(Array.isArray(d) ? d : []); })
      .catch((e: unknown) => { if (!cancelled) setError(errMessage(e) || "Не удалось загрузить дисциплины"); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Мой профиль"
        description="Учётная запись, дисциплины и уведомления преподавателя."
      />
      <AccountCard displayName={displayName} email={email} roleLabel="Преподаватель" />
      <TwoFactorCard />
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <BookOpen className="size-5 text-success" />
              Зона преподавания
            </span>
          }
          description="Дисциплины направления 09.03.02 (только просмотр)"
        />
        <CardBody>
          {discs === null && !error && (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Загрузка…
            </p>
          )}
          {error && <ErrorState description={error} />}
          {discs !== null && !error && (
            discs.length === 0 ? (
              <EmptyState title="Дисциплины не найдены" />
            ) : (
              <ul className="divide-y divide-border">
                {discs.map((d) => (
                  <li key={d.name} className="py-2 flex items-center justify-between gap-3">
                    <span className="text-sm text-foreground min-w-0">
                      <span className="block truncate">{d.name}</span>
                      <span className="flex items-center gap-1.5 mt-1 flex-wrap">
                        {typeof d.semester === "number" && (
                          <Badge tone="neutral" className="text-xs">сем. {d.semester}</Badge>
                        )}
                        {typeof d.competencies_count === "number" && (
                          <Badge tone="neutral" className="text-xs">компетенций: {d.competencies_count}</Badge>
                        )}
                        {typeof d.skills_count === "number" && (
                          <Badge tone="neutral" className="text-xs">навыков: {d.skills_count}</Badge>
                        )}
                      </span>
                    </span>
                    {onNavigate && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => onNavigate("teacher")}
                        title={`Анализ дисциплины «${d.name}»`}
                        className="shrink-0 cursor-pointer"
                        iconLeft={<Activity className="size-3.5" />}
                      >
                        Анализ
                      </Button>
                    )}
                  </li>
                ))}
              </ul>
            )
          )}
          {onNavigate && (
            <div className="mt-4 flex flex-wrap gap-2 border-t border-border pt-4">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onNavigate("teacher")}
                className="cursor-pointer"
                iconLeft={<Activity className="size-3.5" />}
              >
                Преподавательский анализ
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onNavigate("taxonomy")}
                className="cursor-pointer"
                iconLeft={<BookOpen className="size-3.5" />}
              >
                Таксономия
              </Button>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
