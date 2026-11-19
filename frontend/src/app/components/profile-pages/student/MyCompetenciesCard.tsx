import { BookOpen, CheckCircle2, ChevronDown, Database, Plus } from "lucide-react";
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  Skeleton,
} from "../../../../design";
import type { DisciplineStat, KrmData, MissingSkill } from "../types";

type MyCompetenciesCardProps = {
  krmLoading: boolean;
  krmError: string | null;
  onRetry: () => void;
  krm: KrmData | null;
  dirCode: string;
  topGaps: DisciplineStat[];
  missingPool: MissingSkill[];
  discStats: DisciplineStat[];
  openDisc: Record<string, boolean>;
  onToggleDisc: (discipline: string) => void;
  saving: boolean;
  onAddSkill: (text: string) => void;
  onNavigate: (tab: string) => void;
};

export function MyCompetenciesCard({
  krmLoading,
  krmError,
  onRetry,
  krm,
  dirCode,
  topGaps,
  missingPool,
  discStats,
  openDisc,
  onToggleDisc,
  saving,
  onAddSkill,
  onNavigate,
}: MyCompetenciesCardProps) {
  return (
    <Card>
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <BookOpen className="size-5 text-primary" />
            Мои компетенции
          </span>
        }
        description={`Программа ${krm?.direction_name || dirCode}: что совпало, чего не хватает`}
      />
      <CardBody className="space-y-3">
        {krmLoading && (
          <div className="space-y-2" aria-label="Загрузка компетенций">
            {[0, 1, 2].map((i) => (
              <div key={i} className="rounded-lg border border-border p-3">
                <Skeleton className="h-4 w-2/3" />
                <div className="mt-2 flex gap-1.5">
                  <Skeleton className="h-5 w-16" />
                  <Skeleton className="h-5 w-24" />
                </div>
              </div>
            ))}
          </div>
        )}
        {krmError && !krmLoading && (
          <ErrorState description={krmError} onRetry={onRetry} />
        )}
        {krm && !krmLoading && (
          <>
            {krm.disciplines.length === 0 ? (
              <EmptyState title="Нет данных" description="Добавьте навыки выше" />
            ) : (
              <>
                {topGaps.length > 0 && (
                  <div className="rounded-lg border border-border p-3">
                    <div className="text-sm font-medium text-foreground">
                      Топ пробелов ({topGaps.length})
                    </div>
                    <ul className="mt-1.5 space-y-1 text-sm text-muted-foreground">
                      {topGaps.map((g) => (
                        <li key={g.name} className="flex items-center justify-between gap-2">
                          <span className="truncate">{g.name}</span>
                          <Badge tone="neutral" className="text-xs shrink-0">
                            не хватает {g.missing}/{g.total}
                          </Badge>
                        </li>
                      ))}
                    </ul>
                    {missingPool.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {missingPool.slice(0, 4).map((m) => (
                          <Badge
                            key={`${m.discipline}::${m.code}::${m.text}`}
                            tone="neutral"
                            className="max-w-full text-xs"
                          >
                            <span className="break-words">{m.text}</span>
                            <button
                              onClick={() => onAddSkill(m.text)}
                              disabled={saving}
                              className="cursor-pointer rounded p-0.5 text-muted-foreground transition-colors duration-200 hover:bg-muted hover:text-success focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                              aria-label={`Добавить «${m.text}»`}
                            >
                              <Plus className="size-3" />
                            </button>
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                <div className="space-y-2">
                  {krm.disciplines.map((d) => {
                    const st = discStats.find((s) => s.name === d.discipline);
                    const total = st?.total ?? 0;
                    const has = st?.has ?? 0;
                    const open = !!openDisc[d.discipline];
                    const matched = d.competencies.flatMap((c) => c.skills.filter((s) => s.student_has));
                    const missing = d.competencies.flatMap((c) =>
                      c.skills.filter((s) => !s.student_has).map((s) => ({ ...s, code: c.code }))
                    );
                    return (
                      <div key={d.discipline} className="rounded-lg border border-border bg-card transition-colors duration-200">
                        <button
                          onClick={() => onToggleDisc(d.discipline)}
                          className="flex w-full cursor-pointer items-center justify-between gap-2 px-3 py-2.5 text-left transition-colors duration-200 hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                          aria-expanded={open}
                        >
                          <span className="truncate text-sm font-medium text-foreground">
                            {d.discipline}
                          </span>
                          <span className="flex shrink-0 items-center gap-2">
                            <Badge tone={has > 0 ? "primary" : "neutral"} className="text-xs">
                              {has}/{total}
                            </Badge>
                            <ChevronDown className={`size-4 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
                          </span>
                        </button>
                        {open && (
                          <div className="space-y-2 px-3 pb-3">
                            {total === 0 && (
                              <p className="text-xs text-muted-foreground">
                                нет данных - добавьте навыки выше
                              </p>
                            )}
                            {matched.length > 0 && (
                              <div>
                                <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                                  Совпало ({matched.length})
                                </div>
                                <div className="flex flex-wrap gap-1.5">
                                  {matched.map((s, i) => (
                                    <Badge
                                      key={i}
                                      tone="success"
                                      className="max-w-full text-xs"
                                      title="Есть у вас"
                                    >
                                      <CheckCircle2 className="size-3 shrink-0" />
                                      <span className="break-words">{s.text}</span>
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                            {missing.length > 0 && (
                              <div>
                                <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                                  Не хватает ({missing.length})
                                </div>
                                <div className="flex flex-wrap gap-1.5">
                                  {missing.map((s, i) => (
                                    <Badge
                                      key={i}
                                      tone="neutral"
                                      className="max-w-full text-xs"
                                      title="Нет у вас - нажмите +, чтобы добавить"
                                    >
                                      <span className="break-words">{s.text}</span>
                                      <button
                                        onClick={() => onAddSkill(s.text)}
                                        disabled={saving}
                                        className="cursor-pointer rounded p-0.5 text-muted-foreground transition-colors duration-200 hover:bg-muted hover:text-success focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                                        aria-label={`Добавить «${s.text}»`}
                                      >
                                        <Plus className="size-3" />
                                      </button>
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </>
            )}
            <div>
              <Button
                variant="secondary"
                onClick={() => onNavigate("data")}
                className="cursor-pointer"
                iconLeft={<Database className="size-4" />}
              >
                Открыть результаты
              </Button>
            </div>
          </>
        )}
      </CardBody>
    </Card>
  );
}
