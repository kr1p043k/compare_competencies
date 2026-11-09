import { useEffect, useState } from "react";
import { Activity } from "lucide-react";
import { api } from "../api";
import { reportError } from "../../lib/errors";
import {
  Badge,
  type BadgeTone,
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorState,
  Meter,
  Skeleton,
} from "../../design";

/** Направление SHAP-вклада из `/results/ml-explanations/{profile}`. */
type ShapDirection = "positive" | "negative" | "zero";

/** Один вклад признака: что и насколько сдвинуло оценку навыка. */
interface ShapContribution {
  feature: string;
  value: number;
  direction: ShapDirection;
  feature_value?: number | null;
}

/** Оценка одного навыка: скор + текстовое объяснение + топ-вклады. */
interface ShapExplanation {
  skill: string;
  /** Оценка влияния навыка 0..100 из LTR-модели. */
  score: number;
  explanation: string;
  contributions: ShapContribution[];
}

interface MlExplanationsPayload {
  profile?: string;
  model?: string;
  message?: string;
  explanations?: ShapExplanation[];
}

/** RU-подписи runtime-признаков LTR-модели; незнакомый ключ показываем как есть. */
const FEATURE_LABELS: Record<string, string> = {
  level_encoded: "Уровень востребованности",
  category_encoded: "Категория навыка",
  cosine_sim: "Сходство с профилем",
  in_student_profile: "Наличие в профиле",
  skill_freq_normalized: "Частота на рынке",
  co_occurrence: "Совместная встречаемость",
  category_avg_weight: "Средний вес категории",
  student_skills_count: "Размер профиля",
};

const DIRECTION_META: Record<ShapDirection, { tone: BadgeTone; label: string; fill: string }> = {
  positive: { tone: "success", label: "повышает", fill: "bg-success" },
  negative: { tone: "danger", label: "снижает", fill: "bg-destructive" },
  zero: { tone: "neutral", label: "не влияет", fill: "bg-muted-foreground" },
};

/** Знаковое значение SHAP: "+" для повышающих, фикс. 4 знака после запятой. */
function formatShapValue(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(4)}`;
}

/**
 * ML #78: карточка per-skill SHAP-объяснений.
 *
 * Живой инференс: `/results/ml-explanations/{profile}` (LTR XGBoost).
 * На каждый навык - скор модели, готовое объяснение и топ-вклады
 * признаков (Badge направления + Meter величины |SHAP|, нормированный
 * по максимуму внутри навыка). Все состояния честные: скелет при
 * загрузке, EmptyState с message бэкенда, ErrorState с повтором.
 */
export function ShapExplanationCard({ profile }: { profile: string }) {
  const [state, setState] = useState<"loading" | "ready" | "empty" | "error">("loading");
  const [items, setItems] = useState<ShapExplanation[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let alive = true;
    setState("loading");
    api<MlExplanationsPayload>(`/results/ml-explanations/${profile}`)
      .then((payload) => {
        if (!alive) return;
        const explanations = (Array.isArray(payload?.explanations) ? payload.explanations : [])
          .filter(
            (e): e is ShapExplanation =>
              typeof e?.skill === "string" &&
              typeof e?.score === "number" &&
              typeof e?.explanation === "string",
          );
        setItems(explanations);
        setMessage(typeof payload?.message === "string" ? payload.message : null);
        setState(explanations.length > 0 ? "ready" : "empty");
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setState("error");
        reportError("ShapExplanationCard:ml-explanations", e);
      });
    return () => {
      alive = false;
    };
  }, [profile, reload]);

  return (
    <Card>
      <CardHeader
        title="Объяснение ML-рекомендаций"
        description="Что повлияло на оценку каждого навыка (SHAP)"
      />
      <CardBody className="space-y-5">
        {state === "loading" && (
          <div className="space-y-3">
            <Skeleton className="h-5 w-56" />
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-20 w-full" />
          </div>
        )}
        {state === "error" && (
          <ErrorState
            description="Не удалось загрузить объяснения ML-модели."
            onRetry={() => setReload((n) => n + 1)}
          />
        )}
        {state === "empty" && (
          <EmptyState
            icon={<Activity className="size-10 opacity-40" aria-hidden="true" />}
            title="Нет ML-объяснений"
            description={message || "Модель не обучена или профиль уже покрывает рынок"}
          />
        )}
        {state === "ready" && (
          <ul className="space-y-4" aria-label="Объяснения ML-модели по навыкам">
            {items.map((e) => {
              // Числовой фильтр: некорректный вклад не роняет всю карточку.
              const contribs = e.contributions.filter(
                (c): c is ShapContribution =>
                  typeof c?.value === "number" && typeof c?.feature === "string",
              );
              // Шкала Meter - |SHAP| относительно максимума внутри навыка.
              const maxAbs = Math.max(...contribs.map((c) => Math.abs(c.value)), 0);
              return (
                <li
                  key={e.skill}
                  className="space-y-2.5 rounded-lg border border-border bg-muted/40 p-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-foreground">{e.skill}</span>
                    <Badge tone="primary">Оценка {Math.round(e.score)}/100</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">{e.explanation}</p>
                  {contribs.length > 0 ? (
                    <ul className="space-y-2.5">
                      {contribs.map((c) => {
                        const meta = DIRECTION_META[c.direction] || DIRECTION_META.zero;
                        return (
                          <li key={c.feature} className="space-y-1">
                            <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
                              <span className="text-xs font-medium text-foreground">
                                {FEATURE_LABELS[c.feature] ?? c.feature}
                              </span>
                              <span className="flex items-center gap-1.5">
                                <span className="tabular text-xs text-muted-foreground">
                                  {formatShapValue(c.value)}
                                </span>
                                <Badge tone={meta.tone}>{meta.label}</Badge>
                              </span>
                            </div>
                            {/* Полоса декоративна: число уже показано текстом. */}
                            <div aria-hidden="true">
                              <Meter
                                value={maxAbs > 0 ? (Math.abs(c.value) / maxAbs) * 100 : 0}
                                fillClassName={meta.fill}
                              />
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      Вклады признаков для этого навыка не рассчитаны.
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}
