import { useEffect, useState } from "react";
import { motion } from "motion/react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Download,
  Flame,
  Loader2,
  Maximize2,
  Network,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api } from "../api";
import { reportError } from "../../lib/errors";
import { ShapExplanationCard } from "./ShapExplanationCard";
import {
  axisProps,
  Button,
  Card,
  CardBody,
  chartSeries,
  EmptyState,
  ErrorState,
  gridProps,
  Skeleton,
  tooltipProps,
} from "../../design";
import { cn } from "./ui/utils";

interface GapAnalysisVisualizerProps {
  profile: string;
  onProfileChange?: (p: string) => void;
}

const PROFILES = [
  { id: "base", label: "Base (junior)" },
  { id: "dc", label: "Data Scientist (middle)" },
  { id: "top_dc", label: "Top (senior)" },
];

/** Тип представления: ML-важность рисуется интерактивным графиком из
 * -API `/results/recommendations/{profile}`; кластеры продолжают браться
 *  картинкой, пока бэкенд не отдаёт их JSON-ом. */
type ViewType = "ml_importance" | "cluster_insights";

interface ViewConfig {
  type: ViewType;
  title: string;
  description: string;
  icon: LucideIcon;
  /** Фон квадратного значка: токен вместо жёсткого цвета. */
  chipClass: string;
}

const VIEW_CONFIGS: ViewConfig[] = [
  {
    type: "ml_importance",
    title: "Важность признаков ML",
    description: "Приоритизация компетенций по модели",
    icon: Activity,
    chipClass: "bg-chart-5 text-primary-foreground",
  },
  {
    type: "cluster_insights",
    title: "Кластерные инсайты",
    description: "Группировка схожих навыков",
    icon: Network,
    chipClass: "bg-chart-3 text-primary-foreground",
  },
];

/** Одна запись рекомендаций из `/results/recommendations/{profile}`. */
interface MlRecommendation {
  skill?: string;
  /** 0..1 от LTR-модели; в интерфейсе переводится в проценты. */
  importance_score?: number;
  priority?: string;
  why_important?: string;
}

/** Данные столбца графика: навык и его важность в процентах. */
interface MlBarItem {
  skill: string;
  importance: number;
}

function clipSkill(value: string): string {
  return value.length > 24 ? `${value.slice(0, 23)}…` : value;
}

export function GapAnalysisVisualizer({ profile, onProfileChange }: GapAnalysisVisualizerProps) {
  const [viewProfile, setViewProfile] = useState(profile);
  const [selectedView, setSelectedView] = useState<ViewType>("ml_importance");

  // Состояния картинки (кластерные инсайты) - как раньше.
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Состояния ML-графика: данные берутся из API рекомендаций.
  const [mlState, setMlState] = useState<"loading" | "ready" | "empty" | "error">("loading");
  const [mlItems, setMlItems] = useState<MlBarItem[]>([]);
  const [mlMessage, setMlMessage] = useState<string | null>(null);
  const [mlReload, setMlReload] = useState(0);

  const imageUrl = `/api/results/images/${viewProfile}/${selectedView}`;
  const coverageUrl = "/api/results/images/coverage-comparison";
  const heatmapUrl = "/api/results/images/skills-heatmap";
  const correlationUrl = "/api/results/images/skill-correlation";

  const handleImageLoad = () => {
    setImageLoaded(true);
    setLoading(false);
    setError(null);
  };

  const handleImageError = () => {
    setImageLoaded(false);
    setLoading(false);
    setError("Изображение не найдено. Запустите GAP-анализ для генерации визуализаций.");
  };

  // Reset state when profile changes (image URLs change too)
  useEffect(() => {
    setLoading(true);
    setImageLoaded(false);
    setError(null);
  }, [viewProfile]);

  // Keep in sync if parent switches profile
  useEffect(() => {
    setViewProfile(profile);
  }, [profile]);

  // Интерактивный график ML-важности: топ-10 рекомендаций профиля.
  useEffect(() => {
    let alive = true;
    setMlState("loading");
    api<{ recommendations?: MlRecommendation[]; message?: string }>(
      `/results/recommendations/${viewProfile}`,
    )
      .then((payload) => {
        if (!alive) return;
        const recs = Array.isArray(payload?.recommendations) ? payload.recommendations : [];
        const items = recs
          .filter(
            (r): r is { skill: string; importance_score: number } =>
              typeof r?.skill === "string" && typeof r?.importance_score === "number",
          )
          .map((r) => ({
            skill: r.skill,
            importance: Math.min(100, Math.max(0, r.importance_score * 100)),
          }))
          .sort((a, b) => b.importance - a.importance)
          .slice(0, 10)
          // ось Y Recharts идёт снизу вверх: сортируем по возрастанию,
          // чтобы самый важный навык оказался сверху столбцов
          .sort((a, b) => a.importance - b.importance);
        setMlItems(items);
        setMlMessage(payload?.message || null);
        setMlState(items.length > 0 ? "ready" : "empty");
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setMlItems([]);
        setMlState("error");
        reportError("GapAnalysisVisualizer:ml", e);
      });
    return () => {
      alive = false;
    };
  }, [viewProfile, mlReload]);

  const handleViewChange = (type: ViewType) => {
    setSelectedView(type);
    // Для картинки смена типа = смена URL: сбрасываем её состояние.
    // ML-вкладка живёт отдельным состоянием из API-эффекта.
    setLoading(true);
    setImageLoaded(false);
    setError(null);
  };

  const selectedConfig = VIEW_CONFIGS.find((cfg) => cfg.type === selectedView);

  const startAnalysis = () =>
    window.dispatchEvent(
      new CustomEvent("navigate-analysis", { detail: { profile: viewProfile } }),
    );

  return (
    <div className="space-y-6">
      {/* Main visualization card */}
      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-border bg-muted px-5 py-4">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "grid size-10 shrink-0 place-items-center rounded-xl shadow-md",
                selectedConfig?.chipClass,
              )}
            >
              {selectedConfig && <selectedConfig.icon className="size-5" aria-hidden="true" />}
            </div>
            <div>
              <h2 className="text-xl font-semibold tracking-tight text-foreground">
                Визуализация GAP-анализа
              </h2>
              <p className="text-sm text-muted-foreground">{selectedConfig?.description}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Select
              value={viewProfile}
              onValueChange={(v) => {
                setViewProfile(v);
                onProfileChange?.(v);
              }}
            >
              <SelectTrigger className="w-32 h-9 text-sm font-semibold" aria-label="Профиль компетенций">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PROFILES.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <CardBody className="space-y-8 p-8">
          {/* Image type selector */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {VIEW_CONFIGS.map((config) => (
              <motion.div
                key={config.type}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Button
                  onClick={() => handleViewChange(config.type)}
                  variant={selectedView === config.type ? "primary" : "secondary"}
                  aria-pressed={selectedView === config.type}
                  className="h-auto w-full flex-col items-start gap-2 p-4"
                >
                  <config.icon className="size-5" aria-hidden="true" />
                  <span className="text-xs font-semibold text-left">{config.title}</span>
                </Button>
              </motion.div>
            ))}
          </div>

          {/* Interactive ML feature-importance chart */}
          {selectedView === "ml_importance" &&
            (mlState === "loading" ? (
              <div className="rounded-xl border border-border bg-muted p-4">
                <Skeleton className="h-8 w-64" />
                <Skeleton className="mt-4 h-[420px] w-full" />
              </div>
            ) : mlState === "error" ? (
              <ErrorState
                description="Не удалось загрузить ML-рекомендации профиля."
                onRetry={() => setMlReload((n) => n + 1)}
              />
            ) : mlState === "empty" ? (
              <EmptyState
                icon={<Activity className="size-12 opacity-40" aria-hidden="true" />}
                title={`Нет ML-рекомендаций для '${viewProfile}'`}
                description={mlMessage || "Модель не обучена или нет данных"}
                action={
                  <Button variant="primary" size="sm" onClick={startAnalysis}>
                    Запустить анализ для построения графиков
                  </Button>
                }
              />
            ) : (
              <motion.div
                key={`ml-${viewProfile}`}
                className="rounded-xl border border-border bg-muted p-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <h3 className="mb-3 text-sm font-semibold text-foreground">
                  ТОП важных навыков (LTR XGBoost) - {viewProfile}
                </h3>
                <div
                  className="h-[440px] w-full"
                  role="img"
                  aria-label={`Важность недостающих навыков по ML-модели, топ-${mlItems.length}`}
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={mlItems} layout="vertical" margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                      <CartesianGrid {...gridProps} vertical horizontal={false} />
                      <XAxis type="number" domain={[0, 100]} unit="%" {...axisProps} />
                      <YAxis
                        type="category"
                        dataKey="skill"
                        width={170}
                        tickFormatter={(v) => clipSkill(String(v))}
                        {...axisProps}
                      />
                      <RechartsTooltip
                        {...tooltipProps}
                        formatter={(value) => [`${Number(value).toFixed(1)}%`, "Важность навыка"]}
                      />
                      <Bar dataKey="importance" name="Важность" radius={[0, 6, 6, 0]}>
                        {mlItems.map((item, i) => (
                          <Cell
                            key={item.skill}
                            fill={chartSeries[(mlItems.length - 1 - i) % chartSeries.length]}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                {/* Текстовая альтернатива графика для скринридеров. */}
                <ul className="sr-only">
                  {mlItems.map((item) => (
                    <li key={item.skill}>
                      {item.skill}: {item.importance.toFixed(1)}%
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}

          {/* ML #78: per-skill SHAP-объяснения под графиком важности.
           * Карточка монтируется только на выбранной вкладке ml_importance,
           * чтобы дорогой инференс не дёргался на других представлениях. */}
          {selectedView === "ml_importance" && <ShapExplanationCard profile={viewProfile} />}

          {/* Image display (cluster insights) */}
          {selectedView === "cluster_insights" && (
            <motion.div
              className="relative bg-muted rounded-2xl overflow-hidden shadow-inner min-h-[500px] flex items-center justify-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {loading && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-muted/80 backdrop-blur-sm">
                  <div className="text-center">
                    <Loader2 className="size-12 animate-spin text-primary mx-auto mb-3" aria-hidden="true" />
                    <p className="text-sm text-muted-foreground">
                      Загрузка визуализации...
                    </p>
                  </div>
                </div>
              )}

              {error && (
                <div className="absolute inset-0 flex items-center justify-center p-8">
                  <div className="max-w-md text-center">
                    <div className="mb-4 inline-flex rounded-2xl bg-warning-soft p-4">
                      <AlertCircle className="size-12 text-warning" aria-hidden="true" />
                    </div>
                    <h3 className="mb-2 text-lg font-semibold text-foreground">
                      Визуализация недоступна
                    </h3>
                    <p className="text-sm text-muted-foreground">{error}</p>
                    <Button className="mt-4" onClick={startAnalysis}>
                      Запустить анализ для построения графиков
                    </Button>
                  </div>
                </div>
              )}

              {!error && (
                <img
                  key={`manual-${selectedView}`}
                  src={imageUrl}
                  alt={selectedConfig?.title}
                  onLoad={handleImageLoad}
                  onError={handleImageError}
                  className={`max-w-full h-auto transition-opacity duration-300 ${
                    imageLoaded ? "opacity-100" : "opacity-0"
                  }`}
                />
              )}
            </motion.div>
          )}

          {/* Action buttons (image views only) */}
          {selectedView === "cluster_insights" && imageLoaded && (
            <motion.div
              className="flex gap-3"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <Button
                variant="secondary"
                size="sm"
                onClick={() => window.open(imageUrl, "_blank")}
              >
                <Maximize2 className="mr-2 size-4" aria-hidden="true" />
                Открыть в полном размере
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  const link = document.createElement("a");
                  link.href = imageUrl;
                  link.download = `${selectedView}_${viewProfile}.png`;
                  link.click();
                }}
              >
                <Download className="mr-2 size-4" aria-hidden="true" />
                Скачать
              </Button>
            </motion.div>
          )}
        </CardBody>
      </Card>

      {/* Global comparisons */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <GlobalVisualizationCard
          title="Сравнение покрытия"
          description="Сопоставление всех профилей"
          imageUrl={coverageUrl}
          icon={BarChart3}
          chipClass="bg-chart-1"
        />
        <GlobalVisualizationCard
          title="Тепловая карта навыков"
          description="Распределение компетенций"
          imageUrl={heatmapUrl}
          icon={Flame}
          chipClass="bg-chart-4"
        />
        <GlobalVisualizationCard
          title="Корреляция навыков"
          description="Взаимосвязь компетенций"
          imageUrl={correlationUrl}
          icon={Network}
          chipClass="bg-chart-5"
        />
      </div>
    </div>
  );
}

interface GlobalVisualizationCardProps {
  title: string;
  description: string;
  imageUrl: string;
  icon: LucideIcon;
  /** Фон значка: токен вместо жёсткого цвета. */
  chipClass: string;
}

function GlobalVisualizationCard({
  title,
  description,
  imageUrl,
  icon: Icon,
  chipClass,
}: GlobalVisualizationCardProps) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [error, setError] = useState(false);

  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <Card tone="flat" className="h-full overflow-hidden">
        <div className="flex items-center gap-3 border-b border-border bg-muted px-5 py-4">
          <div className={cn("grid size-8 shrink-0 place-items-center rounded-lg shadow-xs", chipClass)}>
            <Icon className="size-4 text-primary-foreground" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <h3 className="truncate text-base font-semibold text-foreground">{title}</h3>
            <p className="text-xs text-muted-foreground">{description}</p>
          </div>
        </div>
        <CardBody className="space-y-3 p-4">
          <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded-xl bg-muted">
            {!imageLoaded && !error && (
              <Skeleton className="absolute inset-0 rounded-none" aria-hidden="true" />
            )}
            {error && (
              <div className="text-center p-4">
                <AlertCircle className="size-8 text-warning mx-auto mb-2" aria-hidden="true" />
                <p className="text-xs text-muted-foreground">
                  Недоступно
                </p>
              </div>
            )}
            <img
              src={imageUrl}
              alt={title}
              onLoad={() => setImageLoaded(true)}
              onError={() => setError(true)}
              className={`max-w-full h-auto transition-opacity duration-300 ${
                imageLoaded ? "opacity-100" : "opacity-0"
              }`}
            />
          </div>
          {imageLoaded && (
            <Button
              variant="ghost"
              size="sm"
              className="w-full text-xs"
              onClick={() => window.open(imageUrl, "_blank")}
            >
              <Maximize2 className="mr-2 size-3" aria-hidden="true" />
              Открыть
            </Button>
          )}
        </CardBody>
      </Card>
    </motion.div>
  );
}
