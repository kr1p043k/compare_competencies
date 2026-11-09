import { useState, useEffect } from "react";
import { motion } from "motion/react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Badge, Callout, Card, CardBody, CardHeader, EmptyState, PageHeader } from "../../design";
import { useApi } from "../../lib/useApi";
import {
  Target,
  MapPin,
  Briefcase,
  Users,
} from "lucide-react";
import { request } from "../../lib/api";

interface AnalysisTabProps {
  selectedProfile: string;
  onProfileChange: (profile: string) => void;
  pipelineQuery?: string;
  pipelineRegions?: string;
  analysisData?: any;
  onDataLoaded?: (data: any) => void;
  loading?: boolean;
}

export function AnalysisTab({ pipelineQuery, pipelineRegions }: AnalysisTabProps) {

  // Profession trends
  const [professions, setProfessions] = useState<{ name: string }[]>([]);
  const [selectedProf, setSelectedProf] = useState("");
  const [profSkills, setProfSkills] = useState<{ skill: string; frequency: number }[]>([]);
  const [profLoading, setProfLoading] = useState(false);
  const [profError, setProfError] = useState<string | null>(null);

  const professionsQuery = useApi<{ professions: { name: string }[] }>("/api/trends/professions");

  useEffect(() => {
    if (professionsQuery.data?.professions) {
      setProfessions(professionsQuery.data.professions);
    }
  }, [professionsQuery.data]);

  const loadProfessionTrends = async (prof: string) => {
    setSelectedProf(prof);
    if (!prof) return;
    setProfLoading(true);
    setProfError(null);
    try {
      const d = await request<{ skills?: typeof profSkills }>(
        `/api/trends/by-profession?profession=${encodeURIComponent(prof)}&limit=30`,
      );
      setProfSkills(d.skills || []);
    } catch (e: any) {
      setProfError(e?.message || "Не удалось загрузить тренды профессии. Попробуйте позже.");
    } finally {
      setProfLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <PageHeader
        icon={Target}
        title="Анализ компетенций"
        description="Сравнение навыков учебной программы с требованиями рынка труда."
      />
      {(pipelineQuery || pipelineRegions) && (
      <div className="flex flex-wrap justify-center gap-2">
      {pipelineQuery && (
      <Badge tone="primary" className="px-3 py-1 text-sm">
      <Briefcase className="size-3.5" aria-hidden="true" />
      Запрос: {pipelineQuery}
      </Badge>
      )}
      {pipelineRegions && pipelineRegions !== "0" && (
      <Badge tone="primary" className="px-3 py-1 text-sm">
      <MapPin className="size-3.5" aria-hidden="true" />
      Города: {pipelineRegions.split(",").length}
      </Badge>
      )}
      {pipelineRegions === "0" && (
      <Badge tone="primary" className="px-3 py-1 text-sm">
      <MapPin className="size-3.5" aria-hidden="true" />
      Весь рынок
      </Badge>
      )}
      </div>
      )}

      {/* Profession Trends */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card>
          <CardHeader
            className="border-b border-border"
            title={
              <span className="flex items-center gap-3">
                <span aria-hidden="true" className="grid size-9 shrink-0 place-items-center rounded-lg bg-primary text-primary-foreground">
                  <Users className="size-5" />
                </span>
                Тренды по профессиям
              </span>
            }
            description="Самые востребованные навыки в выбранной профессии"
          />
          <CardBody className="pt-6">
            <Select value={selectedProf} onValueChange={loadProfessionTrends}>
              <SelectTrigger className="h-11 border-2 mb-4">
                <SelectValue placeholder="Выберите профессию..." />
              </SelectTrigger>
              <SelectContent>
                {professions.map((p) => (
                  <SelectItem key={p.name} value={p.name}>{p.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>

            {profLoading && <div className="text-center text-muted-foreground py-4">Загрузка...</div>}

            {!profLoading && profError && (
              <div role="alert" className="text-center text-destructive py-4 text-sm">{profError}</div>
            )}

            {!profLoading && !profError && profSkills.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {profSkills.map((s) => (
                  <Badge key={s.skill} tone="neutral" className="px-3 py-1.5 text-sm">
                    {s.skill}
                    <span className="ml-2 text-xs opacity-60">×{s.frequency}</span>
                  </Badge>
                ))}
              </div>
            )}

            {!profLoading && !profError && selectedProf && profSkills.length === 0 && (
              <EmptyState title="Нет данных для этой профессии" className="py-4" />
            )}
          </CardBody>
        </Card>
      </motion.div>

      {/* Результаты переехали: метрики и рекомендации - во вкладке «Данные»,
          формулы и оценка качества - в «Помощь» → «Анализ компетенций». */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Callout tone="info">
          Подгрузка результатов, скачивание Excel и отчёта - во вкладке
          «Данные». Как считаются метрики и насколько им можно доверять - во вкладке
          «Помощь», раздел «Анализ компетенций».
        </Callout>
      </motion.div>

    </div>
  );
}
