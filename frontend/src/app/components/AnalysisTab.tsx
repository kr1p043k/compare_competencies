import { useState, useEffect } from "react";
import { motion } from "motion/react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Badge } from "./ui/badge";
import { PageHeader } from "./PageHeader";
import { useApi } from "../../lib/useApi";
import {
  Target,
  MapPin,
  Briefcase,
  Users,
} from "lucide-react";

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
      const r = await fetch(`/api/trends/by-profession?profession=${encodeURIComponent(prof)}&limit=30`);
      if (!r.ok) {
        const d = await r.json().catch(() => null);
        throw new Error((d && d.detail) || "Не удалось загрузить тренды профессии. Попробуйте позже.");
      }
      const d = await r.json();
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
      <div className="flex flex-wrap justify-center gap-2 text-sm text-slate-500">
      {pipelineQuery && (
      <span className="inline-flex items-center gap-1 px-3 py-1 bg-blue-50 dark:bg-blue-950/30 rounded-full">
      <Briefcase className="size-3.5" />
      Запрос: {pipelineQuery}
      </span>
      )}
      {pipelineRegions && pipelineRegions !== "0" && (
      <span className="inline-flex items-center gap-1 px-3 py-1 bg-blue-50 dark:bg-blue-950/30 rounded-full">
      <MapPin className="size-3.5" />
      Города: {pipelineRegions.split(",").length}
      </span>
      )}
      {pipelineRegions === "0" && (
      <span className="inline-flex items-center gap-1 px-3 py-1 bg-blue-50 dark:bg-blue-950/30 rounded-full">
      <MapPin className="size-3.5" />
      Весь рынок
      </span>
      )}
      </div>
      )}

      {/* Profession Trends */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
          <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-600 rounded-lg">
                <Users className="size-5 text-white" />
              </div>
              <div>
                <CardTitle>Тренды по профессиям</CardTitle>
                <CardDescription>Самые востребованные навыки в выбранной профессии</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-6">
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

            {profLoading && <div className="text-center text-slate-500 py-4">Загрузка...</div>}

            {!profLoading && profError && (
              <div className="text-center text-red-600 dark:text-red-400 py-4 text-sm">{profError}</div>
            )}

            {!profLoading && !profError && profSkills.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {profSkills.map((s) => (
                  <Badge key={s.skill} variant="secondary" className="px-3 py-1.5 text-sm">
                    {s.skill}
                    <span className="ml-2 text-xs opacity-60">×{s.frequency}</span>
                  </Badge>
                ))}
              </div>
            )}

            {!profLoading && !profError && selectedProf && profSkills.length === 0 && (
              <div className="text-center text-slate-400 py-4">Нет данных для этой профессии</div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Результаты переехали: метрики и рекомендации – во вкладке «Данные»,
          формулы и оценка качества – в «Помощь» → «Анализ компетенций». */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="border-2 border-blue-200 dark:border-blue-800 bg-slate-50 dark:bg-slate-900">
          <CardContent className="pt-6 pb-6 text-sm text-slate-700 dark:text-slate-300">
            Подгрузка результатов, скачивание Excel и отчёта – во вкладке
            «Данные». Как считаются метрики и насколько им можно доверять – во вкладке
            «Помощь», раздел «Анализ компетенций».
          </CardContent>
        </Card>
      </motion.div>

    </div>
  );
}
