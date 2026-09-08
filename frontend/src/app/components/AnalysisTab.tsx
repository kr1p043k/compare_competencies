import { useState, useEffect } from "react";
import { motion } from "motion/react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Badge } from "./ui/badge";
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

  useEffect(() => {
    fetch("/api/trends/professions")
      .then((r) => r.ok ? r.json() : { professions: [] })
      .then((d) => setProfessions(d.professions || []))
      .catch(() => {});
  }, []);

  const loadProfessionTrends = async (prof: string) => {
    setSelectedProf(prof);
    if (!prof) return;
    setProfLoading(true);
    try {
      const r = await fetch(`/api/trends/by-profession?profession=${encodeURIComponent(prof)}&limit=30`);
      if (r.ok) {
        const d = await r.json();
        setProfSkills(d.skills || []);
      }
    } catch (e) {
      console.error("Failed to load profession trends:", e);
    } finally {
      setProfLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <div className="inline-flex items-center justify-center gap-3 mb-2">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl blur-xl opacity-50 animate-pulse" />
            <div className="relative bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 rounded-2xl shadow-2xl">
              <Target className="size-8 text-white" />
            </div>
          </div>
          <h2 className="text-4xl font-black bg-gradient-to-r from-slate-900 via-blue-800 to-purple-900 dark:from-white dark:via-blue-200 dark:to-purple-200 bg-clip-text text-transparent">
            Анализ компетенций
          </h2>
        </div>
        <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Полный анализ рынка труда и персональные рекомендации по развитию навыков
        </p>
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
      </motion.div>

      {/* Profession Trends */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
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

            {!profLoading && profSkills.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {profSkills.map((s) => (
                  <Badge key={s.skill} variant="secondary" className="px-3 py-1.5 text-sm">
                    {s.skill}
                    <span className="ml-2 text-xs opacity-60">×{s.frequency}</span>
                  </Badge>
                ))}
              </div>
            )}

            {!profLoading && selectedProf && profSkills.length === 0 && (
              <div className="text-center text-slate-400 py-4">Нет данных для этой профессии</div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Результаты переехали: метрики и рекомендации — во вкладке «Данные»,
          формулы и оценка качества — в «Помощь» → «Анализ компетенций». */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20">
          <CardContent className="pt-6 pb-6 text-sm text-slate-700 dark:text-slate-300">
            Подгрузка результатов, скачивание Excel и отчёта — во вкладке
            «Данные». Как считаются метрики и насколько им можно доверять — во вкладке
            «Помощь», раздел «Анализ компетенций».
          </CardContent>
        </Card>
      </motion.div>

    </div>
  );
}
