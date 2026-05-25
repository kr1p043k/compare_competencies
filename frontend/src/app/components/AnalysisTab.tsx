import { useState } from "react";
import { motion } from "motion/react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import {
  Search,
  Download,
  FileSpreadsheet,
  FileText,
  Target,
  BookOpen,
  MapPin,
  Briefcase,
} from "lucide-react";
import { RecommendationsReport } from "./RecommendationsReport";
import { MetricsExplanation } from "./MetricsExplanation";

interface AnalysisTabProps {
  selectedProfile: string;
  onProfileChange: (profile: string) => void;
  pipelineQuery?: string;
  pipelineRegions?: string;
  analysisData?: any;
  onDataLoaded?: (data: any) => void;
  loading?: boolean;
}

export function AnalysisTab({ selectedProfile, onProfileChange, pipelineQuery, pipelineRegions, analysisData, onDataLoaded, loading: externalLoading }: AnalysisTabProps) {
  const [loading, setLoading] = useState(false);
  const [localData, setLocalData] = useState<any>(null);
  const effectiveLoading = loading || externalLoading;
  const effectiveData = analysisData ?? localData;

  const handleMarketSearch = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/results/recommendations/${selectedProfile}`);
      if (response.ok) {
        const data = await response.json();
        setLocalData(data);
        onDataLoaded?.(data);
      }
    } catch (error) {
      console.error("Failed to load analysis:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadExcel = async () => {
    try {
      const response = await fetch(`/api/admin/export/excel`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `vacancies_${new Date().toISOString().split("T")[0]}.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else if (response.status === 429) {
        alert("Слишком частые запросы. Подождите 20 секунд.");
      } else {
        const d = await response.json().catch(() => ({}));
        alert(d.detail || "Ошибка выгрузки Excel");
      }
    } catch (error) {
      console.error("Failed to download Excel:", error);
    }
  };

  const handleDownloadReport = async () => {
    try {
      const response = await fetch(`/api/results/recommendations/${selectedProfile}`);
      if (response.ok) {
        const data = await response.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `analysis_report_${selectedProfile}_${new Date().toISOString().split("T")[0]}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error("Failed to download analysis report:", error);
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

      {/* Profile Selection */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
          <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
            <CardTitle>Выбор профиля</CardTitle>
            <CardDescription>Выберите профиль студента для анализа</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <Select value={selectedProfile} onValueChange={onProfileChange}>
                  <SelectTrigger className="h-11 border-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="base">Базовый профиль (Junior)</SelectItem>
                    <SelectItem value="dc">DATA SCIENTIST (Middle)</SelectItem>
                    <SelectItem value="top_dc">TOP DATA SCIENTIST (Senior)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button
                onClick={handleMarketSearch}
                disabled={effectiveLoading}
                className="h-11 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 px-8"
              >
                <Search className="size-4 mr-2" />
                Загрузить результаты
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Excel Export */}
          <Card className="border-2 border-green-200 dark:border-green-800 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-600 rounded-lg">
                  <FileSpreadsheet className="size-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-lg">Excel вакансий</CardTitle>
                  <CardDescription>Скачать список вакансий с навыками</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Button
                onClick={handleDownloadExcel}
                variant="outline"
                className="w-full border-green-300 dark:border-green-700 hover:bg-green-100 dark:hover:bg-green-900/50"
              >
                <Download className="size-4 mr-2" />
                Скачать Excel
              </Button>
            </CardContent>
          </Card>

          {/* Analysis Report */}
          <Card className="border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-600 rounded-lg">
                  <FileText className="size-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-lg">Отчёт по анализу</CardTitle>
                  <CardDescription>Скачать результаты gap-анализа</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Button
                onClick={handleDownloadReport}
                variant="outline"
                className="w-full border-blue-300 dark:border-blue-700 hover:bg-blue-100 dark:hover:bg-blue-900/50"
              >
                <Download className="size-4 mr-2" />
                Скачать отчёт
              </Button>
            </CardContent>
          </Card>


        </div>
      </motion.div>

      {/* Metrics Explanation */}
      <MetricsExplanation />

      {/* Analysis Results */}
      {effectiveData && <RecommendationsReport data={effectiveData} />}

      {/* No Data State */}
      {!effectiveData && !effectiveLoading && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <Card className="border-2 border-slate-200 dark:border-slate-700">
            <CardContent className="pt-6 text-center py-20">
              <BookOpen className="size-16 text-slate-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Нет данных анализа
              </h3>
              <p className="text-slate-600 dark:text-slate-400">
                Выберите профиль и нажмите "Загрузить результаты"
              </p>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
