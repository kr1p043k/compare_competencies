import { useState } from "react";
import { motion } from "motion/react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Badge } from "./ui/badge";
import {
  Search,
  Download,
  Upload,
  FileSpreadsheet,
  TrendingUp,
  Award,
  Target,
  AlertCircle,
  CheckCircle2,
  BookOpen,
} from "lucide-react";
import { RecommendationsReport } from "./RecommendationsReport";
import { MetricsExplanation } from "./MetricsExplanation";

interface AnalysisTabProps {
  selectedProfile: string;
  onProfileChange: (profile: string) => void;
}

export function AnalysisTab({ selectedProfile, onProfileChange }: AnalysisTabProps) {
  const [loading, setLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState<any>(null);

  const handleMarketSearch = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/results/recommendations/${selectedProfile}`);
      if (response.ok) {
        const data = await response.json();
        setAnalysisData(data);
      }
    } catch (error) {
      console.error("Failed to load analysis:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadExcel = async () => {
    try {
      const response = await fetch(`/api/export/excel/${selectedProfile}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `analysis_${selectedProfile}_${new Date().toISOString().split("T")[0]}.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error("Failed to download Excel:", error);
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
                    <SelectItem value="dc">Профиль ЦК (Middle)</SelectItem>
                    <SelectItem value="top_dc">Топ ЦК (Senior)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button
                onClick={handleMarketSearch}
                disabled={loading}
                className="h-11 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 px-8"
              >
                <Search className="size-4 mr-2" />
                Поиск по всему рынку
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
                  <CardTitle className="text-lg">Excel отчёт</CardTitle>
                  <CardDescription>Скачать полный отчёт по анализу</CardDescription>
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

          {/* Whitelist Expansion */}
          <Card className="border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-sky-50 dark:from-blue-950/20 dark:to-sky-950/20">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-600 rounded-lg">
                  <TrendingUp className="size-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-lg">Белые списки</CardTitle>
                  <CardDescription>Расширение базы навыков</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Button
                variant="outline"
                className="w-full border-blue-300 dark:border-blue-700 hover:bg-blue-100 dark:hover:bg-blue-900/50"
                disabled
              >
                <Award className="size-4 mr-2" />
                Расширить белые списки
                <Badge variant="secondary" className="ml-2">Скоро</Badge>
              </Button>
            </CardContent>
          </Card>

          {/* Student Competencies Upload */}
          <Card className="border-2 border-purple-200 dark:border-purple-800 bg-gradient-to-br from-purple-50 to-fuchsia-50 dark:from-purple-950/20 dark:to-fuchsia-950/20">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-600 rounded-lg">
                  <Upload className="size-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-lg">Компетенции студентов</CardTitle>
                  <CardDescription>Загрузить профили студентов</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Button
                variant="outline"
                className="w-full border-purple-300 dark:border-purple-700 hover:bg-purple-100 dark:hover:bg-purple-900/50"
                disabled
              >
                <Upload className="size-4 mr-2" />
                Подгрузить компетенции
                <Badge variant="secondary" className="ml-2">Скоро</Badge>
              </Button>
            </CardContent>
          </Card>
        </div>
      </motion.div>

      {/* Metrics Explanation */}
      <MetricsExplanation />

      {/* Analysis Results */}
      {analysisData && <RecommendationsReport data={analysisData} />}

      {/* No Data State */}
      {!analysisData && !loading && (
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
                Выберите профиль и нажмите "Поиск по всему рынку"
              </p>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
