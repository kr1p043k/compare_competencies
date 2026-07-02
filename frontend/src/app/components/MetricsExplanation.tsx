import { motion } from "motion/react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { CheckCircle2, TrendingUp, Target, Award, BarChart3, Brain } from "lucide-react";

export function MetricsExplanation() {
  const metrics = [
    {
      name: "Match Score (Общий балл соответствия)",
      icon: Target,
      color: "blue",
      description: "Средневзвешенная оценка по трём базовым метрикам",
      explanation: "Усредняет Market Coverage, Skill Coverage и Readiness с весами 0.4/0.3/0.3. Не является независимой метрикой.",
      trustReason: "Зависит от качества трёх нижележащих метрик. Погрешность определяется качеством данных из вакансий.",
      formula: "Match Score = (Market Coverage × 0.4) + (Skill Coverage × 0.3) + (Readiness × 0.3)"
    },
    {
      name: "Market Coverage Score (Покрытие рынка)",
      icon: TrendingUp,
      color: "green",
      description: "Доля навыков студента от всех навыков на рынке",
      explanation: "Сумма весов навыков студента делится на сумму весов всех навыков в БД. Вес навыка = его частота в вакансиях.",
      trustReason: "Данные из ~10 000 вакансий, собранных через API hh.ru. Частоты пересчитываются при каждом сборе.",
      formula: "Market Coverage = (Σ весов навыков студента) / (Σ весов всех навыков) × 100"
    },
    {
      name: "Skill Coverage (Покрытие навыков)",
      icon: Award,
      color: "purple",
      description: "Сравнение с профилем уровня из кластеров",
      explanation: "Для каждого уровня (Junior/Middle/Senior) K-Means выделяет эталонный набор навыков. Считается пересечение навыков студента с эталоном.",
      trustReason: "K-Means работает на parsed_skills из вакансий. Качество кластеризации зависит от чистоты данных и полноты выборки.",
      formula: "Skill Coverage = (|навыки студента ∩ эталон|) / |эталон| × 100"
    },
    {
      name: "Readiness Score (Готовность)",
      icon: CheckCircle2,
      color: "orange",
      description: "Оценка с учётом критичности навыков",
      explanation: "Критичные навыки (упоминаются в >50% вакансий уровня) имеют вес ×2. Учитывается баланс hard/soft skills.",
      trustReason: "Критичность навыка определяется автоматически по частоте в вакансиях, не экспертным мнением.",
      formula: "Readiness = (Critical Skills × 0.5) + (Role Match × 0.3) + (Balance × 0.2)"
    },
    {
      name: "Semantic Similarity (Семантическая близость)",
      icon: Brain,
      color: "pink",
      description: "Косинусная близость эмбеддингов навыков к профилю профессии",
      explanation: "Набор навыков студента эмбеддируется через sentence-transformers и сравнивается с эталонным профилем профессии.",
      trustReason: "Модель: paraphrase-multilingual-mpnet-base-v2. Порог совпадения: 0.78. Не обучена на этих данных, а взята готовая.",
      formula: "Similarity = cos(embedding(студент), embedding(профессия))"
    },
    {
      name: "Domain Coverage (Покрытие доменов)",
      icon: BarChart3,
      color: "indigo",
      description: "Распределение навыков по категориям таксономии",
      explanation: "Всего 23 категории в таксономии (языки, фреймворки, БД, DevOps и т.д.). Считается, сколько из них покрыто хотя бы одним навыком студента.",
      trustReason: "Категории фиксированы в skill_taxonomy.json, не обновляются автоматически под новые технологии.",
      formula: "Domain Coverage = (покрытые категории) / 23 × средний вес"
    }
  ];

  const getColorClass = (color: string) => {
    const colors: Record<string, string> = {
      blue: "from-blue-500 to-blue-600",
      green: "from-green-500 to-green-600",
      purple: "from-purple-500 to-purple-600",
      orange: "from-orange-500 to-orange-600",
      pink: "from-pink-500 to-pink-600",
      indigo: "from-indigo-500 to-indigo-600",
    };
    return colors[color] || colors.blue;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
    >
      <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl">
        <CardHeader className="border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-white/50 to-slate-50/50 dark:from-slate-900/50 dark:to-slate-800/50">
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <Brain className="size-5 text-white" />
            </div>
            Как считаются метрики
          </CardTitle>
          <CardDescription>
            Описание формул и источников данных
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="space-y-4">
            {metrics.map((metric, index) => {
              const Icon = metric.icon;
              return (
                <motion.div
                  key={metric.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 * index }}
                  className="border-2 border-slate-100 dark:border-slate-800 rounded-xl p-4 hover:border-slate-200 dark:hover:border-slate-700 transition-colors"
                >
                  <div className="flex items-start gap-4">
                    <div className={`p-3 bg-gradient-to-br ${getColorClass(metric.color)} rounded-lg shadow-md flex-shrink-0`}>
                      <Icon className="size-6 text-white" />
                    </div>
                    <div className="flex-1 space-y-2">
                      <h4 className="font-bold text-slate-900 dark:text-white">{metric.name}</h4>
                      <p className="text-sm text-slate-600 dark:text-slate-400">{metric.description}</p>

                      <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
                        <p className="text-xs font-semibold text-blue-900 dark:text-blue-100 mb-1">📊 Как рассчитывается:</p>
                        <p className="text-xs text-blue-800 dark:text-blue-200">{metric.explanation}</p>
                      </div>

                      <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-3 border border-green-200 dark:border-green-800">
                        <p className="text-xs font-semibold text-green-900 dark:text-green-100 mb-1">✅ Почему можно доверять:</p>
                        <p className="text-xs text-green-800 dark:text-green-200">{metric.trustReason}</p>
                      </div>

                      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
                        <p className="text-xs font-mono text-slate-700 dark:text-slate-300">{metric.formula}</p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* Overall Trust Statement */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mt-6 p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-xl border-2 border-blue-200 dark:border-blue-800"
          >
            <h4 className="font-bold text-lg text-slate-900 dark:text-white mb-3 flex items-center gap-2">
              <CheckCircle2 className="size-5 text-blue-600" />
              Оценка качества
            </h4>
            <div className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
              <p>
                <strong>Источник данных:</strong> ~10 000 IT-вакансий, собранных через API hh.ru
              </p>
              <p>
                <strong>ML-модели:</strong> K-Means кластеризация уровней, XGBoost (LTR) для ранжирования, sentence-transformers для семантики
              </p>
              <p>
                <strong>LTR метрики:</strong> R² = 0.75, Spearman ρ = 0.59, MAE = 0.01 (на тестовой выборке 1527 пар)
              </p>
              <p>
                <strong>Ограничения:</strong> Частота навыков зависит от качества парсинга описаний вакансий. Прогнозы Prophet на 3 точках данных — индикативны, не точны.
              </p>
            </div>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
