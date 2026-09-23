import { motion } from "motion/react";
import { Card, CardContent } from "./ui/card";
import { TrendingUp, Target, Award, Zap } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  icon: any;
  gradient: string;
  delay?: number;
}

function StatCard({ title, value, icon: Icon, gradient, delay = 0 }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, type: "spring", stiffness: 100 }}
      whileHover={{ scale: 1.03, y: -2 }}
    >
      <Card className="border-0 shadow-lg bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl overflow-hidden relative">
        <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-5`} />
        <CardContent className="p-6 relative">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-sm font-medium text-slate-600 dark:text-slate-400">
                {title}
              </p>
              <p className="text-3xl font-bold text-slate-900 dark:text-white">
                {value}
              </p>
            </div>
            <div className={`p-3 bg-gradient-to-br ${gradient} rounded-xl shadow-lg`}>
              <Icon className="size-6 text-white" />
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

interface StatsCardsProps {
  stats?: {
    totalVacancies?: number;
    coverage?: number;
    recommendations?: number;
    accuracy?: number;
  };
}

export function StatsCards({ stats }: StatsCardsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <StatCard
        title="Вакансий собрано"
        value={stats?.totalVacancies ?? "–"}
        icon={TrendingUp}
        gradient="from-blue-50 dark:from-blue-950/30 to-cyan-50 dark:to-cyan-950/30"
        delay={0}
      />
      <StatCard
        title="Покрытие рынка"
        value={stats?.coverage ? `${stats.coverage}%` : "–"}
        icon={Target}
        gradient="from-purple-50 dark:from-purple-950/30 to-pink-50 dark:to-pink-950/30"
        delay={0.1}
      />
      <StatCard
        title="Рекомендаций"
        value={stats?.recommendations ?? "–"}
        icon={Award}
        gradient="from-emerald-50 dark:from-emerald-950/30 to-teal-50 dark:to-teal-950/30"
        delay={0.2}
      />
      <StatCard
        title="Точность ML"
        value={stats?.accuracy ? `${stats.accuracy}%` : "–"}
        icon={Zap}
        gradient="from-orange-50 dark:from-orange-950/30 to-red-50 dark:to-red-950/30"
        delay={0.3}
      />
    </div>
  );
}
