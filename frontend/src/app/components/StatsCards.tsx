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
        value={stats?.totalVacancies ?? "—"}
        icon={TrendingUp}
        gradient="from-blue-500 to-cyan-500"
        delay={0}
      />
      <StatCard
        title="Покрытие рынка"
        value={stats?.coverage ? `${stats.coverage}%` : "—"}
        icon={Target}
        gradient="from-purple-500 to-pink-500"
        delay={0.1}
      />
      <StatCard
        title="Рекомендаций"
        value={stats?.recommendations ?? "—"}
        icon={Award}
        gradient="from-emerald-500 to-teal-500"
        delay={0.2}
      />
      <StatCard
        title="Точность ML"
        value={stats?.accuracy ? `${stats.accuracy}%` : "—"}
        icon={Zap}
        gradient="from-orange-500 to-red-500"
        delay={0.3}
      />
    </div>
  );
}
