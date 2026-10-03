import { useMemo } from "react";
import { motion } from "motion/react";
import { BarChart3, Briefcase, CalendarDays, Database, Trophy } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";

interface MarketSkill {
  skill: string;
  weight: number;
}

interface MarketViewProps {
  data: {
    skills?: MarketSkill[];
    total?: number;
    vacancy_count?: number | null;
    date_from?: string | null;
    date_to?: string | null;
  };
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "–";
  const p = iso.slice(0, 10).split("-");
  return p.length === 3 ? `${p[2]}.${p[1]}.${p[0]}` : iso;
}

function fmtNum(n: number | null | undefined): string {
  if (n === null || n === undefined) return "–";
  return n.toLocaleString("ru-RU");
}

export function MarketView({ data }: MarketViewProps) {
  const skills = useMemo(
    () => [...(data.skills || [])].sort((a, b) => b.weight - a.weight),
    [data]
  );
  const top = skills.slice(0, 15);
  const maxW = top[0]?.weight || 1;
  const leader = top[0];

  const stats = [
    { icon: Briefcase, label: "Вакансий в выборке", value: fmtNum(data.vacancy_count) },
    { icon: Database, label: "Навыков в индексе", value: fmtNum(data.total) },
    {
      icon: CalendarDays,
      label: "Период данных",
      value: data.date_from || data.date_to ? `${fmtDate(data.date_from)} — ${fmtDate(data.date_to)}` : "–",
    },
    { icon: Trophy, label: "Топ-навык", value: leader ? leader.skill : "–" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s, i) => (
          <Card key={i} className="border border-slate-200 dark:border-slate-700">
            <CardContent className="pt-5">
              <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
                <s.icon className="size-4" />
                <span className="text-xs font-medium">{s.label}</span>
              </div>
              <div className="mt-1 text-xl font-bold text-slate-900 dark:text-slate-100 truncate" title={s.value}>
                {s.value}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-900/80">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-base">
            <div className="p-2 bg-blue-700 rounded-lg">
              <BarChart3 className="size-5 text-white" />
            </div>
            Топ-15 навыков рынка
          </CardTitle>
          <CardDescription>
            Вес — нормализованная метрика спроса (частота с учётом BM25 и эмбеддингов), а не число вакансий
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2.5">
            {top.map((t, i) => (
              <div key={t.skill} className="flex items-center gap-3">
                <span className="w-6 text-right text-xs font-mono text-slate-400">{i + 1}</span>
                <span className="w-44 truncate text-sm font-medium text-slate-800 dark:text-slate-200" title={t.skill}>
                  {t.skill}
                </span>
                <div className="flex-1 h-2.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(2, (t.weight / maxW) * 100)}%` }}
                    transition={{ delay: i * 0.03, duration: 0.4, ease: "easeOut" }}
                    className="h-full rounded-full bg-blue-700"
                  />
                </div>
                <span className="w-14 text-right text-xs font-mono text-slate-500 dark:text-slate-400">
                  {((t.weight / maxW) * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
          {top.length === 0 && (
            <p className="text-sm text-slate-500 dark:text-slate-400 text-center py-8">
              Нет данных. Запустите сбор вакансий.
            </p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

export default MarketView;
