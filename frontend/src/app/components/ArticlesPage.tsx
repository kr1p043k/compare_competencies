import { useEffect, useState } from "react";
import { motion } from "motion/react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import {
  Newspaper,
  Briefcase,
  Wallet,
  TrendingUp,
  Radar as RadarIcon,
  Layers,
  BookOpen,
  AlertCircle,
  RefreshCw,
  BarChart3,
} from "lucide-react";
import { api } from "../api";

const fmt = new Intl.NumberFormat("ru-RU");

function formatSalary(value: number): string {
  if (!value) return "—";
  return `${fmt.format(Math.round(value))} ₽`;
}

type VacancyStats = {
  total: number;
  by_experience: { junior: number; middle: number; senior: number };
  salary: { average: number; min: number; max: number; count: number };
};

type TopSkill = { skill: string; weight: number };

type TaxonomyCoverage = {
  coverage: Record<string, { label: string; icon: string; total: number; covered: number; percent: number }>;
};

type ProfessionsResponse = {
  professions: { name: string; snapshot: string; date: string }[];
};

type ProfessionTrends = {
  profession: string;
  source?: string;
  snapshot_date?: string;
  skills: { skill: string; frequency: number }[];
};

type ImageEntry = {
  src: string;
  title: string;
  description: string;
};

const EXP_LEVELS = [
  { key: "junior", label: "Junior", color: "#3b82f6" },
  { key: "middle", label: "Middle", color: "#8b5cf6" },
  { key: "senior", label: "Senior", color: "#059669" },
] as const;

function BarRow({
  label,
  value,
  max,
  color,
  suffix = "",
  valueText,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
  suffix?: string;
  valueText?: string;
}) {
  const width = max > 0 ? Math.max((value / max) * 100, value > 0 ? 2 : 0) : 0;
  return (
    <div className="flex items-center gap-3">
      <div className="w-28 shrink-0 text-sm text-gray-600 truncate text-right" title={label}>
        {label}
      </div>
      <div className="flex-1 h-6 bg-gray-100 rounded overflow-hidden">
        <div
          className="h-full rounded transition-all duration-700"
          style={{ width: `${width}%`, backgroundColor: color }}
        />
      </div>
      <div className="w-24 shrink-0 text-sm font-medium text-gray-800 tabular-nums">
        {valueText ?? `${fmt.format(value)}${suffix}`}
      </div>
    </div>
  );
}

function BlockCard({
  icon: Icon,
  title,
  description,
  loading,
  error,
  empty,
  onRetry,
  children,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  loading: boolean;
  error: string | null;
  empty?: boolean;
  onRetry?: () => void;
  children: React.ReactNode;
}) {
  return (
    <Card className="border border-gray-200 shadow-sm">
      <CardHeader className="border-b border-gray-200 bg-gray-50">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-9 h-9 bg-blue-600 rounded-lg shrink-0">
            <Icon className="size-5 text-white" />
          </div>
          <div className="min-w-0">
            <CardTitle className="text-lg font-semibold text-gray-900">{title}</CardTitle>
            <CardDescription className="text-sm text-gray-600">{description}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-6">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-10 text-gray-400">
            <RefreshCw className="size-8 mb-3 animate-spin" />
            <p className="text-sm">Загрузка данных...</p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <AlertCircle className="size-8 mb-3 text-amber-500" />
            <p className="text-sm text-gray-600">Данные временно недоступны.</p>
            <p className="text-xs text-gray-400 mt-1 max-w-md">{error}</p>
            {onRetry && (
              <button
                onClick={onRetry}
                className="mt-4 inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
              >
                <RefreshCw className="size-4" /> Обновить
              </button>
            )}
          </div>
        ) : empty ? (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <BarChart3 className="size-8 mb-3 text-gray-300" />
            <p className="text-sm text-gray-500">Данные появятся после запуска пайплайна анализа.</p>
            <p className="text-xs text-gray-400 mt-1">Пока бэкенд прогревается, аналитические метрики не рассчитаны.</p>
          </div>
        ) : (
          children
        )}
      </CardContent>
    </Card>
  );
}

const IMAGES: ImageEntry[] = [
  {
    src: "/api/results/images/base/radar",
    title: "Радар эталонных профилей",
    description: "Сравнение профилей уровней Junior / Middle / Senior по ключевым навыкам",
  },
  {
    src: "/api/results/images/coverage-comparison",
    title: "Покрытие рынка по уровням",
    description: "Доля навыков, покрываемых эталонными профилями каждого уровня",
  },
  {
    src: "/api/results/images/skills-heatmap",
    title: "Тепловая карта навыков",
    description: "Распределение навыков по категориям таксономии и уровням",
  },
];

export function ArticlesPage() {
  const [stats, setStats] = useState<VacancyStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState<string | null>(null);

  const [topSkills, setTopSkills] = useState<TopSkill[]>([]);
  const [skillsLoading, setSkillsLoading] = useState(true);
  const [skillsError, setSkillsError] = useState<string | null>(null);

  const [coverage, setCoverage] = useState<TaxonomyCoverage["coverage"] | null>(null);
  const [coverageLoading, setCoverageLoading] = useState(true);
  const [coverageError, setCoverageError] = useState<string | null>(null);

  const [professions, setProfessions] = useState<{ name: string }[]>([]);
  const [profError, setProfError] = useState<string | null>(null);
  const [selectedProf, setSelectedProf] = useState("");
  const [profTrends, setProfTrends] = useState<ProfessionTrends | null>(null);
  const [profTrendsLoading, setProfTrendsLoading] = useState(false);

  const [images, setImages] = useState(IMAGES.map((i) => ({ ...i, broken: false })));

  const loadStats = () => {
    setStatsLoading(true);
    setStatsError(null);
    api("/vacancies/stats/summary")
      .then(setStats)
      .catch((e) => setStatsError(e.message))
      .finally(() => setStatsLoading(false));
  };

  const loadTopSkills = () => {
    setSkillsLoading(true);
    setSkillsError(null);
    api("/market/top-skills?limit=15")
      .then((d) => setTopSkills(d.skills || []))
      .catch((e) => setSkillsError(e.message))
      .finally(() => setSkillsLoading(false));
  };

  const loadCoverage = () => {
    setCoverageLoading(true);
    setCoverageError(null);
    api("/taxonomy/coverage")
      .then((d) => setCoverage(d.coverage || {}))
      .catch((e) => setCoverageError(e.message))
      .finally(() => setCoverageLoading(false));
  };

  const loadProfessions = () => {
    setProfError(null);
    api("/trends/professions")
      .then((d) => setProfessions(d.professions || []))
      .catch((e) => setProfError(e.message));
  };

  const loadProfTrends = (prof: string) => {
    setSelectedProf(prof);
    if (!prof) {
      setProfTrends(null);
      return;
    }
    setProfTrendsLoading(true);
    api(`/trends/by-profession?profession=${encodeURIComponent(prof)}&limit=30`)
      .then(setProfTrends)
      .catch(() => setProfTrends(null))
      .finally(() => setProfTrendsLoading(false));
  };

  useEffect(() => {
    loadStats();
    loadTopSkills();
    loadCoverage();
    loadProfessions();
  }, []);

  const statsMax = Math.max(
    stats?.by_experience.junior ?? 0,
    stats?.by_experience.middle ?? 0,
    stats?.by_experience.senior ?? 0,
  );

  const skillsMax = topSkills.length ? Math.max(...topSkills.map((s) => s.weight)) : 0;

  const coverageRows = coverage
    ? Object.entries(coverage)
        .map(([id, c]) => ({ id, ...c }))
        .sort((a, b) => b.percent - a.percent)
    : [];

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center space-y-4">
        <div className="inline-flex items-center justify-center gap-3 mb-2">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl blur-xl opacity-50 animate-pulse" />
            <div className="relative bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 rounded-2xl shadow-2xl">
              <Newspaper className="size-8 text-white" />
            </div>
          </div>
          <div className="text-left">
            <h2 className="text-3xl font-bold text-gray-900">Статьи и аналитика</h2>
            <p className="text-gray-600">Статистика рынка вакансий и рекомендации по развитию навыков</p>
          </div>
        </div>
      </motion.div>

      {/* KPI cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border border-gray-200 shadow-sm">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 bg-blue-100 rounded-xl">
              <Briefcase className="size-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Всего вакансий на рынке</p>
              {statsLoading ? (
                <div className="h-7 w-28 bg-gray-200 animate-pulse rounded mt-1" />
              ) : (
                <p className="text-2xl font-bold text-gray-900 tabular-nums">
                  {stats ? fmt.format(stats.total) : "—"}
                </p>
              )}
            </div>
          </CardContent>
        </Card>
        <Card className="border border-gray-200 shadow-sm">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 bg-green-100 rounded-xl">
              <Wallet className="size-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Средняя зарплата</p>
              {statsLoading ? (
                <div className="h-7 w-28 bg-gray-200 animate-pulse rounded mt-1" />
              ) : (
                <p className="text-2xl font-bold text-gray-900 tabular-nums">
                  {stats ? formatSalary(stats.salary.average) : "—"}
                </p>
              )}
            </div>
          </CardContent>
        </Card>
        <Card className="border border-gray-200 shadow-sm">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 bg-purple-100 rounded-xl">
              <TrendingUp className="size-6 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Вакансий с указанной зарплатой</p>
              {statsLoading ? (
                <div className="h-7 w-28 bg-gray-200 animate-pulse rounded mt-1" />
              ) : (
                <p className="text-2xl font-bold text-gray-900 tabular-nums">
                  {stats ? fmt.format(stats.salary.count) : "—"}
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Vacancies by experience + top skills */}
      <div className="grid gap-6 lg:grid-cols-2">
        <BlockCard
          icon={Briefcase}
          title="Вакансии по уровню опыта"
          description="Распределение собранных вакансий по уровням Junior / Middle / Senior"
          loading={statsLoading}
          error={statsError}
          onRetry={loadStats}
          empty={!stats || stats.total === 0}
        >
          {stats && (
            <div className="space-y-3">
              {EXP_LEVELS.map(({ key, label, color }) => (
                <BarRow
                  key={key}
                  label={label}
                  value={stats.by_experience[key]}
                  max={statsMax}
                  color={color}
                />
              ))}
              <div className="pt-4 border-t border-gray-100 flex justify-between text-sm text-gray-500">
                <span>Всего в базе</span>
                <span className="font-medium text-gray-800 tabular-nums">{fmt.format(stats.total)}</span>
              </div>
            </div>
          )}
        </BlockCard>

        <BlockCard
          icon={TrendingUp}
          title="Топ востребованных навыков"
          description="Навыки с наибольшим весом на рынке (частота упоминаний в вакансиях)"
          loading={skillsLoading}
          error={skillsError}
          onRetry={loadTopSkills}
          empty={topSkills.length === 0}
        >
          <div className="space-y-2.5">
            {topSkills.map((s) => (
              <BarRow
                key={s.skill}
                label={s.skill}
                value={s.weight}
                max={skillsMax}
                color="#8b5cf6"
                valueText={s.weight.toFixed(3)}
              />
            ))}
          </div>
        </BlockCard>
      </div>

      {/* Embedded images */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <div className="flex items-center justify-center w-9 h-9 bg-blue-600 rounded-lg">
            <RadarIcon className="size-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Аналитические графики</h3>
            <p className="text-sm text-gray-600">Визуализация профилей компетенций по результатам анализа</p>
          </div>
        </div>
        <div className="grid gap-6 lg:grid-cols-3">
          {images.map((img) => (
            <Card key={img.src} className="border border-gray-200 shadow-sm overflow-hidden">
              {img.broken ? (
                <CardContent className="p-6 flex flex-col items-center justify-center text-center">
                  <AlertCircle className="size-8 mb-3 text-amber-500" />
                  <p className="text-sm text-gray-500">График ещё не сгенерирован.</p>
                </CardContent>
              ) : (
                <>
                  <div className="bg-gray-50 border-b border-gray-200 flex items-center justify-center h-52">
                    <img
                      src={img.src}
                      alt={img.title}
                      className="max-h-full max-w-full object-contain"
                      onError={() => setImages((prev) => prev.map((p) => (p.src === img.src ? { ...p, broken: true } : p)))}
                    />
                  </div>
                  <CardHeader>
                    <CardTitle className="text-base font-semibold text-gray-900">{img.title}</CardTitle>
                    <CardDescription className="text-sm text-gray-600">{img.description}</CardDescription>
                  </CardHeader>
                </>
              )}
            </Card>
          ))}
        </div>
      </div>

      {/* Taxonomy coverage */}
      <BlockCard
        icon={Layers}
        title="Покрытие таксономии навыков"
        description="Процент навыков каждой категории, присутствующих в вакансиях рынка"
        loading={coverageLoading}
        error={coverageError}
        onRetry={loadCoverage}
        empty={coverageRows.length === 0}
      >
        <div className="space-y-2.5">
          {coverageRows.map((c) => (
            <BarRow
              key={c.id}
              label={c.label}
              value={c.percent}
              max={100}
              color="#3b82f6"
              suffix="%"
              valueText={`${c.percent.toFixed(1)}%`}
            />
          ))}
        </div>
      </BlockCard>

      {/* Profession trends */}
      <Card className="border border-gray-200 shadow-sm">
        <CardHeader className="border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 bg-blue-600 rounded-lg">
              <BookOpen className="size-5 text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-lg font-semibold text-gray-900">Тренды по профессиям</CardTitle>
              <CardDescription className="text-sm text-gray-600">
                Топ навыков из последнего снапшота анализа по выбранной профессии
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          {profError ? (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <AlertCircle className="size-8 mb-3 text-amber-500" />
              <p className="text-sm text-gray-600">Данные временно недоступны.</p>
              <p className="text-xs text-gray-400 mt-1 max-w-md">{profError}</p>
              <button
                onClick={loadProfessions}
                className="mt-4 inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
              >
                <RefreshCw className="size-4" /> Обновить
              </button>
            </div>
          ) : professions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <BookOpen className="size-8 mb-3 text-gray-300" />
              <p className="text-sm text-gray-500">Снапшоты профессий ещё не сформированы.</p>
              <p className="text-xs text-gray-400 mt-1">Данные появятся после первого запуска пайплайна анализа.</p>
            </div>
          ) : (
            <div className="space-y-5">
              <Select value={selectedProf} onValueChange={loadProfTrends}>
                <SelectTrigger className="w-full md:w-80">
                  <SelectValue placeholder="Выберите профессию" />
                </SelectTrigger>
                <SelectContent>
                  {professions.map((p) => (
                    <SelectItem key={p.name} value={p.name}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {profTrendsLoading ? (
                <div className="flex flex-col items-center justify-center py-8 text-gray-400">
                  <RefreshCw className="size-7 mb-3 animate-spin" />
                  <p className="text-sm">Загрузка навыков профессии...</p>
                </div>
              ) : profTrends && profTrends.skills.length > 0 ? (
                <div className="space-y-2.5">
                  {profTrends.skills.slice(0, 15).map((s, i) => {
                    const max = profTrends!.skills[0]?.frequency || 1;
                    return (
                      <BarRow
                        key={`${s.skill}-${i}`}
                        label={s.skill}
                        value={s.frequency}
                        max={max}
                        color="#059669"
                        valueText={String(s.frequency)}
                      />
                    );
                  })}
                  {profTrends.source === "snapshot" && profTrends.snapshot_date && (
                    <div className="pt-3 text-xs text-gray-400">
                      Источник: снапшот от {new Date(profTrends.snapshot_date).toLocaleDateString("ru-RU")}
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <BarChart3 className="size-7 mb-3 text-gray-300" />
                  <p className="text-sm text-gray-500">Навыки по этой профессии пока не рассчитаны.</p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
