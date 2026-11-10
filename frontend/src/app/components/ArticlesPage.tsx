import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Meter } from "../../design";
import { cn } from "./ui/utils";

// Соответствие CSS-значений цветов уровня классам-токенам (вместо inline-стилей).
const FILL_BY_COLOR: Record<string, string> = {
  "var(--chart-1)": "bg-chart-1",
  "var(--chart-2)": "bg-chart-2",
  "var(--chart-3)": "bg-chart-3",
  "var(--success)": "bg-success",
  "var(--destructive)": "bg-destructive",
};
import {
  Newspaper,
  Briefcase,
  Wallet,
  TrendingUp,
  Layers,
  BookOpen,
  AlertCircle,
  RefreshCw,
  BarChart3,
  ListChecks,
  MapPin,
  Building2,
} from "lucide-react";
import { api } from "../api";
import { AnalyticsCharts } from "./AnalyticsCharts";
import { ShowMore } from "./ui/show-more";
import { PageHeader } from "./PageHeader";

const fmt = new Intl.NumberFormat("ru-RU");

function formatSalary(value: number): string {
  if (!value) return "-";
  return `${fmt.format(Math.round(value))} ₽`;
}

type VacancyStats = {
  total: number;
  by_experience: { junior: number; middle: number; senior: number };
  salary: { average: number; min: number; max: number; count: number };
};

type TopSkill = { skill: string; weight: number; frequency?: number };

type TaxonomyCoverage = {
  coverage: Record<string, { label: string; icon: string; total: number; covered: number; percent: number }>;
};

type ProfessionTrends = {
  profession: string;
  source?: string;
  snapshot_date?: string;
  skills: { skill: string; frequency: number }[];
};

/** Элемент /trends/professions: профессия + её последний снапшот. */
type ProfessionOption = {
  name: string;
  snapshot?: string;
  date?: string;
};

type Analytics = {
  salary_by_level: Record<string, { count: number; average: number }>;
  top_regions: { name: string; count: number }[];
  top_employers: { name: string; count: number }[];
  skills: {
    with_skills: number;
    total: number;
    percent: number;
    avg_skills_per_vacancy: number;
  };
};

type SkillTrend = {
  skill: string;
  current_freq: number;
  prev_freq: number;
  change_pct: number;
  prev_label: string;
};

type TrendsData = {
  rising: SkillTrend[];
  falling: SkillTrend[];
};

const EXP_LEVELS = [
  { key: "junior", label: "Junior", color: "var(--chart-1)" },
  { key: "middle", label: "Middle", color: "var(--chart-2)" },
  { key: "senior", label: "Senior", color: "var(--chart-3)" },
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
      <div className="w-28 shrink-0 text-sm text-muted-foreground truncate text-right" title={label}>
        {label}
      </div>
      <div className="flex-1 h-6 bg-muted rounded overflow-hidden">
        <Meter
          value={width}
          className="h-full"
          trackClassName="bg-transparent"
          fillClassName={cn("rounded transition-all duration-700", FILL_BY_COLOR[color] ?? "bg-primary")}
        />
      </div>
      <div className="w-24 shrink-0 text-sm font-medium text-foreground tabular-nums">
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
    <Card className="border border-border shadow-sm">
      <CardHeader className="border-b border-border bg-muted">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-9 h-9 bg-primary rounded-lg shrink-0">
            <Icon className="size-5 text-white" />
          </div>
          <div className="min-w-0">
            <CardTitle className="text-lg font-semibold text-foreground">{title}</CardTitle>
            <CardDescription className="text-sm text-muted-foreground">{description}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-6">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-10 text-subtle-foreground">
            <RefreshCw className="size-8 mb-3 animate-spin" />
            <p className="text-sm">Загрузка данных...</p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <AlertCircle className="size-8 mb-3 text-warning" />
            <p className="text-sm text-muted-foreground">Данные временно недоступны.</p>
            <p className="text-xs text-subtle-foreground mt-1 max-w-md">{error}</p>
            {onRetry && (
              <button
                onClick={onRetry}
                className="mt-4 inline-flex items-center gap-2 text-sm text-primary hover:text-primary/80 font-medium"
              >
                <RefreshCw className="size-4" /> Обновить
              </button>
            )}
          </div>
        ) : empty ? (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <BarChart3 className="size-8 mb-3 text-subtle-foreground" />
            <p className="text-sm text-muted-foreground">Данные появятся после запуска пайплайна анализа.</p>
            <p className="text-xs text-subtle-foreground mt-1">Пока бэкенд прогревается, аналитические метрики не рассчитаны.</p>
          </div>
        ) : (
          children
        )}
      </CardContent>
    </Card>
  );
}

export function ArticlesPage({ onStartGapAnalysis }: { onStartGapAnalysis?: () => void }) {
  const [stats, setStats] = useState<VacancyStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState<string | null>(null);

  const [topSkills, setTopSkills] = useState<TopSkill[]>([]);
  const [skillsLoading, setSkillsLoading] = useState(true);
  const [skillsError, setSkillsError] = useState<string | null>(null);

  const [coverage, setCoverage] = useState<TaxonomyCoverage["coverage"] | null>(null);
  const [coverageLoading, setCoverageLoading] = useState(true);
  const [coverageError, setCoverageError] = useState<string | null>(null);

  const [professions, setProfessions] = useState<ProfessionOption[]>([]);
  const [profError, setProfError] = useState<string | null>(null);
  const [selectedProf, setSelectedProf] = useState("");
  const [profTrends, setProfTrends] = useState<ProfessionTrends | null>(null);
  const [profTrendsLoading, setProfTrendsLoading] = useState(false);

  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [showAllRegions, setShowAllRegions] = useState(false);
  const [showAllRising, setShowAllRising] = useState(false);
  const [showAllFalling, setShowAllFalling] = useState(false);
  const [showAllTopSkills, setShowAllTopSkills] = useState(false);
  const [showAllCoverage, setShowAllCoverage] = useState(false);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);
  const [analyticsError, setAnalyticsError] = useState<string | null>(null);

  const [trends, setTrends] = useState<TrendsData | null>(null);
  const [trendsLoading, setTrendsLoading] = useState(true);
  const [trendsError, setTrendsError] = useState<string | null>(null);

  const loadAnalytics = () => {
    setAnalyticsLoading(true);
    setAnalyticsError(null);
    api<Analytics>("/vacancies/stats/analytics")
      .then(setAnalytics)
      .catch((e) => setAnalyticsError(e.message))
      .finally(() => setAnalyticsLoading(false));
  };

  const loadTrends = () => {
    setTrendsLoading(true);
    setTrendsError(null);
    api<{ trends: TrendsData }>("/trends?top_n=10&min_change=3.0")
      .then((d) => setTrends(d.trends || { rising: [], falling: [] }))
      .catch((e) => setTrendsError(e.message))
      .finally(() => setTrendsLoading(false));
  };

  const loadStats = () => {
    setStatsLoading(true);
    setStatsError(null);
    api<VacancyStats>("/vacancies/stats/summary")
      .then(setStats)
      .catch((e) => setStatsError(e.message))
      .finally(() => setStatsLoading(false));
  };

  const loadTopSkills = () => {
    setSkillsLoading(true);
    setSkillsError(null);
    api<{ skills: TopSkill[] }>("/market/top-skills?limit=15")
      .then((d) => setTopSkills(d.skills || []))
      .catch((e) => setSkillsError(e.message))
      .finally(() => setSkillsLoading(false));
  };

  const loadCoverage = () => {
    setCoverageLoading(true);
    setCoverageError(null);
    api<TaxonomyCoverage>("/taxonomy/coverage")
      .then((d) => setCoverage(d.coverage || {}))
      .catch((e) => setCoverageError(e.message))
      .finally(() => setCoverageLoading(false));
  };

  const loadProfessions = () => {
    setProfError(null);
    api<{ professions: ProfessionOption[] }>("/trends/professions")
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
    api<ProfessionTrends>(`/trends/by-profession?profession=${encodeURIComponent(prof)}&limit=30`)
      .then(setProfTrends)
      .catch(() => setProfTrends(null))
      .finally(() => setProfTrendsLoading(false));
  };

  useEffect(() => {
    loadStats();
    loadAnalytics();
    loadTrends();
    loadTopSkills();
    loadCoverage();
    loadProfessions();
  }, []);

  const statsMax = Math.max(
    stats?.by_experience.junior ?? 0,
    stats?.by_experience.middle ?? 0,
    stats?.by_experience.senior ?? 0,
  );

    const skillsMax = topSkills.length ? Math.max(...topSkills.map((s) => s.frequency ?? s.weight)) : 0;

  const salaryMax = analytics
    ? Math.max(
        ...EXP_LEVELS.map((l) => analytics.salary_by_level[l.key]?.average ?? 0),
        1,
      )
    : 1;

  const regionMax = analytics?.top_regions[0]?.count ?? 1;
  const employerMax = analytics?.top_employers[0]?.count ?? 1;

  const risingMax = trends?.rising.length
    ? Math.max(...trends.rising.map((t) => Math.abs(t.change_pct)), 1)
    : 1;
  const fallingMax = trends?.falling.length
    ? Math.max(...trends.falling.map((t) => Math.abs(t.change_pct)), 1)
    : 1;

  const coverageRows = coverage
    ? Object.entries(coverage)
        .map(([id, c]) => ({ id, ...c }))
        .sort((a, b) => b.percent - a.percent)
    : [];

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Newspaper}
        title="Статьи рынка"
        description="Обзоры вакансий и компетенций по разделам рынка."
      />

      {/* KPI cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="border border-border shadow-sm">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 bg-primary-soft rounded-xl">
              <Briefcase className="size-6 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Всего вакансий на рынке</p>
              {statsLoading ? (
                <div className="h-7 w-28 bg-muted rounded mt-1" />
              ) : (
                <p className="text-2xl font-bold text-foreground tabular-nums">
                  {stats ? fmt.format(stats.total) : "-"}
                </p>
              )}
            </div>
          </CardContent>
        </Card>
        <Card className="border border-border shadow-sm">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 bg-success-soft rounded-xl">
              <Wallet className="size-6 text-success" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Средняя зарплата</p>
              {statsLoading ? (
                <div className="h-7 w-28 bg-muted rounded mt-1" />
              ) : (
                <p className="text-2xl font-bold text-foreground tabular-nums">
                  {stats ? formatSalary(stats.salary.average) : "-"}
                </p>
              )}
            </div>
          </CardContent>
        </Card>
        <Card className="border border-border shadow-sm">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 bg-info-soft rounded-xl">
              <TrendingUp className="size-6 text-info" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Вакансий с указанной зарплатой</p>
              {statsLoading ? (
                <div className="h-7 w-28 bg-muted rounded mt-1" />
              ) : (
                <p className="text-2xl font-bold text-foreground tabular-nums">
                  {stats ? fmt.format(stats.salary.count) : "-"}
                </p>
              )}
            </div>
          </CardContent>
        </Card>
        <Card className="border border-border shadow-sm">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 bg-warning-soft rounded-xl">
              <ListChecks className="size-6 text-warning" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Вакансии с навыками</p>
              {analyticsLoading ? (
                <div className="h-7 w-28 bg-muted rounded mt-1" />
              ) : (
                <p className="text-2xl font-bold text-foreground tabular-nums">
                  {analytics
                    ? fmt.format(analytics.skills.with_skills)
                    : "-"}
                </p>
              )}
              {!analyticsLoading && analytics && (
                <p className="text-xs text-subtle-foreground mt-1">
                  в среднем {analytics.skills.avg_skills_per_vacancy} навыка на вакансию
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Salary by level + top regions */}
      <div className="grid gap-6 lg:grid-cols-2">
        <BlockCard
          icon={Wallet}
          title="Средняя зарплата по уровню опыта"
          description="Средняя зарплата в вакансиях с указанным окладом по уровням Junior / Middle / Senior"
          loading={analyticsLoading}
          error={analyticsError}
          onRetry={loadAnalytics}
          empty={!analytics || Object.keys(analytics.salary_by_level).length === 0}
        >
          {analytics && (
            <div className="space-y-3">
              {EXP_LEVELS.map(({ key, label, color }) => {
                const sal = analytics.salary_by_level[key];
                return (
                  <BarRow
                    key={key}
                    label={label}
                    value={sal?.average ?? 0}
                    max={salaryMax}
                    color={color}
                    valueText={formatSalary(sal?.average ?? 0)}
                  />
                );
              })}
              <div className="pt-4 border-t border-border flex justify-between text-sm text-muted-foreground">
                <span>Всего вакансий с окладом</span>
                <span className="font-medium text-foreground tabular-nums">
                  {fmt.format(stats?.salary.count ?? 0)}
                </span>
              </div>
            </div>
          )}
        </BlockCard>

        <BlockCard
          icon={MapPin}
          title="Топ регионов по спросу"
          description="Регионы с наибольшим числом собранных вакансий"
          loading={analyticsLoading}
          error={analyticsError}
          onRetry={loadAnalytics}
          empty={!analytics || analytics.top_regions.length === 0}
        >
          {analytics && (
            <div className="space-y-2.5">
              {(showAllRegions ? analytics.top_regions : analytics.top_regions.slice(0, 5)).map((r) => (
                <BarRow
                  key={r.name}
                  label={r.name}
                  value={r.count}
                  max={regionMax}
                  color="var(--chart-1)"
                />
              ))}
              <ShowMore total={analytics.top_regions.length} shown={5} expanded={showAllRegions} onToggle={() => setShowAllRegions((v) => !v)} />
            </div>
          )}
        </BlockCard>
      </div>

      {/* Top employers + demand dynamics */}
      <div className="grid gap-6 lg:grid-cols-2">
        <BlockCard
          icon={Building2}
          title="Топ работодателей"
          description="Компании с наибольшим числом размещённых вакансий"
          loading={analyticsLoading}
          error={analyticsError}
          onRetry={loadAnalytics}
          empty={!analytics || analytics.top_employers.length === 0}
        >
          {analytics && (
            <div className="space-y-2.5">
              {analytics.top_employers.map((e) => (
                <BarRow
                  key={e.name}
                  label={e.name}
                  value={e.count}
                  max={employerMax}
                  color="var(--success)"
                />
              ))}
            </div>
          )}
        </BlockCard>

        <BlockCard
          icon={TrendingUp}
          title="Динамика спроса на навыки"
          description="Навыки с наибольшим ростом и падением частоты между последними снапшотами"
          loading={trendsLoading}
          error={trendsError}
          onRetry={loadTrends}
          empty={!trends || (trends.rising.length === 0 && trends.falling.length === 0)}
        >
          {trends && (
            <div className="space-y-4">
              {trends.rising.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-success">Растут</span>
                    <span className="text-xs text-subtle-foreground">
                      относительно снапшота {trends.rising[0]?.prev_label}
                    </span>
                  </div>
                  <div className="space-y-2">
                    {(showAllRising ? trends.rising : trends.rising.slice(0, 5)).map((t) => (
                      <BarRow
                        key={t.skill}
                        label={t.skill}
                        value={Math.abs(t.change_pct)}
                        max={risingMax}
                        color="var(--success)"
                        valueText={`+${t.change_pct.toFixed(1)}%`}
                      />
                    ))}
                    <ShowMore total={trends.rising.length} shown={5} expanded={showAllRising} onToggle={() => setShowAllRising((v) => !v)} />
                  </div>
                </div>
              )}
              {trends.falling.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-destructive">Падают</span>
                    <span className="text-xs text-subtle-foreground">
                      относительно снапшота {trends.falling[0]?.prev_label}
                    </span>
                  </div>
                  <div className="space-y-2">
                    {(showAllFalling ? trends.falling : trends.falling.slice(0, 5)).map((t) => (
                      <BarRow
                        key={t.skill}
                        label={t.skill}
                        value={Math.abs(t.change_pct)}
                        max={fallingMax}
                        color="var(--destructive)"
                        valueText={`${t.change_pct.toFixed(1)}%`}
                      />
                    ))}
                    <ShowMore total={trends.falling.length} shown={5} expanded={showAllFalling} onToggle={() => setShowAllFalling((v) => !v)} />
                  </div>
                </div>
              )}
            </div>
          )}
        </BlockCard>
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
              <div className="pt-4 border-t border-border flex justify-between text-sm text-muted-foreground">
                <span>Всего в базе</span>
                <span className="font-medium text-foreground tabular-nums">{fmt.format(stats.total)}</span>
              </div>
            </div>
          )}
        </BlockCard>

        <BlockCard
          icon={TrendingUp}
          title="Топ востребованных навыков"
          description="Число упоминаний навыка в вакансиях (вес - нормализованная метрика)"
          loading={skillsLoading}
          error={skillsError}
          onRetry={loadTopSkills}
          empty={topSkills.length === 0}
        >
          <div className="space-y-2.5">
            {(showAllTopSkills ? topSkills : topSkills.slice(0, 5)).map((s) => (
              <BarRow
                key={s.skill}
                label={s.skill}
                value={s.frequency ?? s.weight}
                max={skillsMax}
                color="var(--chart-2)"
                valueText={s.frequency != null ? fmt.format(s.frequency) : s.weight.toFixed(3)}
              />
            ))}
            <ShowMore total={topSkills.length} shown={5} expanded={showAllTopSkills} onToggle={() => setShowAllTopSkills((v) => !v)} />
          </div>
        </BlockCard>
      </div>

      {/* Интерактивные графики */}
      <AnalyticsCharts onStartGapAnalysis={onStartGapAnalysis} />

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
          {(showAllCoverage ? coverageRows : coverageRows.slice(0, 5)).map((c) => (
            <BarRow
              key={c.id}
              label={c.label}
              value={c.percent}
              max={100}
              color="var(--chart-1)"
              suffix="%"
              valueText={`${c.percent.toFixed(1)}%`}
            />
          ))}
          <ShowMore
            total={coverageRows.length}
            shown={5}
            expanded={showAllCoverage}
            onToggle={() => setShowAllCoverage((v) => !v)}
          />
        </div>
      </BlockCard>

      {/* Profession trends */}
      <Card className="border border-border shadow-sm">
        <CardHeader className="border-b border-border bg-muted">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 bg-primary rounded-lg">
              <BookOpen className="size-5 text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-lg font-semibold text-foreground">Тренды по профессиям</CardTitle>
              <CardDescription className="text-sm text-muted-foreground">
                Топ навыков из последнего снапшота анализа по выбранной профессии
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          {profError ? (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <AlertCircle className="size-8 mb-3 text-warning" />
              <p className="text-sm text-muted-foreground">Данные временно недоступны.</p>
              <p className="text-xs text-subtle-foreground mt-1 max-w-md">{profError}</p>
              <button
                onClick={loadProfessions}
                className="mt-4 inline-flex items-center gap-2 text-sm text-primary hover:text-primary/80 font-medium"
              >
                <RefreshCw className="size-4" /> Обновить
              </button>
            </div>
          ) : professions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <BookOpen className="size-8 mb-3 text-subtle-foreground" aria-hidden="true" />
              <p className="text-sm text-muted-foreground">
                Данные по профессиям появятся после запуска анализа.
              </p>
            </div>
          ) : (
            <div className="space-y-5">
              <Select value={selectedProf} onValueChange={loadProfTrends}>
                <SelectTrigger className="w-full md:w-80" aria-label="Профессия для трендов">
                  <SelectValue placeholder="Выберите профессию" />
                </SelectTrigger>
                <SelectContent>
                  {professions.map((p) => (
                    <SelectItem key={p.name} value={p.name}>
                      {p.date ? `${p.name} · ${p.date}` : p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {profTrendsLoading ? (
                <div className="flex flex-col items-center justify-center py-8 text-subtle-foreground">
                  <RefreshCw className="size-7 mb-3 animate-spin" />
                  <p className="text-sm">Загрузка навыков профессии...</p>
                </div>
              ) : profTrends && profTrends.skills.length > 0 ? (
                <div className="space-y-2.5">
                  {(profTrends.source === "live" || profTrends.snapshot_date) && (
                    <p className="text-xs text-subtle-foreground">
                      {profTrends.source === "live"
                        ? "Живой расчёт по базе: снапшота для этой профессии ещё нет."
                        : `Снапшот от ${profTrends.snapshot_date}`}
                    </p>
                  )}
                  {profTrends.skills.slice(0, 15).map((s, i) => {
                    const max = profTrends!.skills[0]?.frequency || 1;
                    return (
                      <BarRow
                        key={`${s.skill}-${i}`}
                        label={s.skill}
                        value={s.frequency}
                        max={max}
                        color="var(--success)"
                        valueText={String(s.frequency)}
                      />
                    );
                  })}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <BarChart3 className="size-7 mb-3 text-subtle-foreground" />
                  <p className="text-sm text-muted-foreground">Навыки по этой профессии пока не рассчитаны.</p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
