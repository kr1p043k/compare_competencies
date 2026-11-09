import { useState, useEffect } from "react";
import { api } from "../api";
import { Badge, Sparkbars, StatCard } from "../../design";

type SkillTrend = {
  name: string;
  change_pct: number;
  frequency: number;
  prev_frequency: number;
  history?: { date: string; freq: number }[];
};

type CompetencyTrend = {
  code: string;
  name: string;
  skill_count: number;
  active_skills_count: number;
  change_pct: number;
  direction: string;
  skills: SkillTrend[];
};

type TrendsResponse = {
  total: number;
  trends: CompetencyTrend[];
};

type Props = {
  dirCode: string;
  competencyCodes?: string[];
};

function trendArrow(pct: number): string {
  if (pct > 0) return "↑";
  if (pct < 0) return "↓";
  return "→";
}

function trendTextClass(pct: number): string {
  if (pct > 5) return "text-success";
  if (pct < -5) return "text-destructive";
  return "text-warning";
}

function trendTone(pct: number): "success" | "warning" | "danger" {
  if (pct > 5) return "success";
  if (pct < -5) return "danger";
  return "warning";
}

/** Тон направления компетенции: рост - зелёный, падение - красное, стабильность - янтарный.
 * Раньше сюда подставлялись HEX-цвета в className (недействительные классы:
 * цвет молча терялся); токен-классы чинят это и переключают темы автоматически. */
function directionTone(direction: string): "success" | "warning" | "danger" {
  if (direction === "rising") return "success";
  if (direction === "falling") return "danger";
  return "warning";
}

function directionTextClass(direction: string): string {
  const tone = directionTone(direction);
  if (tone === "success") return "text-success";
  if (tone === "danger") return "text-destructive";
  return "text-warning";
}

export default function CompetencyTrendsPanel({ dirCode, competencyCodes }: Props) {
  const [data, setData] = useState<TrendsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ limit: "200" });
    if (filter && filter !== "all") params.set("direction", filter);
    api<TrendsResponse>(`/competency-trends?${params}`)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [dirCode, filter]);

  const noData = !data || (competencyCodes && competencyCodes.length > 0
    ? !data.trends.some((c) => competencyCodes.includes(c.code))
    : data.trends.length === 0);

  if (loading) {
    return (
      <div className="text-muted-foreground text-[13px] p-3">
        Загрузка трендов...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-destructive text-[13px] p-3">
        {error}
      </div>
    );
  }

  if (noData) {
    return (
      <div className="text-muted-foreground text-[13px] p-3">
        Нет данных по трендам. Запустите teacher analysis.
      </div>
    );
  }

  const displayed = competencyCodes
    ? data.trends.filter((c) => competencyCodes.includes(c.code))
    : data.trends;
  const rising = displayed.filter((c) => c.direction === "rising").length;
  const stable = displayed.filter((c) => c.direction === "stable").length;
  const falling = displayed.filter((c) => c.direction === "falling").length;
  const avgTrend = displayed.length
    ? displayed.reduce((s, c) => s + c.change_pct, 0) / displayed.length
    : 0;

  const filtered = data.trends
    .filter((c) => !competencyCodes || competencyCodes.includes(c.code))
    .filter((c) => !filter || filter === "all" || c.direction === filter);

  return (
    <div>
      {/* Summary bar */}
      <div className="flex flex-wrap gap-3 mb-4">
        <StatCard label="Компетенций" value={displayed.length} tone="primary" className="min-w-[110px] px-4 py-3 grow-0 basis-auto" />
        <StatCard label="Растут" value={rising} tone="success" className="min-w-[110px] px-4 py-3 grow-0 basis-auto" />
        <StatCard label="Стабильно" value={stable} tone="warning" className="min-w-[110px] px-4 py-3 grow-0 basis-auto" />
        <StatCard label="Падают" value={falling} tone="danger" className="min-w-[110px] px-4 py-3 grow-0 basis-auto" />
        <StatCard
          label="Средний тренд"
          value={`${avgTrend > 0 ? "+" : ""}${avgTrend.toFixed(1)}%`}
          tone={trendTone(avgTrend)}
          className="min-w-[110px] px-4 py-3 grow-0 basis-auto"
        />
      </div>

      {/* Filter */}
      <div
        className="flex flex-wrap gap-1.5 mb-3"
      >
        {[
          { key: null as string | null, label: `Все (${displayed.length})` },
          { key: "rising", label: `Растут (${rising})` },
          { key: "stable", label: `Стабильно (${stable})` },
          { key: "falling", label: `Падают (${falling})` },
        ].map((f) => (
          <button
            key={f.key || "all"}
            onClick={() => setFilter(f.key)}
            className={
              "cursor-pointer rounded px-3 py-1 text-xs border " +
              (filter === f.key
                ? "border-primary/60 bg-primary-soft text-primary-soft-foreground font-bold"
                : "border-border-strong bg-card text-muted-foreground font-normal hover:bg-muted")
            }
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Competency rows */}
      {filtered.map((comp) => {
        const isOpen = expanded === comp.code;
        return (
          <div
            key={comp.code}
            className="mb-2 overflow-hidden rounded-lg border border-border"
          >
            <div
              onClick={() => setExpanded(isOpen ? null : comp.code)}
              className="flex cursor-pointer items-center justify-between bg-muted px-4 py-2.5"
            >
              <div
                className="flex items-center gap-2.5"
              >
                <span
                  className="text-sm font-semibold text-primary"
                >
                  {comp.code}
                </span>
                <span className="text-xs text-muted-foreground">
                  {comp.skill_count} навык{comp.skill_count !== 1 ? "а" : ""}
                  {comp.active_skills_count !== undefined &&
                    comp.active_skills_count !== comp.skill_count && (
                      <span
                        className="ml-1 text-xs text-subtle-foreground"
                      >
                        ({comp.active_skills_count} в тренде)
                      </span>
                    )}
                </span>
              </div>
              <div
                className="flex items-center gap-2"
              >
                <span
                  className={`text-[15px] font-bold ${directionTextClass(comp.direction)}`}
                >
                  {trendArrow(comp.change_pct)}{" "}
                  {comp.change_pct > 0 ? "+" : ""}
                  {comp.change_pct}%
                </span>
                <Badge tone={directionTone(comp.direction)}>
                  {comp.direction}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {isOpen ? "v" : ">"}
                </span>
              </div>
            </div>

            {/* Expanded per-skill trends */}
            {isOpen && (
              <div className="px-4 pt-2 pb-3">
                {comp.skills.length === 0 && (
                  <div className="text-muted-foreground text-xs">
                    Навыки не извлечены
                  </div>
                )}
                {comp.skills.map((sk, i) => (
                  <div key={i}>
                    <div
                      className="flex items-center justify-between border-b border-border py-1.5 text-[13px]"
                    >
                      <span className="text-foreground font-medium">{sk.name}</span>
                      <span
                        className={`text-[13px] font-semibold ${trendTextClass(sk.change_pct)}`}
                      >
                        {trendArrow(sk.change_pct)}{" "}
                        {sk.change_pct > 0 ? "+" : ""}
                        {sk.change_pct}%
                        <span
                          className="ml-1.5 text-xs font-normal text-muted-foreground"
                        >
                          (freq: {sk.frequency})
                        </span>
                      </span>
                    </div>
                    {/* History chart */}
                    {sk.history && sk.history.length > 1 && (
                      <div
                        className="pb-4 pt-3 text-xs text-muted-foreground"
                      >
                        <Sparkbars
                          items={[...sk.history].reverse().map((h, hi) => ({
                            key: hi,
                            value: h.freq,
                            label: h.date.slice(5),
                            title: `${h.date}: ${h.freq}`,
                          }))}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
