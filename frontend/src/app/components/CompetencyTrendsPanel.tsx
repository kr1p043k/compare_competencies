import { useState, useEffect } from "react";
import { api } from "../api";
import { useTheme } from "../../lib/theme";

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
  if (pct > 5) return "text-emerald-600";
  if (pct < -5) return "text-red-600";
  return "text-amber-600";
}

export default function CompetencyTrendsPanel({ dirCode, competencyCodes }: Props) {
  const { theme } = useTheme();
  const dk = theme === "dark";
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

  const card: React.CSSProperties = {
    background: dk ? "#0f172a" : "#fff",
    borderRadius: 8,
    border: "1px solid " + (dk ? "#334155" : "#e5e7eb"),
    padding: "12px 18px",
    marginBottom: 10,
    boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
  };

  const noData = !data || (competencyCodes && competencyCodes.length > 0
    ? !data.trends.some((c) => competencyCodes.includes(c.code))
    : data.trends.length === 0);

  if (loading) {
    return (
      <div className="text-gray-400 text-[13px] p-3">
        Загрузка трендов...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-600 text-[13px] p-3">
        {error}
      </div>
    );
  }

  if (noData) {
    return (
      <div className="text-gray-400 text-[13px] p-3">
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
      <div
        className="flex flex-wrap gap-3 mb-4"
      >
          <div style={card}>
          <div className="text-xs text-gray-500 dark:text-slate-400">Компетенций</div>
          <div className="text-[22px] font-bold text-violet-600">
            {displayed.length}
          </div>
        </div>
        <div style={card}>
          <div className="text-xs text-gray-500 dark:text-slate-400">Растут</div>
          <div className="text-[22px] font-bold text-emerald-600">
            {rising}
          </div>
        </div>
        <div style={card}>
          <div className="text-xs text-gray-500 dark:text-slate-400">Стабильно</div>
          <div className="text-[22px] font-bold text-amber-600">
            {stable}
          </div>
        </div>
        <div style={card}>
          <div className="text-xs text-gray-500 dark:text-slate-400">Падают</div>
          <div className="text-[22px] font-bold text-red-600">
            {falling}
          </div>
        </div>
        <div style={card}>
          <div className="text-xs text-gray-500 dark:text-slate-400">Средний тренд</div>
          <div
            className={`text-xl font-bold ${trendTextClass(avgTrend)}`}
          >
            {avgTrend > 0 ? "+" : ""}
            {avgTrend.toFixed(1)}%
          </div>
        </div>
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
            className={`cursor-pointer rounded px-3 py-1 text-xs ${
              filter === f.key
                ? "bg-violet-600 font-bold text-white"
                : "bg-slate-50 font-normal text-violet-600 dark:bg-slate-800"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Competency rows */}
      {filtered.map((comp) => {
        const isOpen = expanded === comp.code;
        const clsColor =
          comp.direction === "rising"
            ? "#059669"
            : comp.direction === "falling"
              ? "#dc2626"
              : "#d97706";
        return (
          <div
            key={comp.code}
            className="mb-2 overflow-hidden rounded-lg border border-slate-200 dark:border-slate-700"
          >
            <div
              onClick={() => setExpanded(isOpen ? null : comp.code)}
              className="flex cursor-pointer items-center justify-between bg-slate-50 px-4 py-2.5 dark:bg-slate-800"
            >
              <div
                className="flex items-center gap-2.5"
              >
                <span
                  className="text-sm font-semibold text-violet-600"
                >
                  {comp.code}
                </span>
                <span className="text-xs text-gray-500 dark:text-slate-400">
                  {comp.skill_count} навык{comp.skill_count !== 1 ? "а" : ""}
                  {comp.active_skills_count !== undefined &&
                    comp.active_skills_count !== comp.skill_count && (
                      <span
                        className="ml-1 text-xs text-slate-400"
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
                  className={`text-[15px] font-bold ${clsColor}`}
                >
                  {trendArrow(comp.change_pct)}{" "}
                  {comp.change_pct > 0 ? "+" : ""}
                  {comp.change_pct}%
                </span>
                <span
                  className={`inline-block rounded px-2 py-0.5 text-xs font-semibold ${clsColor} ${
                    comp.direction === "rising"
                      ? "bg-emerald-100 dark:bg-emerald-950/40"
                      : comp.direction === "falling"
                        ? "bg-red-100 dark:bg-red-950/40"
                        : "bg-amber-100 dark:bg-amber-950/40"
                  }`}
                >
                  {comp.direction}
                </span>
                <span className="text-xs text-gray-400">
                  {isOpen ? "v" : ">"}
                </span>
              </div>
            </div>

            {/* Expanded per-skill trends */}
            {isOpen && (
              <div className="px-4 pt-2 pb-3">
                {comp.skills.length === 0 && (
                  <div className="text-gray-400 text-xs">
                    Навыки не извлечены
                  </div>
                )}
                {comp.skills.map((sk, i) => (
                  <div key={i}>
                    <div
                      className="flex items-center justify-between border-b border-slate-200 py-1.5 text-[13px] dark:border-slate-700"
                    >
                      <span className="text-gray-700 dark:text-slate-300 font-medium">{sk.name}</span>
                      <span
                        className={`text-[13px] font-semibold ${trendTextClass(sk.change_pct)}`}
                      >
                        {trendArrow(sk.change_pct)}{" "}
                        {sk.change_pct > 0 ? "+" : ""}
                        {sk.change_pct}%
                        <span
                          className="ml-1.5 text-xs font-normal text-slate-400"
                        >
                          (freq: {sk.frequency})
                        </span>
                      </span>
                    </div>
                    {/* History chart */}
                    {sk.history && sk.history.length > 1 && (
                      <div
                        className="pb-4 pt-3 text-xs text-slate-500 dark:text-slate-400"
                      >
                        <div
                          className="flex h-[60px] items-end gap-1.5 py-1"
                        >
                          {[...sk.history].reverse().map((h, hi) => {
                            const maxFreq = Math.max(
                              ...sk.history!.map((x) => x.freq)
                            );
                            const pct =
                              maxFreq > 0 ? (h.freq / maxFreq) * 100 : 0;
                            return (
                              <div
                                key={hi}
                                className="flex h-full flex-1 flex-col items-center justify-end"
                              >
                                <span
                                  className="mb-0.5 text-xs font-semibold text-slate-600 dark:text-slate-400"
                                >
                                  {h.freq}
                                </span>
                                <div
                                  title={`${h.date}: ${h.freq}`}
                                  className="w-full rounded-t-sm bg-violet-600 opacity-90"
                                  style={{
                                    height: `${Math.max(pct, 2)}%`,
                                    minWidth: 16,
                                  }}
                                />
                                <span
                                  className="mt-0.5 text-xs text-slate-500 dark:text-slate-400"
                                >
                                  {h.date.slice(5)}
                                </span>
                              </div>
                            );
                          })}
                        </div>
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
