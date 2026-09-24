import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip as ReTooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Radar as RadarIcon, X, Loader2 } from "lucide-react";
import { api } from "../api";
import { useTheme } from "../../lib/theme";

const LEVELS = [
  { key: "base", label: "BASE (junior)" },
  { key: "dc", label: "DATA SCIENTIST (middle)" },
  { key: "top_dc", label: "TOP (senior)" },
] as const;

type LevelKey = (typeof LEVELS)[number]["key"];

type Drill = {
  skill: string;
  frequency: number;
  weight: number;
  category: string;
  has: Record<string, boolean>;
} | null;

export function AnalyticsCharts({ onStartGapAnalysis }: { onStartGapAnalysis?: () => void }) {
  const { theme } = useTheme();
  const dk = theme === "dark";
  const [level, setLevel] = useState<LevelKey>("base");
  const [axesCount, setAxesCount] = useState<12 | 15 | 20>(12);
  const [topSkills, setTopSkills] = useState<{ skill: string; weight: number }[]>([]);
  const [profiles, setProfiles] = useState<Record<LevelKey, Set<string>>>({
    base: new Set(),
    dc: new Set(),
    top_dc: new Set(),
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [drill, setDrill] = useState<Drill>(null);
  const [drillLoading, setDrillLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [top, ...prof] = await Promise.all([
          api("/market/top-skills?limit=50"),
          ...LEVELS.map((l) => api(`/profiles/${l.key}?full=true`).catch(() => null)),
        ]);
        if (!alive) return;
        const sets = {} as Record<LevelKey, Set<string>>;
        LEVELS.forEach((l, i) => {
          const skills: string[] = prof[i]?.skills || [];
          sets[l.key] = new Set(skills.map((s) => s.toLowerCase()));
        });
        setProfiles(sets);
        setTopSkills(top?.skills || []);
      } catch (e: any) {
        if (alive) setError(e?.message || "Ошибка загрузки");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const openDrill = async (skill: string) => {
    setDrillLoading(true);
    const has = {
      base: profiles.base.has(skill.toLowerCase()),
      dc: profiles.dc.has(skill.toLowerCase()),
      top_dc: profiles.top_dc.has(skill.toLowerCase()),
    };
    try {
      const info = await api(`/market/skill/${encodeURIComponent(skill)}`);
      setDrill({
        skill,
        frequency: info?.frequency ?? 0,
        weight: info?.weight ?? 0,
        category: info?.category ?? "–",
        has,
      });
    } catch {
      setDrill({ skill, frequency: 0, weight: 0, category: "–", has });
    } finally {
      setDrillLoading(false);
    }
  };

  const radarData = useMemo(() => {
    const axes = topSkills.slice(0, axesCount);
    const maxW = axes[0]?.weight || 1;
    const mine = profiles[level];
    return axes.map((t) => ({
      skill: t.skill,
      market: +(t.weight / maxW).toFixed(3),
      profile: mine.has(t.skill.toLowerCase()) ? +(t.weight / maxW).toFixed(3) : 0,
      has: mine.has(t.skill.toLowerCase()),
      weight: t.weight,
    }));
  }, [topSkills, profiles, level, axesCount]);

  const coverageData = useMemo(() => {
    const top50 = topSkills.slice(0, 50).map((t) => t.skill.toLowerCase());
    return LEVELS.map((l) => {
      const mine = profiles[l.key];
      const hit = top50.filter((s) => mine.has(s));
      return {
        level: l.label,
        key: l.key,
        pct: top50.length ? Math.round((hit.length / top50.length) * 100) : 0,
        hit,
        miss: top50.filter((s) => !mine.has(s)),
      };
    });
  }, [topSkills, profiles]);

  const heatRows = useMemo(() => {
    const maxW = topSkills[0]?.weight || 1;
    return topSkills.slice(0, 15).map((t) => ({
      skill: t.skill,
      weight: t.weight,
      intensity: t.weight / maxW,
      has: {
        base: profiles.base.has(t.skill.toLowerCase()),
        dc: profiles.dc.has(t.skill.toLowerCase()),
        top_dc: profiles.top_dc.has(t.skill.toLowerCase()),
      } as Record<LevelKey, boolean>,
    }));
  }, [topSkills, profiles]);

  const tick = dk ? "#94a3b8" : "#475569";
  const grid = dk ? "#1e293b" : "#e2e8f0";
  const tipStyle = {
    background: dk ? "#0f172a" : "#ffffff",
    border: `1px solid ${grid}`,
    borderRadius: 8,
    fontSize: 12,
    color: dk ? "#e2e8f0" : "#0f172a",
  };
  const LEVEL_COLORS: Record<LevelKey, string> = {
    base: "#0ea5e9",
    dc: "#6366f1",
    top_dc: "#7c3aed",
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-gray-400 dark:text-slate-500">
        <Loader2 className="size-7 animate-spin" />
      </div>
    );
  }
  if (error || topSkills.length === 0) {
    return (
      <div className="text-center py-10">
        <p className="text-sm text-gray-500 dark:text-slate-400 mb-4">
          {error || "Нет данных для графиков. Запустите gap-анализ."}
        </p>
        {onStartGapAnalysis && (
          <button
            onClick={onStartGapAnalysis}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 rounded-lg transition-colors cursor-pointer"
          >
            Запустить gap-анализ
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-9 h-9 bg-blue-600 rounded-lg">
          <RadarIcon className="size-5 text-white" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-100">Аналитические графики</h3>
          <p className="text-sm text-gray-600 dark:text-slate-400">Нажмите на точку, столбец или ячейку, чтобы увидеть детали навыка</p>
        </div>
      </div>

      <section>
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <Select value={level} onValueChange={(v) => setLevel(v as LevelKey)}>
            <SelectTrigger className="w-52 h-9 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LEVELS.map((l) => (
                <SelectItem key={l.key} value={l.key}>
                  {l.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={String(axesCount)} onValueChange={(v) => setAxesCount(Number(v) as 12 | 15 | 20)}>
            <SelectTrigger className="w-36 h-9 bg-white dark:bg-slate-950 border-gray-300 dark:border-slate-600 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[12, 15, 20].map((n) => (
                <SelectItem key={n} value={String(n)}>
                  {n} навыков
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Card className="border border-gray-200 dark:border-slate-700 shadow-sm overflow-hidden">
          <CardHeader>
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <CardTitle className="text-base">Радар: профиль vs рынок (топ-{axesCount} навыков)</CardTitle>
              <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-slate-400">
                <span className="inline-flex items-center gap-1.5">
                  <span className="size-2.5 rounded-full" style={{ background: "#8b5cf6" }} />
                  Рынок
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="size-2.5 rounded-full" style={{ background: "#2563eb" }} />
                  Профиль
                </span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-[480px]">
              <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData} outerRadius="80%" style={{ outline: "none" }}>
                  <PolarGrid stroke={grid} />
                  <PolarAngleAxis dataKey="skill" tick={{ fill: tick, fontSize: 11 }} />
                  {/* Шкала 0..1 фиксирована; цифры скрыты, чтобы не липнуть к подписям осей */}
                  <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                  <ReTooltip
                    contentStyle={tipStyle}
                    cursor={{ stroke: tick, strokeWidth: 1, fill: "transparent" }}
                    formatter={(_value: any, name: any, props: any) => {
                      const p = props?.payload;
                      if (!p) return [_value, name];
                      return [
                        `${p.has ? "в профиле" : "нет в профиле"} · вес ${p.weight}`,
                        name === "market" ? "Рынок" : "Профиль",
                      ];
                    }}
                    labelFormatter={(label) => String(label)}
                  />
                  <Radar name="market" dataKey="market" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.15} strokeWidth={2} />
                  <Radar
                    name="profile"
                    dataKey="profile"
                    stroke="#2563eb"
                    fill="#2563eb"
                    fillOpacity={0.35}
                    strokeWidth={2}
                    dot={{ r: 3, fill: "#2563eb", strokeWidth: 0, cursor: "pointer" }}
                    activeDot={{
                      r: 5,
                      fill: "#2563eb",
                      stroke: dk ? "#38bdf8" : "#1e3a8a",
                      strokeWidth: 2,
                      cursor: "pointer",
                      onClick: (_e: any, payload: any) => payload?.payload?.skill && openDrill(payload.payload.skill),
                    } as any}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </section>

      <section>
        <Card className="border border-gray-200 dark:border-slate-700 shadow-sm overflow-hidden">
          <CardHeader>
            <CardTitle className="text-base">Покрытие топ-50 навыков рынка по уровням</CardTitle>
            <CardDescription>Доля топовых рыночных навыков в эталонном профиле. Клик по столбцу покажет списки</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={coverageData} layout="vertical" margin={{ left: 8, right: 24 }}>
                  <CartesianGrid stroke={grid} strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" domain={[0, 100]} tick={{ fill: tick, fontSize: 11 }} unit="%" />
                  <YAxis type="category" dataKey="level" tick={{ fill: tick, fontSize: 12 }} width={170} />
                  <ReTooltip
                    contentStyle={tipStyle}
                    cursor={{ fill: dk ? "rgba(148, 163, 184, 0.12)" : "rgba(100, 116, 139, 0.12)" }}
                    formatter={(value: any) => [`${value}%`, "Покрытие"]}
                  />
                  <Bar
                    dataKey="pct"
                    radius={[0, 6, 6, 0]}
                    cursor="pointer"
                    onClick={(data: any) =>
                      data?.payload &&
                      setDrill({
                        skill: `__coverage__${data.payload.key}`,
                        frequency: 0,
                        weight: 0,
                        category: "",
                        has: { base: false, dc: false, top_dc: false },
                      })
                    }
                  >
                    {coverageData.map((c) => (
                      <Cell key={c.key} fill={LEVEL_COLORS[c.key]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </section>

      <section>
        <Card className="border border-gray-200 dark:border-slate-700 shadow-sm overflow-hidden">
          <CardHeader>
            <CardTitle className="text-base">Тепловая карта: топ-15 навыков × уровни</CardTitle>
            <CardDescription>Яркость по рыночному весу, зелёный навык есть в профиле. Клик по ячейке откроет детали</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="text-left font-medium text-gray-500 dark:text-slate-400 p-2 min-w-36">Навык</th>
                    {LEVELS.map((l) => (
                      <th key={l.key} className="font-medium text-gray-500 dark:text-slate-400 p-2 min-w-28">{l.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {heatRows.map((row) => (
                    <tr key={row.skill} className="border-t border-gray-100 dark:border-slate-800">
                      <td className="p-2 font-medium text-gray-800 dark:text-slate-200">{row.skill}</td>
                      {LEVELS.map((l) => {
                        const has = row.has[l.key];
                        const a = 0.12 + row.intensity * 0.55;
                        return (
                          <td key={l.key} className="p-1">
                            <button
                              onClick={() => openDrill(row.skill)}
                              title={`${row.skill}: ${has ? "есть" : "нет"} · вес ${row.weight}`}
                              className="w-full rounded-md px-2 py-2 font-mono cursor-pointer transition-transform hover:scale-[1.03]"
                              style={{
                                background: has
                                  ? `rgba(16, 185, 129, ${a})`
                                  : dk
                                    ? "rgba(30, 41, 59, 0.9)"
                                    : "rgba(241, 245, 249, 0.9)",
                                color: has ? (dk ? "#a7f3d0" : "#065f46") : tick,
                                border: `1px solid ${has ? "rgba(16,185,129,0.4)" : grid}`,
                              }}
                            >
                              {has ? "●" : "○"}
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </section>

      <AnimatePresence>
        {drill && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4"
            onClick={() => setDrill(null)}
          >
            <motion.div
              initial={{ scale: 0.96, y: 8 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.96, y: 8 }}
              transition={{ type: "tween", duration: 0.15, ease: "easeOut" }}
              className="w-full max-w-md rounded-xl bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-700 shadow-2xl p-6"
              onClick={(e) => e.stopPropagation()}
            >
              {drill.skill.startsWith("__coverage__") ? (
                <CoverageDetail
                  entry={coverageData.find((c) => `__coverage__${c.key}` === drill.skill)}
                  onClose={() => setDrill(null)}
                />
              ) : (
                <>
                  <div className="flex items-start justify-between gap-3 mb-4">
                    <div>
                      <h4 className="text-lg font-bold text-gray-900 dark:text-slate-100">{drill.skill}</h4>
                      <p className="text-xs text-gray-500 dark:text-slate-400">{drill.category}</p>
                    </div>
                    <button
                      onClick={() => setDrill(null)}
                      className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 dark:hover:text-slate-200 hover:bg-gray-100 dark:hover:bg-slate-800 cursor-pointer"
                      aria-label="Закрыть"
                    >
                      <X className="size-4" />
                    </button>
                  </div>
                  {drillLoading ? (
                    <p className="text-sm text-gray-400">Загрузка...</p>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 gap-3 mb-4 text-center">
                        <div className="rounded-lg bg-gray-50 dark:bg-slate-950/60 p-3">
                          <div className="text-xl font-bold text-gray-900 dark:text-slate-100">{drill.frequency}</div>
                          <div className="text-xs text-gray-500 dark:text-slate-400">вакансий</div>
                        </div>
                        <div className="rounded-lg bg-gray-50 dark:bg-slate-950/60 p-3">
                          <div className="text-xl font-bold text-gray-900 dark:text-slate-100">{drill.weight}</div>
                          <div className="text-xs text-gray-500 dark:text-slate-400">рыночный вес</div>
                        </div>
                      </div>
                      <div className="space-y-2">
                        {LEVELS.map((l) => (
                          <div key={l.key} className="flex items-center justify-between text-sm rounded-lg border border-gray-200 dark:border-slate-700 px-3 py-2">
                            <span className="text-gray-700 dark:text-slate-300">{l.label}</span>
                            <span
                              className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                                drill.has[l.key]
                                  ? "bg-emerald-100 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300"
                                  : "bg-gray-100 dark:bg-slate-800 text-gray-500 dark:text-slate-400"
                              }`}
                            >
                              {drill.has[l.key] ? "в профиле" : "нет"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function CoverageDetail({ entry, onClose }: { entry: any; onClose: () => void }) {
  if (!entry) return null;
  return (
    <>
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <h4 className="text-lg font-bold text-gray-900 dark:text-slate-100">{entry.level}</h4>
          <p className="text-xs text-gray-500 dark:text-slate-400">Покрытие топ-50: {entry.pct}%</p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 dark:hover:text-slate-200 hover:bg-gray-100 dark:hover:bg-slate-800 cursor-pointer"
          aria-label="Закрыть"
        >
          <X className="size-4" />
        </button>
      </div>
      <div className="grid grid-cols-2 gap-3 max-h-80 overflow-y-auto">
        <div>
          <p className="text-xs font-semibold text-emerald-700 dark:text-emerald-300 mb-2">Есть ({entry.hit.length})</p>
          <div className="flex flex-wrap gap-1.5">
            {entry.hit.map((s: string) => (
              <span key={s} className="px-2 py-0.5 text-xs rounded-full bg-emerald-100 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300">{s}</span>
            ))}
          </div>
        </div>
        <div>
          <p className="text-xs font-semibold text-gray-500 dark:text-slate-400 mb-2">Нет ({entry.miss.length})</p>
          <div className="flex flex-wrap gap-1.5">
            {entry.miss.map((s: string) => (
              <span key={s} className="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-slate-800 text-gray-500 dark:text-slate-400">{s}</span>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
