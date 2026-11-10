import { useFocusTrap } from "../../lib/a11y/useFocusTrap";
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
import { Card, CardAction, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Alert, AlertDescription } from "./ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Radar as RadarIcon, X, Loader2, AlertTriangle, ChevronDown } from "lucide-react";
import { api } from "../api";
import { useTheme } from "../../lib/theme";
import { profileLabel } from "../../lib/profiles";
import { Meter } from "../../design";
import { cn } from "./ui/utils";

const LEVELS = [
  { key: "base", label: profileLabel("base") },
  { key: "dc", label: profileLabel("dc") },
  { key: "top_dc", label: profileLabel("top_dc") },
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
  const [profileError, setProfileError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [drill, setDrill] = useState<Drill>(null);
  const drillTrapRef = useFocusTrap<HTMLDivElement>(drill !== null, () => setDrill(null));
  const [drillLoading, setDrillLoading] = useState(false);
  // Блок покрытия по умолчанию свёрнут в компактный свод: диаграмма раскрывается по клику.
  const [coverageOpen, setCoverageOpen] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [top, ...prof] = await Promise.all([
          api<{ skills: { skill: string; weight: number }[] }>("/market/top-skills?limit=50"),
          ...LEVELS.map((l) => api<{ skills: string[] }>(`/profiles/${l.key}?full=true`).catch(() => null)),
        ]);
        if (!alive) return;
        const sets = {} as Record<LevelKey, Set<string>>;
        const failed: string[] = [];
        LEVELS.forEach((l, i) => {
          const skills: string[] = prof[i]?.skills || [];
          if (!prof[i] || skills.length === 0) failed.push(l.key);
          sets[l.key] = new Set(skills.map((s) => s.toLowerCase()));
        });
        if (failed.length > 0) {
          setProfileError(`Профили без навыков: ${failed.join(", ")}. Перезапустите бэкенд и обновите страницу (Ctrl+F5).`);
        }
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
      const info = await api<{ frequency?: number; weight?: number; category?: string }>(`/market/skill/${encodeURIComponent(skill)}`);
      setDrill({
        skill,
        frequency: info?.frequency ?? 0,
        weight: info?.weight ?? 0,
        category: info?.category ?? "-",
        has,
      });
    } catch {
      setDrill({ skill, frequency: 0, weight: 0, category: "-", has });
    } finally {
      setDrillLoading(false);
    }
  };

  // Открыть детали покрытия уровня (списки «есть/нет»), как при клике по столбцу диаграммы.
  const openCoverageDrill = (key: LevelKey) => {
    setDrill({
      skill: `__coverage__${key}`,
      frequency: 0,
      weight: 0,
      category: "",
      has: { base: false, dc: false, top_dc: false },
    });
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

  const tick = "var(--muted-foreground)";
  const grid = "var(--chart-grid)";
  const tipStyle = {
    background: "var(--popover)",
    border: `1px solid ${grid}`,
    borderRadius: 8,
    fontSize: 12,
    color: "var(--popover-foreground)",
  };
const LEVEL_COLORS: Record<LevelKey, string> = {
  base: "var(--chart-1)",
  dc: "var(--chart-2)",
  top_dc: "var(--chart-3)",
};

// Классы-токены для тех же уровней (вместо inline background).
const LEVEL_BG: Record<LevelKey, string> = {
  base: "bg-chart-1",
  dc: "bg-chart-2",
  top_dc: "bg-chart-3",
};

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-subtle-foreground">
        <Loader2 className="size-7 animate-spin" />
      </div>
    );
  }
  if (error || topSkills.length === 0) {
    return (
      <div className="text-center py-10">
        <p className="text-sm text-muted-foreground mb-4">
          {error || "Нет данных для графиков. Запустите gap-анализ."}
        </p>
        {onStartGapAnalysis && (
          <button
            onClick={onStartGapAnalysis}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-primary-foreground bg-warning hover:bg-warning/90 rounded-lg transition-colors cursor-pointer"
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
        <div className="flex items-center justify-center w-9 h-9 bg-primary rounded-lg">
          <RadarIcon className="size-5 text-primary-foreground" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">Аналитические графики</h3>
          <p className="text-sm text-muted-foreground">Нажмите на точку, столбец или ячейку, чтобы увидеть детали навыка</p>
        </div>
      </div>

      <section>
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <Select value={level} onValueChange={(v) => setLevel(v as LevelKey)}>
            <SelectTrigger className="w-52 h-9 bg-input-background border-input text-xs">
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
            <SelectTrigger className="w-36 h-9 bg-input-background border-input text-xs">
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
        <Card className="border border-border shadow-sm overflow-hidden">
          <CardHeader>
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <CardTitle className="text-base">Радар: профиль vs рынок (топ-{axesCount} навыков)</CardTitle>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1.5">
                  <span className="size-2.5 rounded-full bg-[var(--chart-5)]"  />
                  Рынок
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="size-2.5 rounded-full bg-[var(--chart-1)]"  />
                  Профиль
                </span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-[480px]">
              <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData} outerRadius="80%" className="outline-none">
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
                        `${p.has ? "в профиле" : "нет в профиле"} · ${(p.market * 100).toFixed(0)}% от топ-навыка`,
                        name === "market" ? "Рынок" : "Профиль",
                      ];
                    }}
                    labelFormatter={(label) => String(label)}
                  />
                  <Radar name="market" dataKey="market" stroke="var(--chart-5)" fill="var(--chart-5)" fillOpacity={0.15} strokeWidth={2} />
                  <Radar
                    name="profile"
                    dataKey="profile"
                    stroke="var(--chart-1)"
                    fill="var(--chart-1)"
                    fillOpacity={0.35}
                    strokeWidth={2}
                    dot={(p: any) => (
                      <circle key={`dot-${p?.payload?.skill ?? p?.index ?? "x"}`} cx={p?.cx} cy={p?.cy} r={3} fill="var(--chart-1)" strokeWidth={0} cursor="pointer"
                        onClick={() => p?.payload?.skill && openDrill(p.payload.skill)} />
                    )}
                    activeDot={{
                      r: 5,
                      fill: "var(--chart-1)",
                      stroke: "var(--chart-1)",
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
        <Card className="border border-border shadow-sm overflow-hidden">
          <CardHeader>
            <CardTitle className="text-base">Покрытие топ-50 навыков рынка по уровням</CardTitle>
            <CardDescription>
              {coverageOpen
                ? "Доля топовых рыночных навыков в эталонном профиле. Клик по столбцу покажет списки"
                : "Доля топовых рыночных навыков в эталонном профиле. Клик по уровню откроет списки"}
            </CardDescription>
            <CardAction>
              <button
                type="button"
                onClick={() => setCoverageOpen((v) => !v)}
                aria-expanded={coverageOpen}
                aria-controls="coverage-chart"
                className="inline-flex items-center gap-1.5 rounded-full border border-border-strong bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground shadow-sm transition-all hover:border-primary hover:text-primary cursor-pointer"
              >
                <ChevronDown
                  className={`size-3.5 transition-transform duration-200 ${coverageOpen ? "rotate-180" : ""}`}
                  aria-hidden="true"
                />
                {coverageOpen ? "Свернуть" : "Показать диаграмму"}
              </button>
            </CardAction>
          {profileError && (
            <Alert className="mt-2 border-warning/30 bg-warning-soft text-warning">
              <AlertTriangle className="size-4 shrink-0" aria-hidden="true" />
              <AlertDescription className="text-warning">{profileError}</AlertDescription>
            </Alert>
          )}
          </CardHeader>
          <CardContent>
            {coverageOpen ? (
              <div id="coverage-chart" className="h-64">
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
                      onClick={(data: any) => data?.payload && openCoverageDrill(data.payload.key)}
                    >
                      {coverageData.map((c) => (
                        <Cell key={c.key} fill={LEVEL_COLORS[c.key]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="flex flex-col gap-2 lg:flex-row" aria-label="Сводка покрытия по уровням">
                {coverageData.map((c) => (
                  <button
                    type="button"
                    key={c.key}
                    onClick={() => openCoverageDrill(c.key)}
                    aria-label={`${c.level}: покрытие ${c.pct}% топ-50, открыть списки навыков`}
                    title={`Покрытие топ-50: ${c.pct}%. Клик покажет списки навыков`}
                    className="group flex min-w-0 flex-1 items-center gap-2.5 rounded-lg border border-border px-3 py-2 text-left transition-colors hover:border-primary cursor-pointer"
                  >
                    <span className="flex min-w-0 items-center gap-1.5 text-xs text-foreground">
                      <span
                        className={cn("size-2 shrink-0 rounded-full", LEVEL_BG[c.key])}
                        aria-hidden="true"
                      />
                      <span className="truncate">{c.level}</span>
                    </span>
                    <span className="h-1.5 min-w-10 flex-1 shrink" aria-hidden="true">
                      <Meter value={c.pct} className="h-full" fillClassName={LEVEL_BG[c.key]} />
                    </span>
                    <span className="shrink-0 text-xs font-semibold tabular-nums text-foreground">
                      {c.pct}%
                    </span>
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      <section>
        <Card className="border border-border shadow-sm overflow-hidden">
          <CardHeader>
            <CardTitle className="text-base">Тепловая карта: топ-15 навыков × уровни</CardTitle>
            <CardDescription>Яркость по рыночному весу, зелёный навык есть в профиле. Клик по ячейке откроет детали</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="text-left font-medium text-muted-foreground p-2 min-w-36">Навык</th>
                    {LEVELS.map((l) => (
                      <th key={l.key} className="font-medium text-muted-foreground p-2 min-w-28">{l.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {heatRows.map((row) => (
                    <tr key={row.skill} className="border-t border-border">
                      <td className="p-2 font-medium text-foreground">{row.skill}</td>
                      {LEVELS.map((l) => {
                        const has = row.has[l.key];
                        const a = 0.12 + row.intensity * 0.55;
                        return (
                          <td key={l.key} className="p-1">
                            <button
                              onClick={() => openDrill(row.skill)}
                              title={`${row.skill}: ${has ? "есть" : "нет"} · ${(row.intensity * 100).toFixed(0)}% от топ-навыка`}
                              className={cn(
                                "relative w-full rounded-md border px-2 py-2 font-mono cursor-pointer transition-transform hover:scale-[1.03]",
                                has
                                  ? "border-success/40 text-success"
                                  : "border-border bg-muted/60 dark:bg-muted/90 text-muted-foreground"
                              )}
                            >
                              {has && (
                                <motion.span
                                  className="pointer-events-none absolute inset-0 rounded-md bg-success"
                                  initial={false}
                                  animate={{ opacity: a }}
                                  aria-hidden="true"
                                />
                              )}
                              <span className="relative">{has ? "●" : "○"}</span>
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
            className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/50 p-4"
            onClick={() => setDrill(null)}
          >
            <motion.div
              ref={drillTrapRef}
              role="dialog"
              aria-modal="true"
              tabIndex={-1}
              initial={{ scale: 0.96, y: 8 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.96, y: 8 }}
              transition={{ type: "tween", duration: 0.15, ease: "easeOut" }}
              className="w-full max-w-md rounded-xl bg-card border border-border shadow-sm p-6"
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
                      <h4 className="text-lg font-bold text-foreground">{drill.skill}</h4>
                      <p className="text-xs text-muted-foreground">{drill.category}</p>
                    </div>
                    <button
                      onClick={() => setDrill(null)}
                      className="p-1.5 rounded-lg text-subtle-foreground hover:text-foreground hover:bg-muted cursor-pointer"
                      aria-label="Закрыть"
                    >
                      <X className="size-4" />
                    </button>
                  </div>
                  {drillLoading ? (
                    <p className="text-sm text-subtle-foreground">Загрузка...</p>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 gap-3 mb-4 text-center">
                        <div className="rounded-lg bg-muted p-3">
                          <div className="text-xl font-bold text-foreground">{drill.frequency}</div>
                          <div className="text-xs text-muted-foreground">вакансий</div>
                        </div>
                        <div className="rounded-lg bg-muted p-3">
                          <div className="text-xl font-bold text-foreground">{drill.weight}</div>
                          <div className="text-xs text-muted-foreground">рыночный вес</div>
                        </div>
                      </div>
                      <div className="space-y-2">
                        {LEVELS.map((l) => (
                          <div key={l.key} className="flex items-center justify-between text-sm rounded-lg border border-border px-3 py-2">
                            <span className="text-foreground">{l.label}</span>
                            <span
                              className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                                drill.has[l.key]
                                  ? "bg-success-soft text-success"
                                  : "bg-muted text-muted-foreground"
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
          <h4 className="text-lg font-bold text-foreground">{entry.level}</h4>
          <p className="text-xs text-muted-foreground">Покрытие топ-50: {entry.pct}%</p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-subtle-foreground hover:text-foreground hover:bg-muted cursor-pointer"
          aria-label="Закрыть"
        >
          <X className="size-4" />
        </button>
      </div>
      <div className="grid grid-cols-2 gap-3 max-h-80 overflow-y-auto">
        <div>
          <p className="text-xs font-semibold text-success mb-2">Есть ({entry.hit.length})</p>
          <div className="flex flex-wrap gap-1.5">
            {entry.hit.map((s: string) => (
              <span key={s} className="px-2 py-0.5 text-xs rounded-full bg-success-soft text-success">{s}</span>
            ))}
          </div>
        </div>
        <div>
          <p className="text-xs font-semibold text-muted-foreground mb-2">Нет ({entry.miss.length})</p>
          <div className="flex flex-wrap gap-1.5">
            {entry.miss.map((s: string) => (
              <span key={s} className="px-2 py-0.5 text-xs rounded-full bg-muted text-muted-foreground">{s}</span>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
