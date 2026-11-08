import { useMemo, useState } from "react";
import { Award, BarChart3, ChevronDown, ChevronUp } from "lucide-react";
import { MetricPair } from "../../design";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { GapBreakdown } from "./ui/gap-metrics";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";

interface MetricDef {
  key: string;
  label: string;
  hint: string;
}

const METRICS: MetricDef[] = [
  { key: "market_coverage_score", label: "Покрытие рынка", hint: "Сколько требований рынка закрывает профиль: 60% навыки + 40% домены" },
  { key: "readiness_score", label: "Готовность", hint: "Покрытие рынка с поправкой на силу навыков: +за сильные, −за слабые" },
  { key: "market_skill_coverage", label: "Навыки (строгое)", hint: "Сколько рыночных навыков у вас есть, без скидок" },
  { key: "skill_coverage", label: "Навыки (взвешенные)", hint: "Сильный навык идёт целиком, слабый — наполовину; вес — по спросу" },
  { key: "domain_coverage_score", label: "Покрытие доменов", hint: "Охват профессиональных областей" },
  { key: "profession_coverage", label: "Покрытие профессии", hint: "Совпадение с целевой профессией" },
  { key: "avg_gap", label: "Средний разрыв", hint: "Насколько в среднем не хватает до требуемого уровня" },
];

function rowVisible(key: string, evals: Record<string, EvalEntry>, profiles: string[]): boolean {
  const vals = profiles.map((p) => evals[p]?.[key as keyof EvalEntry]);
  if (vals.every((v) => v === undefined || v === null)) return false;
  // profession_coverage без фокуса всегда 0 - строку не показываем.
  if (key === "profession_coverage" && vals.every((v) => v === 0)) return false;
  return true;
}

interface GapDetail {
  skill?: string;
  gap_j?: number;
  gap_m?: number;
  gap_s?: number;
  demand_j?: number;
  demand_m?: number;
  demand_s?: number;
  cluster_relevance?: number;
  user_level?: number;
  importance?: number;
  category?: string;
}

interface EvalEntry {
  market_coverage_score?: number;
  skill_coverage?: number;
  readiness_score?: number;
  domain_coverage_score?: number;
  profession_coverage?: number;
  market_skill_coverage?: number;
  coverage_strict?: number;
  coverage_weighted?: number;
  coverage_strict_scope?: string;
  avg_gap?: number;
  match_score?: number;
  target_profession?: string;
  dominant_domain_name?: string;
  closest_roles?: Array<{ role: string; semantic_similarity?: number }>;
  gaps?: Record<string, GapDetail>;
}

interface SummaryReportProps {
  data: { evaluations?: Record<string, EvalEntry>; profiles?: string[] };
}

function scoreColor(v: number, isGap = false) {
  if (isGap) {
    if (v < 30) return "text-green-600 dark:text-green-400";
    if (v < 60) return "text-orange-500 dark:text-orange-400";
    return "text-red-500 dark:text-red-400";
  }
  if (v >= 60) return "text-green-600 dark:text-green-400";
  if (v >= 30) return "text-orange-500 dark:text-orange-400";
  return "text-red-500 dark:text-red-400";
}

function scoreCell(v: number | undefined, isGap: boolean) {
  if (v === undefined || v === null) return <span className="text-gray-400 dark:text-slate-500">-</span>;
  // avg_gap уже в процентах 0-100 (бэкенд), gap_j в GapRow - доля 0..1.
  const fixed = `${v.toFixed(1)}%`;
  return <span className={`font-mono font-semibold ${scoreColor(v, isGap)}`}>{fixed}</span>;
}

function GapRow({ skill, entry }: { skill: string; entry: GapDetail }) {
  // «Откуда взято» показываем только по реально пришедшим полям payload
  // (gaps в summary = asdict(SkillMetrics)): ничего не изобретаем.
  const hasOrigin =
    entry.gap_m != null || entry.gap_s != null ||
    entry.demand_j != null || entry.demand_m != null || entry.demand_s != null ||
    entry.user_level != null || entry.cluster_relevance != null;
  return (
    <div className="py-1.5 border-b border-gray-100 dark:border-slate-800 last:border-0">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-medium text-gray-800 dark:text-slate-200 truncate">{entry.skill || skill}</span>
          {entry.category && (
            <Badge variant="outline" className="text-xs px-1.5 py-0">{entry.category}</Badge>
          )}
        </div>
        <div className="flex items-center gap-4 text-sm shrink-0">
          <span className="text-gray-500 dark:text-slate-400 text-xs">
            важность {((entry.importance ?? 0) * 100).toFixed(0)}%
          </span>
          <span
            className="font-mono font-semibold text-red-500 dark:text-red-400"
            title="Разрыв на уровне Junior: насколько не хватает до требуемого"
          >
            +{((entry.gap_j ?? 0) * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      {hasOrigin && (
        <details className="mt-0.5">
          <summary className="text-[11px] text-gray-400 dark:text-slate-500 cursor-pointer select-none hover:text-gray-600 dark:hover:text-slate-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded">
            откуда взято
          </summary>
          {/* gap_j и важность уже в заголовке строки («+100%», «важность …%») - в
              расшифровке не дублируем, подпись уровней строится по факту полей. */}
          <GapBreakdown
            className="mt-1.5"
            fields={{
              gapM: entry.gap_m,
              gapS: entry.gap_s,
              demandJ: entry.demand_j,
              demandM: entry.demand_m,
              demandS: entry.demand_s,
              userLevel: entry.user_level,
              cluster: entry.cluster_relevance,
            }}
          />
        </details>
      )}
    </div>
  );
}

function ProfileDetails({ name, ev }: { name: string; ev: EvalEntry }) {
  const [open, setOpen] = useState(false);
  const gaps = Object.entries(ev.gaps || {}).slice(0, 12);
  const roles = ev.closest_roles || [];

  return (
    <Card className="border-gray-200 dark:border-slate-700">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4 text-left"
      >
        <div className="flex items-center gap-3">
          <Award className="size-5 text-indigo-600" />
          <div>
            <div className="font-bold text-gray-900 dark:text-slate-100">{name}</div>
            <div className="text-xs text-gray-500 dark:text-slate-400">
              {ev.target_profession || "-"}
              {ev.dominant_domain_name ? ` · ${ev.dominant_domain_name}` : ""}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {/* Одиночный процент читался как итоговая оценка; рядом - парная
              метрика качества, но только если она реально пришла в payload. */}
          <MetricPair
            primaryValue={
              ev.market_coverage_score !== undefined
                ? `${ev.market_coverage_score.toFixed(1)}%`
                : "-"
            }
            primaryTitle="Покрытие рынка: 60% взвешенные навыки + 40% домены"
            primaryClassName={`font-mono font-bold text-lg ${scoreColor(ev.market_coverage_score ?? 0)}`}
            secondaryValue={
              ev.skill_coverage !== undefined ? `${ev.skill_coverage.toFixed(1)}%` : undefined
            }
            secondaryLabel="взвеш."
            secondaryTitle="Навыки взвешенно: сильный навык идёт целиком, слабый — наполовину; может завышать. Подробнее — в таблице метрик"
            secondaryClassName="font-mono text-gray-500 dark:text-slate-400"
          />
          {open ? <ChevronUp className="size-4 text-gray-400 dark:text-slate-500" /> : <ChevronDown className="size-4 text-gray-400 dark:text-slate-500" />}
        </div>
      </button>
      {open && (
        <div className="px-5 pb-5 space-y-4">
          {roles.length > 0 && (
            <div>
              <div className="text-xs font-medium text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-2">Ближайшие роли</div>
              <div className="flex flex-wrap gap-1.5">
                {roles.slice(0, 6).map((r, i) => (
                  <Badge key={i} variant="outline" className="text-xs">
                    {r.role}
                    {r.semantic_similarity !== undefined ? ` · ${r.semantic_similarity.toFixed(0)}%` : ""}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          {gaps.length > 0 && (
            <div>
              <div className="text-xs font-medium text-gray-500 dark:text-slate-400 uppercase tracking-wide mb-1">Ключевые разрывы</div>
              <div className="rounded-lg border border-gray-200 dark:border-slate-700 divide-y divide-gray-100">
                {gaps.map(([skill, entry]) => (
                  <GapRow key={skill} skill={skill} entry={entry} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

export function SummaryReport({ data }: SummaryReportProps) {
  const profiles = useMemo(
    () => data.profiles ?? Object.keys(data.evaluations || {}),
    [data]
  );
  const evals = data.evaluations || {};

  if (profiles.length === 0) {
    return (
      <div className="text-sm text-gray-500 dark:text-slate-400 text-center py-10">
        Нет данных для отображения.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <BarChart3 className="size-4 text-indigo-600" />
            Сравнение профилей
          </CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs text-gray-500 dark:text-slate-400">Метрика</TableHead>
                {profiles.map((p) => (
                  <TableHead key={p} className="text-xs text-gray-500 dark:text-slate-400 whitespace-nowrap">{p}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {METRICS.filter((m) => rowVisible(m.key, evals, profiles)).map((m) => (
                <TableRow key={m.key}>
                  <TableCell>
                    <div className="text-sm font-medium text-gray-800 dark:text-slate-200">{m.label}</div>
                    <div className="text-xs text-gray-400 dark:text-slate-500">{m.hint}</div>
                  </TableCell>
                  {profiles.map((p) => (
                    <TableCell key={p} className="whitespace-nowrap">
                      {scoreCell(evals[p]?.[m.key as keyof EvalEntry] as number | undefined, m.key === "avg_gap")}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {profiles.map((p) => (
          <ProfileDetails key={p} name={p} ev={evals[p] || {}} />
        ))}
      </div>
    </div>
  );
}

export default SummaryReport;
