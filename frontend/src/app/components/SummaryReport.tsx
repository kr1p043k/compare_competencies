import { useMemo, useState } from "react";
import { Award, BarChart3, ChevronDown, ChevronUp, Target, Zap } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
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
  { key: "market_coverage_score", label: "Покрытие рынка", hint: "Доля востребованных на рынке навыков" },
  { key: "skill_coverage", label: "Покрытие навыков", hint: "Навыки профиля против требований" },
  { key: "readiness_score", label: "Готовность к рынку", hint: "Совокупная готовность" },
  { key: "domain_coverage_score", label: "Покрытие доменов", hint: "Охват профессиональных доменов" },
  { key: "profession_coverage", label: "Покрытие профессии", hint: "Совпадение с целевой профессией" },
  { key: "market_skill_coverage", label: "Востребованность навыков", hint: "Рыночная востребованность" },
  { key: "avg_gap", label: "Средний разрыв", hint: "Разрыв между текущим и требуемым уровнем" },
];

interface EvalEntry {
  market_coverage_score?: number;
  skill_coverage?: number;
  readiness_score?: number;
  domain_coverage_score?: number;
  profession_coverage?: number;
  market_skill_coverage?: number;
  avg_gap?: number;
  match_score?: number;
  target_profession?: string;
  dominant_domain_name?: string;
  closest_roles?: Array<{ role: string; semantic_similarity?: number }>;
  gaps?: Record<string, { skill?: string; gap_j?: number; importance?: number; category?: string }>;
}

interface SummaryReportProps {
  data: { evaluations?: Record<string, EvalEntry>; profiles?: string[] };
}

function scoreColor(v: number) {
  if (v >= 60) return "text-green-600";
  if (v >= 30) return "text-orange-500";
  return "text-red-500";
}

function scoreCell(v: number | undefined, isGap: boolean) {
  if (v === undefined || v === null) return <span className="text-gray-400">—</span>;
  const fixed = isGap ? v.toFixed(2) : `${v.toFixed(1)}%`;
  return <span className={`font-mono font-semibold ${scoreColor(v)}`}>{fixed}</span>;
}

function GapRow({ skill, entry }: { skill: string; entry: { gap_j?: number; importance?: number; category?: string } }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-sm font-medium text-gray-800 truncate">{entry.skill || skill}</span>
        {entry.category && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">{entry.category}</Badge>
        )}
      </div>
      <div className="flex items-center gap-4 text-sm shrink-0">
        <span className="text-gray-500 text-xs">
          важность {(entry.importance ?? 0).toFixed(2)}
        </span>
        <span className="font-mono font-semibold text-red-500">
          +{(entry.gap_j ?? 0).toFixed(2)}
        </span>
      </div>
    </div>
  );
}

function ProfileDetails({ name, ev }: { name: string; ev: EvalEntry }) {
  const [open, setOpen] = useState(false);
  const gaps = Object.entries(ev.gaps || {}).slice(0, 12);
  const roles = ev.closest_roles || [];

  return (
    <Card className="border-gray-200">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4 text-left"
      >
        <div className="flex items-center gap-3">
          <Award className="size-5 text-indigo-600" />
          <div>
            <div className="font-bold text-gray-900">{name}</div>
            <div className="text-xs text-gray-500">
              {ev.target_profession || "—"}
              {ev.dominant_domain_name ? ` · ${ev.dominant_domain_name}` : ""}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className={`font-mono font-bold text-lg ${scoreColor(ev.market_coverage_score ?? 0)}`}>
            {ev.market_coverage_score !== undefined ? `${ev.market_coverage_score.toFixed(1)}%` : "—"}
          </span>
          {open ? <ChevronUp className="size-4 text-gray-400" /> : <ChevronDown className="size-4 text-gray-400" />}
        </div>
      </button>
      {open && (
        <div className="px-5 pb-5 space-y-4">
          {roles.length > 0 && (
            <div>
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Ближайшие роли</div>
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
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Ключевые разрывы</div>
              <div className="rounded-lg border border-gray-200 divide-y divide-gray-100">
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
      <div className="text-sm text-gray-500 text-center py-10">
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
                <TableHead className="text-xs text-gray-500">Метрика</TableHead>
                {profiles.map((p) => (
                  <TableHead key={p} className="text-xs text-gray-500 whitespace-nowrap">{p}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {METRICS.map((m) => (
                <TableRow key={m.key}>
                  <TableCell>
                    <div className="text-sm font-medium text-gray-800">{m.label}</div>
                    <div className="text-xs text-gray-400">{m.hint}</div>
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
