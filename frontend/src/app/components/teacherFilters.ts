/**
 * Чистые фильтры и сортировки для преподавательского дашборда.
 *
 * Вынесены из компонента, чтобы покрывать тестами без рендера: пороги
 * («слабый охват», «нет данных») легко ломаются незаметно, когда логика
 * размазана по JSX.
 */

export type DiscFilter = "all" | "gaps" | "low" | "nodata";
export type DiscSort = "plan" | "name" | "coverage" | "gaps";
export type CompFilter = "all" | "no-ksa" | "with-ksa";

/** Минимальные структурные типы: компонент передаёт свои полные объекты. */
export type DiscLike = { name: string };
export type AnalysisRow = {
  coverage_ratio?: number;
  coverage_level?: string;
  gaps?: number;
  years?: number[];
};
export type CompLike = {
  id: string;
  code: string;
  name?: string;
  skills?: string[];
  ksa?: { knowledge?: unknown[]; abilities?: unknown[]; skills?: unknown[] } | null;
};
export type RecLike = {
  discipline_id: string;
  competency_id?: string | null;
  suggestion_type?: string;
};

/** «Слабый охват» — ниже 0.5, где цвет в списке уходит в янтарный/красный. */
export const LOW_COVERAGE = 0.5;

export function ksaCount(comp: CompLike): number {
  const ksa = comp.ksa;
  if (ksa) {
    const n = (ksa.knowledge?.length ?? 0) + (ksa.abilities?.length ?? 0) + (ksa.skills?.length ?? 0);
    if (n > 0) return n;
  }
  return comp.skills?.length ?? 0;
}

function hasNoData(a: AnalysisRow | undefined): boolean {
  return !a || a.coverage_level === "no_data";
}

export function filterDisciplines<T extends DiscLike>(
  list: T[],
  analysisByName: Map<string, AnalysisRow | undefined>,
  opts: { search?: string; year?: number; filter?: DiscFilter },
): T[] {
  const q = (opts.search ?? "").trim().toLowerCase();
  return list.filter((d) => {
    if (q && !d.name.toLowerCase().includes(q)) return false;
    if (opts.year) {
      const years = analysisByName.get(d.name)?.years;
      if (!Array.isArray(years) || !years.includes(opts.year)) return false;
    }
    const a = analysisByName.get(d.name);
    switch (opts.filter) {
      case "gaps":
        return (a?.gaps ?? 0) > 0;
      case "low":
        return !hasNoData(a) && (a?.coverage_ratio ?? 0) < LOW_COVERAGE;
      case "nodata":
        return hasNoData(a);
      default:
        return true;
    }
  });
}

/** Сколько дисциплин даст каждый фильтр в текущем контексте (поиск + год). */
export function disciplineFilterCounts(
  list: DiscLike[],
  analysisByName: Map<string, AnalysisRow | undefined>,
  opts: { search?: string; year?: number },
): Record<DiscFilter, number> {
  const base = filterDisciplines(list, analysisByName, opts);
  const counts = { all: base.length, gaps: 0, low: 0, nodata: 0 } as Record<DiscFilter, number>;
  for (const d of base) {
    const a = analysisByName.get(d.name);
    if ((a?.gaps ?? 0) > 0) counts.gaps += 1;
    if (!hasNoData(a) && (a?.coverage_ratio ?? 0) < LOW_COVERAGE) counts.low += 1;
    if (hasNoData(a)) counts.nodata += 1;
  }
  return counts;
}

export function sortDisciplines<T extends DiscLike>(
  list: T[],
  analysisByName: Map<string, AnalysisRow | undefined>,
  sort: DiscSort,
): T[] {
  const arr = [...list];
  const cov = (name: string) => analysisByName.get(name)?.coverage_ratio;
  switch (sort) {
    case "name":
      return arr.sort((a, b) => a.name.localeCompare(b.name, "ru"));
    case "coverage":
      return arr.sort((a, b) => {
        // «Нет данных» — в конец, среди остальных худший охват сверху.
        const noA = hasNoData(analysisByName.get(a.name));
        const noB = hasNoData(analysisByName.get(b.name));
        if (noA !== noB) return noA ? 1 : -1;
        return (cov(a.name) ?? 1) - (cov(b.name) ?? 1);
      });
    case "gaps":
      return arr.sort(
        (a, b) => (analysisByName.get(b.name)?.gaps ?? 0) - (analysisByName.get(a.name)?.gaps ?? 0),
      );
    default:
      return arr;
  }
}

export function filterCompetencies<T extends CompLike>(
  list: T[],
  opts: { search?: string; filter?: CompFilter },
): T[] {
  const q = (opts.search ?? "").trim().toLowerCase();
  return list.filter((c) => {
    if (q && !c.code.toLowerCase().includes(q) && !(c.name ?? "").toLowerCase().includes(q)) return false;
    const n = ksaCount(c);
    if (opts.filter === "no-ksa") return n === 0;
    if (opts.filter === "with-ksa") return n > 0;
    return true;
  });
}

export function competencyFilterCounts<T extends CompLike>(list: T[]): Record<CompFilter, number> {
  const counts = { all: list.length, "no-ksa": 0, "with-ksa": 0 } as Record<CompFilter, number>;
  for (const c of list) {
    if (ksaCount(c) === 0) counts["no-ksa"] += 1;
    else counts["with-ksa"] += 1;
  }
  return counts;
}

export function filterRecommendations<T extends RecLike>(
  list: T[],
  discipline: string,
  opts: { comp?: string; type?: string },
): T[] {
  return list.filter((r) => {
    if (discipline && r.discipline_id !== discipline) return false;
    if (opts.comp && r.competency_id !== opts.comp) return false;
    if (opts.type && r.suggestion_type !== opts.type) return false;
    return true;
  });
}
